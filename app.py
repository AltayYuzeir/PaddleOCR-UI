###################################################################################################################
##
##      Needs Python version 3.10, PaddlePaddle and PaddleX
##      conda create -n pdf2docx_paddlex_env python=3.10 
##      python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
##      pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
##      pip install paddleocr==2.10.0 pymupdf opencv-python numpy pillow python-docx streamlit albucore==0.0.16
##
##
##      in the config.toml file for streamlit add:
##      [theme]
##      base="dark"
##      primaryColor="#336699"
##
##      [server]
##      maxUploadSize = 512
##
###################################################################################################################


import streamlit as st
import fitz  # PyMuPDF
import os
import re
from docx import Document
import io
import numpy as np
from PIL import Image, ImageOps
from paddleocr import PaddleOCR
from paddlex import create_model
import json
from collections import defaultdict
import cv2  # Import OpenCV
import tempfile  # Import tempfile module
import gc
import contextlib
import psutil
import time
import shutil
from contextlib import ExitStack

st.set_page_config(page_title="PDF to Docx PaddleX/OCR", page_icon="📚")


## Add custom CSS to increase slider thickness and change colors
st.markdown(
   """
    <style>
 
div[data-baseweb="slider"] {
    padding: 12px 0 !important;
    min-height: 48px !important;
}


div[data-testid="stSlider"] label p {
    font-size: 20px !important;
    color: white !important;
    font-weight: bold !important;
}

div[data-baseweb="slider"] div[role="slider"] {
    width: 20px !important;
    height: 20px !important;
    margin-top: 0px !important;
    background-color: #cc6600 !important;
    border: 0px solid white !important;
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
}

div[data-testid="stSliderThumbValue"] p {
    font-size: 16px !important;
    color: white !important;
}

  /* Increase upload field text size */
    div[data-testid="stFileUploader"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        color: white !important;
        font-weight: bold !important;
    }



 /* Increase upload field text size */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        color: white !important;
        font-weight: bold !important;
    }


    div.stButton > button {
        background-color: #336699 !important;  /* Change to any color */
        color: white !important;              /* Text color */
        border-radius: 10px !important;       /* Rounded corners */
        border: 2px solid white !important; /* Border color */
        padding: 10px 20px !important;        /* Button padding */
        width:100%;
    }

     div.stButton > button:hover {
        background-color: #204060 !important;  /* Background color on hover */
    }


    </style>
    """,
    unsafe_allow_html=True)

def cleanup_reference(obj):
    del obj

# Function to monitor memory usage
def print_memory_usage(label=""):
    """Print current memory usage for debugging"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    st.sidebar.text(f"Memory usage ({label}): {mem:.2f} MB")

# Forced cleanup function
def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    
    # Try to clean paddle cache if available
    with contextlib.suppress(Exception):
        import paddle
        paddle.device.cuda.empty_cache()
    
    # Try to clean torch cache if available
    with contextlib.suppress(Exception):
        import torch
        torch.cuda.empty_cache()
    
    # Find and remove temp files
    tmp_dir = tempfile.gettempdir()
    count_removed = 0
    for filename in os.listdir(tmp_dir):
        if (filename.startswith('tmp') or filename.startswith('_MEI')) and (
            filename.endswith('.png') or filename.endswith('.pdf') or 
            filename.endswith('.jpg') or filename.endswith('.jpeg')
        ):
            try:
                filepath = os.path.join(tmp_dir, filename)
                if os.path.isfile(filepath):
                    os.unlink(filepath)
                    count_removed += 1
            except:
                pass
    
    if count_removed > 0:
        st.sidebar.text(f"Cleaned up {count_removed} temp files")

# Clean paddle results recursively
def clean_paddle_results(result):
    """Recursively clear paddle results to help garbage collection"""
    if isinstance(result, list):
        for item in result:
            clean_paddle_results(item)
        result.clear()
    elif isinstance(result, dict):
        for key in list(result.keys()):
            clean_paddle_results(result[key])
            if key in result:
                del result[key]
    elif hasattr(result, '__dict__'):
        for attr in list(result.__dict__.keys()):
            clean_paddle_results(getattr(result, attr))
            try:
                delattr(result, attr)
            except:
                pass

# Load PaddleX model
@st.cache_resource
def load_paddle_model():
    model_name = "PP-DocLayout-L"
    model = create_model(model_name=model_name)
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return model, ocr

def clean_text(text):
    # Remove redundant spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix period spacing
    text = re.sub(r' \.', '.', text)
    # Fix comma spacing
    text = re.sub(r' ,', ',', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', '', text)
    return text

def standard_sort_key(box):
    coords = box.get('coordinate', [0, 0, 0, 0])
    try:
        y1 = float(coords[1])  # y1 is the top-left corner's y-coordinate
        x1 = float(coords[0])  # x1 is the top-left corner's x-coordinate
        return y1, x1
    except (ValueError, TypeError):
        return float('inf'), float('inf')  # Assign a high value if conversion fails

    
def classify_box_layout(box, page_width, column_threshold_percent):
    """Classifies the box as being in Column 1, Column 2, or Monolithic based on percentage threshold."""
    coords = box.get('coordinate', [0, 0, 0, 0])
    try:
        x1 = float(coords[0])
        x2 = float(coords[2])
        box_width = x2 - x1
        box_center_x = (x1 + x2) / 2
    except (ValueError, TypeError):
        return "Invalid"  # Handle invalid boxes

    page_center_x = page_width / 2
    
    # Calculate how much the box extends into each column
    left_half_width = page_width / 2
    right_half_width = page_width / 2
    
    # Calculate overlap with each column
    left_overlap = max(0, min(left_half_width, x2) - x1)
    right_overlap = max(0, x2 - max(left_half_width, x1))
    
    # Calculate overlap percentages
    left_pct = left_overlap / box_width * 100 if box_width > 0 else 0
    right_pct = right_overlap / box_width * 100 if box_width > 0 else 0
    
    # Determine if box spans significantly into both columns
    spans_columns = left_pct > column_threshold_percent and right_pct > column_threshold_percent
    
    if spans_columns:
        return "Monolithic"
    elif box_center_x < page_center_x:
        return "Column 1"
    else:
        return "Column 2"
        
def two_column_sort_key(box, page_width, column_threshold_percent):
    """
    Sorting key for two-column layouts, considers layout and y1 value.
    """
    coords = box.get('coordinate', [0, 0, 0, 0])
    try:
        y1 = float(coords[1])  # Top edge
    except (ValueError, TypeError):
        return float('inf'), float('inf')  # Handle invalid boxes

    layout = classify_box_layout(box, page_width, column_threshold_percent)

    if layout == "Column 1":
        return 1, y1 #Sort Column 1
    elif layout == "Column 2":
        return 2, y1  # Sort Column 2
    else:
        return 0, y1 # Sort Monolithic

def remove_overlapping_boxes(boxes, overlap_threshold=95):
    """
    Remove boxes that overlap with other boxes by more than the specified threshold percentage.
    
    Args:
        boxes: List of box dictionaries, each containing 'coordinate' key with [x1, y1, x2, y2] values
        overlap_threshold: Integer between 0 and 100, representing the percentage of overlap required to remove a box
                          (e.g., 80 means 80% of the smaller box must be inside the larger box)
        
    Returns:
        List of boxes with heavily overlapping boxes removed
    """
    if not boxes:
        return []
    
    # Create a copy of boxes to avoid modifying the original during iteration
    filtered_boxes = boxes.copy()
    boxes_to_remove = set()
    
    for i, box1 in enumerate(boxes):
        if i in boxes_to_remove:
            continue
            
        coords1 = box1.get('coordinate', [0, 0, 0, 0])
        
        # Try to convert coordinates to float
        try:
            x1_1 = float(coords1[0])
            y1_1 = float(coords1[1])
            x2_1 = float(coords1[2])
            y2_1 = float(coords1[3])
        except (ValueError, TypeError) as e:
            continue  # Skip if conversion fails
            
        # Calculate area of box1
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        if area1 <= 0:
            continue  # Skip invalid boxes
            
        for j, box2 in enumerate(boxes):
            if i == j or j in boxes_to_remove:
                continue  # Skip comparing a box with itself or already marked boxes
                
            coords2 = box2.get('coordinate', [0, 0, 0, 0])
            
            # Try to convert coordinates to float
            try:
                x1_2 = float(coords2[0])
                y1_2 = float(coords2[1])
                x2_2 = float(coords2[2])
                y2_2 = float(coords2[3])
            except (ValueError, TypeError):
                continue  # Skip if conversion fails
                
            # Calculate area of box2
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            if area2 <= 0:
                continue  # Skip invalid boxes
            
            # Calculate intersection area
            x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
            y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
            intersection_area = x_overlap * y_overlap
            
            # Calculate overlap percentages relative to each box
            overlap_pct1 = intersection_area / area1 if area1 > 0 else 0
            overlap_pct2 = intersection_area / area2 if area2 > 0 else 0
            
            # Decide which box to remove based on overlap percentage and box size
            if overlap_pct1 >= overlap_threshold/100:
                # Box1 is mostly inside Box2
                if area1 <= area2:  # Remove the smaller box
                    boxes_to_remove.add(i)
                    break
            elif overlap_pct2 >= overlap_threshold/100:
                # Box2 is mostly inside Box1
                if area2 <= area1:  # Remove the smaller box
                    boxes_to_remove.add(j)
    
    # Create a new list with non-overlapping boxes
    result = [box for i, box in enumerate(filtered_boxes) if i not in boxes_to_remove]
    
    # Clean up
    del filtered_boxes
    del boxes_to_remove
    gc.collect()
    
    return result

def safe_open_image(img_bytes):
    """Safely open image and ensure cleanup"""
    with io.BytesIO(img_bytes) as img_buffer:
        with Image.open(img_buffer) as image:
            image_np = np.array(image.convert("RGB"))
    return image_np

def get_layout_results(model, image):
    """Get layout results with memory cleanup"""
    try:
        # First make a copy to avoid modifying the original
        image_copy = image.copy()
        results = list(model.predict(image_copy, batch_size=1, layout_nms=True))
        
        # Make a deep copy of the results to avoid references to model internals
        import copy
        results_copy = copy.deepcopy(results)
        
        # Force cleanup
        try:
            model._model._clear_intermediate_tensors()
        except:
            pass
        
        # Clean up copy
        del image_copy
        
        return results_copy
    except Exception as e:
        st.warning(f"Layout prediction failed: {str(e)}")
        return []

def create_fresh_ocr_engine():
    """Create a fresh OCR engine for each page to avoid memory leaks"""
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

@contextlib.contextmanager
def track_temp_objects(*objs):
    """Context manager to track and clean up temporary objects"""
    try:
        yield
    finally:
        for obj in objs:
            del obj
        gc.collect()

def extract_text_with_paddle(pdf_path, paddle_model, ocr_engine):
    document_text = []
    temp_files_created = []  # Track all temporary files created

    dpi = st.session_state.get("dpi", 300)  # Default DPI, make it configurable
    margin_size = st.session_state.get("margin_size", 20)  # Default margin size
    document_layout = st.session_state.get("document_layout", "1 Column")
    
    print_memory_usage("Before PDF processing")
    
    # Use context manager for the PDF document
    with fitz.open(pdf_path) as doc:
        
        #message_interval = len(doc) // 10 
        
        for page_num, page in enumerate(doc):
            page_status = st.empty()
            page_status.markdown(f"**Processing Page {page_num + 1} of {len(doc)}**")
            
            # Run forced cleanup every few pages
        #    if page_num % message_interval == 0 and page_num > 0:
        #        force_memory_cleanup()
        #        print_memory_usage(f"After cleanup for page {page_num}")

            # Create a fresh OCR engine for each page
            page_ocr = create_fresh_ocr_engine()

            # Render the page as an image using a context manager
            with ExitStack() as stack:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))  # Higher DPI
                img_bytes = pix.tobytes("png")
                stack.callback(lambda p=pix: p.__del__())
                stack.callback(lambda: gc.collect())
                
                # Using safe image opener
                image_np = safe_open_image(img_bytes)
                
                # Release img_bytes
                del img_bytes

                # Temporary objects to be cleaned up
                text_blocks = []
                region_images = []
                region_ocr_texts = []
                temp_images = []
                try:
                    # Create a temporary file for the image if needed
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp_img_file:
                        temp_files_created.append(tmp_img_file.name)
                        
                        # Convert image to RGB for PaddleX
                        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        
                        # Get image width from numpy array
                        image_width = image_np_rgb.shape[1] # shape is (height, width, channels)
                        
                        # Get layout results with our safe function
                        layout_results = get_layout_results(paddle_model, image_np_rgb)
                        
                        # Process boxes from layout results
                        if layout_results and isinstance(layout_results, list) and len(layout_results) > 0:
                            first_result = layout_results[0]

                            if 'boxes' in first_result:
                                boxes = first_result['boxes']
                                
                                confidence_threshold = st.session_state.get("confidence_threshold", 0.50)
                                boxes = [box for box in boxes if 'score' in box and float(box['score']) >= confidence_threshold]
                                
                                if document_layout == "2 Columns":
                                    # Apply two column sort
                                    column_threshold = st.session_state.get("column_threshold_percent", 10)
                                    boxes = sorted(boxes, key=lambda box: two_column_sort_key(box, image_width, column_threshold))

                                else:
                                    boxes = sorted(boxes, key=standard_sort_key)
                                
                                overlap_threshold = st.session_state.get("overlap_threshold", 95)
                                boxes = remove_overlapping_boxes(boxes, overlap_threshold)
                            else:
                                st.warning(f"No 'boxes' key found in the first element of layout_results.")
                                boxes = []

                            # Process each box
                            for box_index, box in enumerate(boxes):
                                with ExitStack() as box_stack:
                                    if box.get('label') in st.session_state["selected_box_types"]:
                                        coords = box.get('coordinate', [0, 0, 0, 0])

                                        try:
                                            x1 = int(float(coords[0]))
                                            y1 = int(float(coords[1]))
                                            x2 = int(float(coords[2]))
                                            y2 = int(float(coords[3]))
                                        except (ValueError, TypeError) as e:
                                            st.warning(f"Invalid coordinate format: {coords}. Skipping this box.")
                                            continue

                                        # Convert coordinates to normalized values (0-1)
                                        img_h, img_w = image_np.shape[:2]
                                        x1_norm = x1 / img_w
                                        y1_norm = y1 / img_h
                                        x2_norm = x2 / img_w
                                        y2_norm = y2 / img_h

                                        # Convert coordinates to image pixels with margin
                                        crop_x1 = max(0, int(x1 - margin_size))
                                        crop_y1 = max(0, int(y1 - margin_size))
                                        crop_x2 = min(img_w, int(x2 + margin_size))
                                        crop_y2 = min(img_h, int(y2 + margin_size))

                                        # Extract the region
                                        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
                                            # Extract text region
                                            text_region = image_np[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                                            box_stack.callback(lambda r=text_region: cleanup_reference(r))

                                            # Skip if region is too small
                                            if text_region.shape[0] < 10 or text_region.shape[1] < 10:
                                                continue
                                                
                                            # Convert the region to a PIL image for display
                                        #    region_img = Image.fromarray(text_region)
                                        #    region_images.append(region_img)  # Append the PIL image to the list
                                        #    temp_images.append(region_img) # Added to cleanup

                                            # Process with OCR using fresh instance
                                            ocr_result = None
                                            ocr_result = page_ocr.ocr(text_region, cls=True)
                                            box_stack.callback(lambda: clean_paddle_results(ocr_result) if ocr_result else None)

                                            if ocr_result and len(ocr_result) > 0:
                                                region_text = ""
                                                for line_group in ocr_result:
                                                    if isinstance(line_group, list) and len(line_group) > 0:
                                                        for line in line_group:
                                                            if len(line) == 2 and isinstance(line[0], list) and isinstance(line[1], tuple):
                                                                text = line[1][0] if len(line[1]) > 0 else ""
                                                                region_text += text + " "

                                                region_text = region_text.strip()
                                                region_ocr_texts.append(region_text)
                                                
                                                if region_text:
                                                    text_blocks.append({
                                                        'text': region_text.strip(),
                                                        'y_pos': y1_norm,
                                                        'x_pos': x1_norm
                                                    })
                                
                                    # Force garbage collection every few boxes
                                    if box_index % 5 == 0:
                                        gc.collect()
                            
                            # Clean up boxes
                            del boxes
                        
                        # Clean up layout results
                        clean_paddle_results(layout_results)
                        del layout_results
                    
                    # Join all text_blocks for this page
                    page_text = "\n\n".join([block['text'] for block in text_blocks])
                    document_text.append(page_text)

                except Exception as e:
                    # Fallback for pages where layout detection fails
                    st.warning(f"Layout processing failed on page {page_num+1}. Error: {str(e)}")

                    # Fallback to basic OCR
                    try:
                        ocr_result = page_ocr.ocr(image_np_rgb, cls=True)

                        if ocr_result is not None and len(ocr_result) > 0:
                            page_text = ""
                            for line in ocr_result:
                                if isinstance(line, list) and len(line) > 0:
                                    for text_line in line:
                                        if len(text_line) >= 2:  # Ensure both box and text are present
                                            text_info = text_line[1]
                                            if isinstance(text_info, tuple) and len(text_info) > 0:
                                                text = text_info[0]
                                                page_text += text + "\n"
                            document_text.append(clean_text(page_text))
                        else:
                            document_text.append("")  # Empty text for this page
                        
                        # Clean up OCR results
                        clean_paddle_results(ocr_result)
                        del ocr_result
                    except Exception as ocr_error:
                        st.warning(f"Fallback OCR also failed on page {page_num+1}. Error: {str(ocr_error)}")
                        document_text.append("")  # Empty text for this page
                
                # Try to manually clean up the OCR engine
                try:
                    page_ocr.det_model.cpu()
                    page_ocr.rec_model.cpu()
                    if page_ocr.use_angle_cls:
                        page_ocr.cls_model.cpu()
                    del page_ocr
                except:
                    pass
            
            #if region_images:
            #    st.subheader(f"Extracted Text Regions - Page {page_num + 1}")
            #    for i, img in enumerate(region_images):
            #        col1, col2 = st.columns(2)
            #        with col1:
            #            st.image(img, caption=f"Region from Page {page_num + 1}", use_container_width=True)
            #        with col2:
            #            st.write(f"OCR Text: {region_ocr_texts[i]}")
            
            # Clear status message
            page_status.empty()

            # Explicit cleanup for all page-related objects
            del image_np
            if 'image_np_rgb' in locals():
                del image_np_rgb
            
            # Clear collections
            if 'text_blocks' in locals() and text_blocks:
                text_blocks.clear()
                del text_blocks
            
            # Force garbage collection after each page
            gc.collect()
            
            # Add delay to allow OS to release file handles
            time.sleep(0.1)

    # Join all pages with appropriate spacing
    full_text = "\n\n".join([text for text in document_text if text.strip()])

    # Additional cleanup for better paragraph flow
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    # Clean up temporary files
    for temp_file in temp_files_created:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            pass
    
    # Final cleanup
    del document_text
    del temp_files_created
    gc.collect()
    
    print_memory_usage("After PDF processing")
    force_memory_cleanup()

    return full_text

def save_as_docx(text, file_path):
    try:
        doc = Document()
        for paragraph in text.split('\n\n'):  # Use the original cleaned text
            if paragraph.strip():
                doc.add_paragraph(paragraph.strip())

        docx_path = f"{os.path.splitext(file_path)[0]}.docx"
        doc.save(docx_path)
        
        # Clean up
        del doc
        gc.collect()
        
        return docx_path
    except Exception as e:
        st.error(f"Error saving DOCX: {e}")
        return None

def cleanup_temp_directory():
    """Clean up temporary directories that might contain leaked files"""
    tmp_dir = tempfile.gettempdir()
    
    # Remove PDF and image files from temp directory
    file_patterns = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    count = 0
    
    for pattern in file_patterns:
        for filename in os.listdir(tmp_dir):
            if filename.endswith(pattern):
                try:
                    filepath = os.path.join(tmp_dir, filename)
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                        count += 1
                except:
                    pass
    
    # Look for paddleocr temp directories and clean them
    paddle_dirs = []
    for dirname in os.listdir(tmp_dir):
        if 'paddle' in dirname.lower() or 'ocr' in dirname.lower() or '_MEI' in dirname:
            dir_path = os.path.join(tmp_dir, dirname)
            if os.path.isdir(dir_path):
                paddle_dirs.append(dir_path)
    
    # Clean up paddle temp directories
    for dir_path in paddle_dirs:
        try:
            shutil.rmtree(dir_path)
            count += 1
        except:
            pass
    
    if count > 0:
        st.sidebar.text(f"Cleaned up {count} temporary files/directories")

def main():
    st.title("📚 PDF to Docx with PaddleX/OCR")
    
    # Add a memory status display
    memory_status = st.sidebar.empty()
    
    # Function to update memory status
    def update_memory():
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # MB
        memory_status.text(f"Memory usage: {mem:.2f} MB")
    
    # Initial memory display
    update_memory()
    
    # Add a cleanup button
    #if st.sidebar.button("Force Memory Cleanup"):
    #    cleanup_temp_directory()
    #    force_memory_cleanup()
    #    update_memory()
    #    st.sidebar.success("Memory cleanup completed")

    # Load models (cached)
    with st.spinner("Loading PaddleX model..."):
        paddle_model, ocr_engine = load_paddle_model()

    col1, col2 = st.columns(2)
    with col1:
        # Layout Selection Radio Button
        st.session_state["document_layout"] = st.radio("Document Layout:", ["1 Column", "2 Columns"], 
        help ="""
    ❓ **Document Layout**    
    Select the layout that best describes your document:\n    
    * **1 Column:** Standard document with text flowing from top to bottom.    
    * **2 Columns:** Document with text arranged in two side-by-side columns.    
    
    """
        )
    
    with col2:
        #Dynamic UI Elements based on Layout Selection
        if st.session_state["document_layout"] == "2 Columns":
            st.session_state["column_threshold_percent"] = st.slider(
                "Monolithic Threshold (%)",
                0,
                50,
                10,
                1,
                help="""
    ❓ **Monolithic Threshold (%)**     
    This setting determines how much a text region can be offset from the center of the page before it is considered part of a column.    
    *   **Lower values:** Treat most text regions as belonging to either the left or right column.     
    *   **Higher values:** Classify more text regions as "Monolithic" (spanning the entire page width), such as titles, headings, or paragraphs.    
        
    """
            )
        else:
            st.session_state["column_threshold_percent"] = None  # Not used in single-column layout

    col1, col2 = st.columns(2)

    with col1:
        st.session_state["dpi"] = st.slider(
            "DPI for PDF Rendering", 
            150, 600, 300, 50,
           help="""❓ **DPI for PDF Rendering**     
             This setting controls the **Dots Per Inch (DPI)** used for rendering PDFs into images.      
             Higher DPI values result in **sharper images** but require **more memory** and **longer processing times**.                  
             \n**Recommended Settings:** **300 DPI** –  Balanced choice between quality and speed.      
            
                        """
        )
        
        st.session_state["margin_size"] = st.slider(
            "Margin Size (Pixels)", 
            0, 100, 20, 1,
            help="""❓ **Margin Size (Pixels)**     
                Extra space added around detected text regions.     
                Larger margins help capture text that might be at the edges of detected regions.     
                \n**Recommended Settings:** **20 pixels** – Balanced choice between capturing full text and avoiding foreign text.  
                
    """
        )

    with col2:
        st.session_state["confidence_threshold"] = st.slider(
            "Classification Confidence Threshold", 
            0.40, 1.00, 0.50, 0.01,
            help="""❓ **Classification Confidence Threshold**     
    Minimum confidence required for a classification to be considered valid.     
    Higher values ensure more reliable classifications but may filter out uncertain results.    
    \n**Recommended Settings:** **0.50** – Balanced choice between accuracy and retaining more potential classifications.  
    
    """
        )
        
        st.session_state["overlap_threshold"] = st.slider(
            "Box Overlap Threshold (%)", 
            50, 100, 95, 1,
            help="""❓ **Box Overlap Threshold**     
    Boxes that overlap by more than this percentage will be filtered out.     
    Higher values (closer to 100) only remove boxes that are almost completely overlapping.    
    \n**Recommended Settings:** **95** – Balanced choice between removing duplicates and preserving valid text regions.  
    
    """
        )
        
    box_types_info = {
    'text': 'Regular body text content',
    'paragraph_title': 'Titles or headings for paragraphs',
    'doc_title': 'Main document or page title',
    'abstract': 'Document abstract or summary',
    'aside_text': 'Sidebar text',
    #'image': 'Images and figures',
    'number': 'Page number and others',
    'content': 'Table of contents',
    'figure_title': 'Captions or titles for figures',
    #'formula': 'Mathematical formulas or equations',
    #'table': 'Data tables',
    'table_title': 'Captions or titles for tables',
    'reference': 'Citations or references',
    'footnote': 'Footnotes at page bottom',
    'header': 'Page headers',
    'algorithm': 'Algorithm blocks or pseudocode',
    'footer': 'Page footers',
    #'seal': 'Official seals or stamps',
    'chart_title': 'Same as figure_title - titles for charts/figures'
    #'chart': 'Charts, graphs, or diagrams',
    #'formula_number': 'Equation numbers',
    #'header_image': 'Images in page headers',
    #'footer_image': 'Images in page footers'
    }


    # Default selected box types
    default_box_types = ['text', 'paragraph_title', 'doc_title', 'abstract', 'aside_text']

    # Create checkbox group for box types
    st.markdown(
    '<p style="font-size:20px; color:white; font-weight:bold;">'
    'Select box types to be transcribed (Defaults are all of 1st column):'
    '</p>',
    unsafe_allow_html=True )
    
    col1, col2, col3 = st.columns(3)

    # Store selected box types in session state
    if "selected_box_types" not in st.session_state:
        st.session_state["selected_box_types"] = default_box_types

    # Create checkboxes in three columns for better layout
    all_box_types = list(box_types_info.keys())
    box_types_by_column = np.array_split(all_box_types, 3)
    for i, column in enumerate([col1, col2, col3]):
        with column:
            for box_type in box_types_by_column[i]:
                is_default = box_type in default_box_types
                if st.checkbox(
                    box_type, 
                    value=is_default, 
                    key=f"checkbox_{box_type}",
                    help=box_types_info[box_type]
                ):
                    if box_type not in st.session_state["selected_box_types"]:
                        st.session_state["selected_box_types"].append(box_type)
                else:
                    if box_type in st.session_state["selected_box_types"]:
                        st.session_state["selected_box_types"].remove(box_type)

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        unique_files = []
        seen_filenames = set()

        for file in uploaded_files:
            if file.name not in seen_filenames:
                unique_files.append(file)
                seen_filenames.add(file.name)
            else:
                st.warning(f"⚠️ Skipped duplicate: {file.name}")

        if unique_files:
            st.write(f"✅ {len(unique_files)} unique files uploaded. Click the button to start conversion.")

            if st.button("Start Conversion"):
                failed_files = []
                with st.spinner("Processing files..."):
                    progress_bar = st.progress(0)
                    total_files = len(unique_files)
                    status_box = st.empty()

                    # Clean up before starting
                    force_memory_cleanup()
                    cleanup_temp_directory()
                    update_memory()

                    for i, uploaded_file in enumerate(unique_files):
                        try:
                            # Create temporary file for the uploaded PDF
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                tmp_pdf.write(uploaded_file.getbuffer())
                                file_path = tmp_pdf.name

                            status_box.markdown(
                                f"""
                                <div style="font-size:1.2em; color:white; background-color:#333; padding:8px; border-radius:5px;">
                                    📄 Processing: <strong>{uploaded_file.name}</strong>
                                </div>
                                """, unsafe_allow_html=True
                            )

                            extracted_text = extract_text_with_paddle(file_path, paddle_model, ocr_engine)
                            
                            update_memory()  # Update memory display after processing
                            
                            if extracted_text:
                                output_filename = os.path.join(os.getcwd(), os.path.splitext(uploaded_file.name)[0] + ".docx")
                                docx_path = save_as_docx(extracted_text, output_filename)
                                if docx_path:
                                    st.success(f"✅ Saved: {os.path.basename(docx_path)}")
                                
                                # Clean up extracted text
                                del extracted_text
                            else:
                                failed_files.append(uploaded_file.name)

                            # Clean up temporary file
                            try:
                                os.unlink(file_path)
                            except:
                                pass
                            
                            # Force cleanup between files
                            force_memory_cleanup()
                            update_memory()

                        except Exception as e:
                            failed_files.append(uploaded_file.name)
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            st.exception(e)
                            
                            # Clean up on exception
                            if 'file_path' in locals() and os.path.exists(file_path):
                                try:
                                    os.unlink(file_path)
                                except:
                                    pass
                            
                            # Force cleanup after error
                            force_memory_cleanup()

                        progress_bar.progress((i + 1) / total_files)
                        
                        # Force garbage collection after each file
                        gc.collect()

                st.success("✅ Processing Complete!")

                if failed_files:
                    st.error("❌ Failed Extractions:")
                    st.text_area("The following files failed:", "\n".join([f"{i+1}. {name}" for i, name in enumerate(failed_files)]), height=150)
                else:
                    st.success("All files processed successfully!")
                
                # Final full cleanup
                del unique_files
                del failed_files
                gc.collect()
                force_memory_cleanup()
                cleanup_temp_directory()
                update_memory()

if __name__ == "__main__":

    main()


