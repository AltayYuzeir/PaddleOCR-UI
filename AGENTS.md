## Purpose  
This file tells coding agents how to work on this repository **safely and correctly**: where to run from, which
commands to execute, quality gates to pass, and project conventions. Agents should follow these rules unless a human
explicitly overrides them.

## Repo profile (read me first)  
- **Project root:** `llm-eng-template`  
- **Primary stack:** Python 3.12 (via `.venv`), Ollama, Docker Compose.  
- **Top modules:**  
  - `backend/` — ..., config.  
  - `frontend/` — ...
  - `scripts/` — developer helpers
  - `tests/` — `unit/`, `integration/`, `e2e/` with `conftest.py`.  
  - `docker/` — compose + Dockerfile.  
  - `reports/` — artifacts, coverage.

## Golden rules (non-negotiable)  
1) **Run commands from repo root.** If you need a shell.  
2) **Use the project venv:** run Python via `.venv/bin/python` (never the system interpreter).  
3) **Never commit secrets**; `.secrets.baseline` is enforced. If you touch secrets, stop and ask a human.  

## Setup & environment  
- Create/activate venv if missing and install deps as needed (ask before changing pinned versions).  
- Docker is available; prefer the provided scripts to start services.  
- Network calls to external services must be isolated behind Docker where possible.
 - For local CI with Act, see `docs/AI_instructions.md` for configuration and usage.

## Common tasks (agents may run these automatically)  
> Agents: prefer these exact commands and **stop on first failure**.

- - - - ## Testing & quality gates (must pass before you conclude work)  
- **Local fast path (as module):**  
  `.venv/bin/python -m pytest tests/unit -q`  
  `.venv/bin/python -m pytest tests/integration -q`  
  _Coverage outputs to `reports/coverage`._

- - **Pre-commit (mandatory) bash command:**  
  `uv run pre-commit run --all-files`

- **If UI or logs change intentionally:** update any snapshots or expectations the tests rely on.  
  _If failing, fix the code or tests; do **not** disable checks._

## Code style & conventions  
- Keep lines ≤120 chars. Prefer small, composable functions and explicit error handling.  
- Add or update tests when changing behavior.

## Makefile-first checks  
- Prefer `make` targets over ad-hoc commands for parity with CI:  
  `make pyright`, `make ruff-fix`, `make ruff-format`, `make unit`, `make integration-local`, `make pre-commit`, `make pre-push`.

## Safe edit policy  
- For risky edits (schema changes, cross-module refactors), propose a plan in the Tasklist first.  
- Never change Dockerfiles, Compose, or security-sensitive configs without clearly stating risks and mitigations.  

## Glossary (min spec for agents)  
- **Ollama:** local model runner backing embeddings/LLM endpoints.  

## Agent etiquette  
- When in doubt about security or data handling, stop and request human guidance.
