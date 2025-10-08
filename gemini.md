# Gemini Guidelines

- never remove production docker volumes: This means never use -v when running on non-test containers:
  docker -f docker/docker-compose.yml compose down -v
- after each major code change, run linters to check if some lint errors need manual fixing:
 ruff check . --fix 
 pyright
