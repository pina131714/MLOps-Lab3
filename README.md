[![CICD](https://github.com/pina131714/MLOps-Lab2/actions/workflows/CICD.yml/badge.svg)](https://github.com/pina131714/MLOps-Lab2/actions/workflows/CICD.yml)

# MLOps-Lab2: Image Processing CI/CD Pipeline
## Project Overview
This repository continues the work from Lab 1 and implements a Continuous Integration and Continuous Delivery (CI/CD) pipeline. The primary goal of Lab 2 is to deploy the image prediction application into production environments.
The project is containerized and deployed in two key locations:
1. API Backend: Containerized with Docker and deployed to Render.
2. GUI Frontend: A Gradio application deployed to Hugging Face Spaces.

### Key Changes from Lab 1:
- Dockerization: Project includes a multi-stage `Dockerfile`.
- Deployment: Added jobs for pushing the Docker image to Docker Hub and triggering Render deployment.
- Gradio GUI: A separate `hf-space` branch contains the Gradio application (`app.py` and `requirements.txt`).
- CI/CD Workflow: The pipeline now manages testing, Docker building, and deployment across multiple platforms.

## Project Structure
The project maintains the core application structure while adding deployment assets:

```text
MLOps-Lab1/
├── .github/
│   └── workflows/
│       └── CICD.yml           # Full CI/CD Workflow (Replaces old CI.yml)
├── api/
│   ├── api.py                 # FastAPI application and endpoints
│   └── __init__.py
├── cli/
│   ├── cli.py                 # Click CLI interface
│   └── __init__.py
├── mylib/
│   ├── image_processor.py     # Core image processing logic (PIL, Numpy)
│   └── __init__.py
├── templates/
│   └── home.html              # API documentation home page
├── tests/
│   ├── test_api.py            # Tests for the FastAPI endpoints
│   ├── test_cli.py            # Tests for the CLI commands
│   ├── test_logic.py          # Unit tests for the core logic
│   └── __init__.py
├── Dockerfile                 # Added for containerization
├── Makefile                   # Automation scripts (install, format, lint, test)
├── pyproject.toml             # Dependency management (used by uv)
└── README.md                  # This file
```

## Setup and Installation
### Prerequisites:
You must have Git, the `uv` package manager, and Docker installed globally.

## Local Setup:
Clone the Repository (Lab 2):
```bash
git clone https://github.com/pina131714/MLOps-Lab2.git
cd MLOps-Lab2
```

Run Makefile `install` target: The `Makefile` simplifies environment setup by synchronizing the virtual environment (`.venv`) and installing all dependencies defined in `pyproject.toml`.
```bash
make install
```

This command executes: `uv sync 

## Usage
1. Command Line Interface (CLI)
The CLI still functions for local testing and debugging:

| Command | Description | Example |
| :--- | :--- | :--- |
| `predict` | Gets a random class prediction. | `uv run python -m cli.cli predict tiger.jpg` |
| `info` | Prints image metadata. | `uv run python -m cli.cli info tiger.jpg` |
| `resize` | Resizes and saves the image. | `uv run python -m cli.cli resize tiger.jpg -w 100 -h 100 -o resized.jpg` |
| `grayscale` | Converts image to grayscale. | `uv run python -m cli.cli grayscale tiger.jpg -o gray.jpg` |
| `rotate` | Rotates the image by an angle. | `uv run python -m cli.cli rotate tiger.jpg 90 -o rotated.jpg` |
| `normalize` | Normalizes pixel values to [0, 1]. | `uv run python -m cli.cli normalize tiger.jpg` |

2. Deployed Services
The primary interaction points for the application are the deployed services:
- API (Backend): Accessible via your Render public URL (e.g., `https://mlops-lab2-api.onrender.com/docs`).  
- GUI (Frontend): Accessible via your Hugging Face Space URL (e.g., `https://huggingface.co/spaces/pina131714/mlops-lab2-gui`).

## Testing
The project includes unit and integration tests for all components (logic, CLI and API). The project maintains the same testing standards from Lab 1.

### Running Tests
Use the `Makefile` to run the test suite:
```bash
make test
```
This command executes: `uv run pytest tests/ -vv --cov=mylib --cov=api --cov=cli`

## Continuous Integration / Continuous Delivery (CI/CD)
The pipeline is managed by the `CICD.yml` workflow, which ensures code quality and automatic deployment.

### CI/CD Pipeline Status
The badge above reflects the status of the CICD workflow, which encompasses all tests and deployment jobs.

### Automated Pipeline (CICD.yml)
The pipeline executes the full lifecycle of the application:
1. Build: Runs `make install`, `make format`, `make lint`, and `make test`.
2. Deploy API (`deploy` job):
   - Logs into Docker Hub using secrets.
   - Builds, tags, and pushes the new Docker image (`mlops-lab2-api:latest`).
   - Triggers the Render deployment hook.
3. Deploy GUI (`deploy-hf` job):
   - Switches to the `hf-space` branch.
   - Pushes the `app.py` and `requirements.txt` to the Hugging Face Space.

### Makefile Commands
The `Makefile` defines the necessary automation targets used by the CI/CD pipeline:

| Target | Command | Purpose |
| :--- | :--- | :--- |
| `make install` | `uv sync` | Installs/syncs all Python dependencies. |
| `make format` | `uv run black ...` | Runs the Black formatter on all source code. |
| `make lint` | `uv run pylint ...` | Runs Pylint for static analysis. |
| `make test` | `uv run pytest ...` | Executes all tests and generates coverage data. |
| `make refactor` | `make format lint` | Runs formatting followed by linting. |
| `make all` | `make install format lint test` | Executes the full CI sequence locally or in the pipeline. |
