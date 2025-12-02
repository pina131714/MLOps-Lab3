[![CICD](https://github.com/pina131714/MLOps-Lab3/actions/workflows/CICD.yml/badge.svg)](https://github.com/pina131714/MLOps-Lab3/actions/workflows/CICD.yml)

# MLOps-Lab3: Experiment Tracking and Model Versioning

## Project Overview
This repository represents the third stage of the MLOps project series. Building upon the CI/CD pipeline from Lab 2, this lab replaces the random prediction logic with a **real Deep Learning classifier** trained on the **Oxford-IIIT Pet Dataset**.

The primary objective is to implement **Experiment Tracking** and **Model Versioning** using **MLFlow**. The project trains a **MobileNetV2** model using transfer learning, tracks experiments to select the best performing model, and serializes it to **ONNX** format for efficient production inference.

### Key Changes from Lab 2:
- **Real AI Model:** Replaced random prediction with a MobileNetV2 image classifier.
- **Experiment Tracking:** Integrated **MLFlow** to log metrics, parameters, and models.
- **Model Registry:** Implemented logic to programmatic query and select the best model.
- **Serialization:** Added automated export of PyTorch models to **ONNX** format.
- **Inference Engine:** Updated the API and CLI to use `onnxruntime` for predictions.

## Project Structure
The project structure has evolved to include training and serialization scripts:

```text
MLOps-Lab3/
├── .github/
│   └── workflows/
│       └── CICD.yml           # Full CI/CD Workflow
├── api/
│   ├── api.py                 # FastAPI application
│   └── __init__.py
├── cli/
│   ├── cli.py                 # Click CLI interface
│   └── __init__.py
├── mylib/
│   ├── data_preprocess.py     # Data loading and transformation (Training phase)
│   ├── image_processor.py     # ONNX Inference logic (Production phase)
│   ├── serialize.py           # Model selection and ONNX export script
│   ├── train.py               # MLFlow training and tracking script
│   └── __init__.py
├── templates/
│   └── home.html              # API documentation home page
├── tests/
│   ├── test_api.py            # API Integration tests
│   ├── test_cli.py            # CLI Integration tests
│   ├── test_model_artifacts.py# Verifies ONNX model existence
│   └── __init__.py
├── .pylintrc                  # Pylint configuration for PyTorch/MLFlow
├── Dockerfile                 # Production container definition
├── Makefile                   # Automation scripts
├── pyproject.toml             # Dependency management
└── README.md                  # This file
```

## Setup and Installation

### Prerequisites
* **Git** installed globally.
* **`uv`** package manager installed globally.
* **Python 3.11** (Required for PyTorch compatibility in this lab).
* **Docker** installed globally.

### Local Setup
1.  **Clone the Repository (Lab 3):**
    ```bash
    git clone [https://github.com/pina131714/MLOps-Lab3.git](https://github.com/pina131714/MLOps-Lab3.git)
    cd MLOps-Lab3
    ```

2.  **Install Dependencies:**
    The project is pinned to Python 3.11. The `Makefile` handles the sync. The `Makefile` simplifies environment setup by synchronizing the virtual environment (`.venv`) and installing all dependencies defined in `pyproject.toml`.
    ```bash
    make install
    ```
    *(Executes: `uv sync`)*


## MLFlow Training Workflow (New)

Lab 3 introduces a complete ML lifecycle. To reproduce the model from scratch:

### 1. Train and Track Experiments
Run the training script to fine-tune MobileNetV2 on the Pet dataset. This script runs multiple experiments (grid search) and logs results to the local `mlruns/` directory.
```bash
uv run python -m mylib.train
```

### 2. Visualize results
Launch the MLFlow UI to inspect accuracy curves and compare run parameters.
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` in your browser.

### 3. Serialize Best Model
Run the serialization script. This queries MLFlow for the run with the highest validation accuracy and exports it to `model.onnx` and `class_labels.json`.
```bash
uv run python -m mylib.serialize
```
 

## Usage
Once `model.onnx` is generated, the standard CLI and API interfaces work exactly as before, but now provide real predictions.

### 1. Command Line Interface (CLI)

| Command | Description | Example |
| :--- | :--- | :--- |
| `predict` | Predict the breed of a pet image. | `uv run python -m cli.cli predict my_dog.jpg` |
| `info` | Prints image metadata. | `uv run python -m cli.cli info my_dog.jpg` |
| `resize` | Resizes and saves the image. | `uv run python -m cli.cli resize tiger.jpg -w 100 -h 100 -o resized.jpg` |
| `grayscale` | Converts image to grayscale. | `uv run python -m cli.cli grayscale tiger.jpg -o gray.jpg` |
| `rotate` | Rotates the image by an angle. | `uv run python -m cli.cli rotate tiger.jpg 90 -o rotated.jpg` |
| `normalize` | Normalizes pixel values to [0, 1]. | `uv run python -m cli.cli normalize my_dog.jpg` |

### 2. Deployed Services
The primary interaction points for the application are the deployed services. The application is deployed automatically via the CI/CD pipeline:
- API (Backend): Accessible via your Render public URL (e.g., `https://mlops-lab2-api.onrender.com/docs`).  
- GUI (Frontend): Accessible via your Hugging Face Space URL (e.g., `https://huggingface.co/spaces/pina131714/mlops-lab2-gui`).


## Testing
The project includes unit and integration tests for all components (logic, CLI and API). The project maintains the same testing standards from Lab 1. 
The testing now includes checks for model artifact existence (`model.onnx`).
```bash
make test
```

## Continuous Integration / Continuous Delivery (CI/CD)

The pipeline is managed by the `CICD.yml` workflow, which ensures code quality and automatic deployment.

### CI/CD Pipeline Status
The badge above reflects the status of the CICD workflow, which encompasses all tests and deployment jobs.

### Automated Pipeline
1.  **Build & Test:** Installs dependencies, runs linting/formatting, and verifies that `model.onnx` exists.
2.  **Deploy API:** Builds the Docker image (copying the ONNX model inside) and pushes to Docker Hub, then triggers Render.
3.  **Deploy GUI:** Syncs the Gradio frontend to Hugging Face Spaces.

### Makefile Commands
The `Makefile` defines the necessary automation targets used by the CI/CD pipeline:

| Target | Command | Purpose |
| :--- | :--- | :--- |
| `make install` | `uv sync` | Installs/syncs all Python dependencies. |
| `make format` | `uv run black ...` | Formats code. |
| `make lint` | `uv run pylint ...` | Lints code using `.pylintrc`. |
| `make test` | `uv run pytest ...` | Runs all tests. |
| `make refactor` | `make format lint` | Runs formatting followed by linting. |
| `make all` | ... | Runs the full CI sequence. |
