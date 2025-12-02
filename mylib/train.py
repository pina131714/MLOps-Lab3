import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow
import mlflow.pytorch
import json
import os
import argparse
import numpy as np
import random

from .data_preprocess import create_data_loaders

# --- Configuration ---
MLFLOW_TRACKING_URI = "mlruns"
MODEL_NAME = "PetImageClassifier"
DATASET_NAME = "Oxford-IIIT Pet"
NUM_CLASSES = 37 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the experiments we want to run
# Different combinations of hyperparameters to find the best model
EXPERIMENTS = [
    {"learning_rate": 0.001, "batch_size": 32, "epochs": 2, "note": "Baseline"},
    {"learning_rate": 0.0001, "batch_size": 32, "epochs": 2, "note": "Lower LR"},
    {"learning_rate": 0.001, "batch_size": 64, "epochs": 2, "note": "Larger Batch"},
    {"learning_rate": 0.01, "batch_size": 32, "epochs": 2, "note": "Higher LR"},
]

def set_reproducibility(seed):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_model(model_choice: str, num_classes: int) -> nn.Module:
    """Loads a pre-trained model and adapts its classifier layer."""
    if model_choice == 'mobilenet_v2':
        # Use IMAGENET1K_V1 weights as requested
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # 1. Freeze feature extractor parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Modify the classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")
    return model.to(DEVICE)

def train_single_run(model, criterion, optimizer, dataloaders, config, seed, run_name, class_labels):
    """Executes a single training run and logs it to MLFlow."""
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Starting Run: {run_name} ---")
        
        # Log params
        mlflow.log_params({
            "seed": seed,
            "model_name": model.__class__.__name__,
            "architecture": "mobilenet_v2",
            "dataset_name": DATASET_NAME,
            "num_classes": NUM_CLASSES,
            "batch_size": config['batch_size'],
            "num_epochs": config['epochs'],
            "learning_rate": config['learning_rate'],
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "note": config.get('note', '')
        })
        
        # Log class labels
        class_labels_path = "class_labels.json"
        with open(class_labels_path, 'w', encoding="utf-8") as f:
            json.dump(class_labels, f)
        mlflow.log_artifact(class_labels_path)
        if os.path.exists(class_labels_path):
            os.remove(class_labels_path)

        # Training Loop
        best_val_accuracy = 0.0

        for epoch in range(config['epochs']):
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc.item(), step=epoch)

                if phase == 'val' and epoch_acc > best_val_accuracy:
                    best_val_accuracy = epoch_acc
                    
        print(f"Run finished. Best Val Acc: {best_val_accuracy:.4f}")
        mlflow.log_metric("final_val_accuracy", best_val_accuracy.item())
        
        # Register Model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        print(f"Model registered as: {MODEL_NAME}")

def main(args):
    """Orchestrates the multiple experiments."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)
    
    set_reproducibility(args.seed)
    
    print(f"Starting execution of {len(EXPERIMENTS)} experiments...")

    for i, config in enumerate(EXPERIMENTS):
        # 1. Prepare Data Loaders (Re-create to ensure clean state if batch size changes)
        train_loader, val_loader, class_labels = create_data_loaders(
            batch_size=config['batch_size'], 
            seed=args.seed
        )
        dataloaders = {'train': train_loader, 'val': val_loader}
        
        # 2. Create a fresh model instance for each run
        model = create_model("mobilenet_v2", NUM_CLASSES)
        
        # 3. Setup Optimizer
        criterion = nn.CrossEntropyLoss()
        # Optimizer only on the trainable parameters (the classifier head)
        optimizer = optim.Adam(model.classifier.parameters(), lr=config['learning_rate'])
        
        # 4. Generate a descriptive run name
        run_name = f"Run{i+1}_LR{config['learning_rate']}_BS{config['batch_size']}_{config['note'].replace(' ', '')}"
        
        # 5. Execute Training
        train_single_run(
            model, criterion, optimizer, dataloaders, 
            config, args.seed, run_name, class_labels
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Transfer Learning Experiments")
    parser.add_argument("--experiment_name", type=str, default="PetImageClassifier", help="Name of the MLFlow experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    main(parser.parse_args())
