"""
Data loading and preprocessing utilities for the Oxford-IIIT Pet Dataset.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the target image size required by MobileNet_v2
IMAGE_SIZE = 224
# Standard normalization parameters for models pre-trained on ImageNet
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
# Local directory to store the downloaded dataset
DATA_DIR = "data" 


def get_data_transforms() -> dict:
    """
    Defines the standard set of transformations for training and validation data.
    
    Returns:
        dict: Dictionary containing 'train' and 'val' transforms.
    """
    return {
        # Training transforms often include random elements for data augmentation
        'train': transforms.Compose([
            # Resize must be done before RandomResizedCrop
            transforms.Resize(256), 
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
        # Validation/Test transforms are deterministic
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
    }


def create_data_loaders(batch_size: int, seed: int = 42) -> tuple:
    """
    Downloads the Oxford-IIIT Pet Dataset, creates data splits, and returns 
    the training and validation DataLoaders.

    Args:
        batch_size (int): The number of samples per batch.
        seed (int): Seed for reproducibility of the data split.
        
    Returns:
        tuple: (train_dataloader, val_dataloader, class_labels)
    """
    # 1. Define transforms
    data_transforms = get_data_transforms()
    
    # 2. Download and load the full dataset for the 'train' split
    # 'split='train' downloads and loads the training images
    full_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, 
        split='trainval', # Use the combined trainval split for the 80/20 split
        download=True, 
        transform=data_transforms['train'] # Apply augmentation initially
    )
    
    # 3. Calculate split sizes (80% training / 20% validation)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Ensure reproducibility of the split
    generator = torch.Generator().manual_seed(seed)
    
    # 4. Perform the random split
    train_data, val_data = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )

    # 5. Apply the correct validation transform to the validation set
    # Note: Torchvision datasets apply transforms on access, but since we used random_split 
    # on the full dataset, we must manually ensure the validation split uses the 'val' transform.
    # We assign the validation transform to the validation dataset object
    val_data.dataset.transform = data_transforms['val'] 

    # 6. Create DataLoaders
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=os.cpu_count(),
        worker_init_fn=lambda worker_id: random.seed(seed + worker_id) # ensure worker reproducibility
    )
    val_dataloader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=os.cpu_count()
    )
    
    # 7. Get class labels
    # Use the class names from the underlying dataset
    class_labels = full_dataset.classes
    
    return train_dataloader, val_dataloader, class_labels


# --- Debugging/Verification (optional, for local run) ---
if __name__ == "__main__":
    import torch
    import random
    
    # Test execution
    print("Starting data loading process...")
    try:
        train_loader, val_loader, labels = create_data_loaders(batch_size=32)
        print(f"Total classes: {len(labels)}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Example classes: {labels[:5]}")
        
        # Check one batch
        images, targets = next(iter(train_loader))
        print(f"Image batch shape: {images.shape}")
        print(f"Target batch shape: {targets.shape}")
        
        # Verify normalization (mean should be close to 0, std close to 1)
        print(f"Sample mean (should be ~0.0): {images[0].mean()}")
        
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        print("Ensure PyTorch and Torchvision dependencies are installed (uv sync)!")
