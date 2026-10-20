import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=2):
    """
    Creates and returns PyTorch DataLoader objects for CIFAR-10 train and test sets.
    """
    # 1. Define image preprocessing and augmentation pipelines
    
    # Training transforms: includes basic data augmentation to prevent overfitting
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize channels using CIFAR-10 dataset mean and std: (R, G, B)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Test transforms: strictly normalization (no random augmentations)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 2. Download/load train and test datasets from torchvision
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 3. Wrap datasets into PyTorch DataLoaders for batched processing
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # Sanity check script execution
    train_loader, test_loader = get_cifar10_loaders()
    print(f"Train dataset batches: {len(train_loader)}")
    print(f"Test dataset batches:  {len(test_loader)}")
    
    # Grab one batch to inspect batch tensor shape
    images, labels = next(iter(train_loader))
    print(f"Batch images tensor shape: {images.shape}") # Expect: [128, 3, 32, 32]
    print(f"Batch labels tensor shape: {labels.shape}") # Expect: [128]