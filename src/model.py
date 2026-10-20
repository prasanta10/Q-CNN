import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_cifar_resnet(num_classes=10):
    """
    Loads a standard ResNet-18 and modifies its initial layers and final 
    classification head to work optimally with 32x32 CIFAR-10 images.
    """
    # 1. Load the base ResNet-18 architecture (untrained)
    model = resnet18(weights=None)
    
    # 2. Modify the initial Convolutional Layer
    # Replace the aggressive 7x7 stride-2 conv with a gentle 3x3 stride-1 conv.
    # This prevents the 32x32 image from shrinking to 16x16 immediately.
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 3. Bypass the initial MaxPool layer
    # The original network does another 2x2 downsampling right after conv1.
    # We replace it with an Identity function (a layer that does nothing) 
    # to preserve our spatial dimensions for the deep residual blocks.
    model.maxpool = nn.Identity()
    
    # 4. Modify the Fully Connected (Classification) Head
    # Original ResNet classifies 1000 ImageNet classes. We need 10 for CIFAR.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Sanity Check
    model = get_cifar_resnet(num_classes=10)
    
    # Create a dummy tensor mimicking a single CIFAR-10 image batch
    dummy_input = torch.randn(1, 3, 32, 32) 
    
    # Pass it through the model to ensure our dimensional math works
    output = model(dummy_input)
    
    print(f"Model successfully processed 32x32 input.")
    print(f"Output shape (Batch Size, Classes): {output.shape}") # Expect: [1, 10]