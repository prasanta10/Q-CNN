import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_cifar10_loaders
from model import get_cifar_resnet

def train_baseline(epochs=10):
    # 1. Setup Device and Loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_loader, test_loader = get_cifar10_loaders(data_dir="../data")
    
    # 2. Initialize Model, Loss, and Optimizer
    model = get_cifar_resnet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print("\n--- Starting FP32 Baseline Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm gives us a nice progress bar in the terminal
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()           # Clear old gradients
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()                 # Backward pass (calculate gradients)
            optimizer.step()                # Update weights
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
        
    # 4. Save the trained FP32 model
    os.makedirs("./models", exist_ok=True)
    save_path = "./models/resnet18_cifar10_fp32.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, save_path, test_loader

def benchmark_baseline(model, test_loader, save_path):
    print("\n--- Running Baseline Benchmarks ---")
    
    # Force CPU for benchmarking to simulate edge hardware
    device = torch.device("cpu")
    model = model.to(device)
    model.eval() # Set to evaluation mode (disables dropout/batchnorm updates)
    
    # Metric 1: Accuracy
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient tracking for faster inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    
    # Metric 2: Model Size (MB)
    model_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    # Metric 3: CPU Latency (ms per image)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Warmup runs (to initialize CPU caches and avoid skewed timing)
    for _ in range(50):
        _ = model(dummy_input)
        
    # Actual timing
    start_time = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        _ = model(dummy_input)
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / iterations) * 1000
    
    print(f"FP32 Baseline Accuracy:   {accuracy:.2f}%")
    print(f"FP32 Baseline Model Size: {model_size_mb:.2f} MB")
    print(f"FP32 Baseline CPU Latency:{latency_ms:.2f} ms/image")

if __name__ == "__main__":
    # Run the full pipeline
    # Note: 5 epochs is enough to get a baseline for our project, 
    # but you can increase this to 15-20 later for better accuracy.
    trained_model, path, loader = train_baseline(epochs=5)
    benchmark_baseline(trained_model, loader, path)