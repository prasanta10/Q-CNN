import torch
import torch.nn as nn
import json
from dataset import get_cifar10_loaders
from model import get_cifar_resnet

def evaluate_model(model, test_loader, device):
    """Evaluates accuracy on a representative subset (20 batches = 2560 images)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100.0 * correct / total

def fake_quantize_weights(weights, num_bits=4):
    """
    Simulates low-bit quantization on FP32 weights to stress-test layer sensitivity.
    For num_bits=4, levels = 2^4 - 1 = 15 steps.
    """
    w_min, w_max = weights.min(), weights.max()
    
    if w_max == w_min:
        return weights
        
    levels = (2 ** num_bits) - 1  # 15 for 4-bit, 255 for 8-bit
    scale = (w_max - w_min) / float(levels)
    
    # Quantize and Dequantize
    quantized = torch.round((weights - w_min) / scale)
    dequantized = (quantized * scale) + w_min
    
    return dequantized

def profile_sensitivity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sensitivity analysis on: {device}")
    
    _, test_loader = get_cifar10_loaders(data_dir="../data")
    
    model = get_cifar_resnet(num_classes=10)
    model.load_state_dict(torch.load("./models/resnet18_cifar10_fp32.pth", map_location=device, weights_only=True))
    model.to(device)
    
    # 1. Get Baseline Accuracy on the subset
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"Baseline FP32 Accuracy (Subset): {baseline_acc:.2f}%\n")
    
    sensitivity_scores = {}
    
    # 2. Iterate through layers using 4-bit stress-testing
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            original_weights = module.weight.data.clone()
            
            # Apply 4-bit Fake Quantization to reveal sensitivity gaps
            module.weight.data = fake_quantize_weights(original_weights, num_bits=4)
            
            quantized_acc = evaluate_model(model, test_loader, device)
            accuracy_drop = baseline_acc - quantized_acc
            
            sensitivity_scores[name] = max(0.0, accuracy_drop) # Keep drop positive
            module.weight.data = original_weights
            
            print(f"Profiling layer: {name:<25} 4-bit Drop: {accuracy_drop:+.2f}%")
            
    # 3. Sort and Save
    sorted_scores = dict(sorted(sensitivity_scores.items(), key=lambda item: item[1], reverse=True))
    
    print("\n--- Top 5 Most Sensitive Layers (4-Bit Stress Test) ---")
    for i, (layer, drop) in enumerate(list(sorted_scores.items())[:5]):
        print(f"{i+1}. {layer:<25}: {drop:.2f}% drop")
        
    with open("sensitivity_profile.json", "w") as f:
        json.dump(sorted_scores, f, indent=4)
    print("\nSaved sensitivity profile to sensitivity_profile.json")

if __name__ == "__main__":
    profile_sensitivity()