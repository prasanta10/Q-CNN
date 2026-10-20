import json
import torch
import torch.nn as nn
from dataset import get_cifar10_loaders
from model import get_cifar_resnet

def evaluate_model(model, test_loader, device):
    """Fast subset evaluation for quick search iterations."""
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

def fake_quantize_tensor(tensor, num_bits=4):
    """Applies simulated INT4 quantization to a target weight tensor."""
    t_min, t_max = tensor.min(), tensor.max()
    if t_max == t_min:
        return tensor
    
    levels = (2 ** num_bits) - 1  # 255 for 8-bit
    scale = (t_max - t_min) / float(levels)
    
    quantized = torch.round((tensor - t_min) / scale)
    dequantized = (quantized * scale) + t_min
    return dequantized

def apply_precision_config(model, clean_weights, config):
    """
    Applies the target precision configuration to the model.
    config: dict mapping layer_name -> "INT4" or "FP32"
    """
    for name, module in model.named_modules():
        if name in config:
            # Always start from pristine FP32 copy
            orig_w = clean_weights[name].clone()
            
            if config[name] == "INT4":
                module.weight.data = fake_quantize_tensor(orig_w, num_bits=4)
            else: # FP32
                module.weight.data = orig_w

def run_mixed_precision_allocator(max_acc_drop=1.0):
    """
    Automated Greedy Search Algorithm for Mixed-Precision Allocation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Precision Allocator on: {device}")
    
    # 1. Load Data, Model, and Sensitivity Scores
    _, test_loader = get_cifar10_loaders(data_dir="../data")
    
    model = get_cifar_resnet(num_classes=10)
    model.load_state_dict(torch.load("./models/resnet18_cifar10_fp32.pth", map_location=device, weights_only=True))
    model.to(device)
    
    with open("sensitivity_profile.json", "r") as f:
        sensitivity_profile = json.load(f)
        
    # Store clean original weights in memory to reset during search
    clean_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            clean_weights[name] = module.weight.data.clone()
            
    # 2. Get Baseline FP32 Accuracy
    baseline_acc = evaluate_model(model, test_loader, device)
    target_acc = baseline_acc - max_acc_drop
    print(f"FP32 Baseline Accuracy:    {baseline_acc:.2f}%")
    print(f"Target Accuracy (Drop <= {max_acc_drop}%): {target_acc:.2f}%\n")
    
    # 3. Initialize ALL layers to INT4
    config = {layer: "INT4" for layer in sensitivity_profile.keys()}
    
    # Rank layers by sensitivity (highest drop first)
    sorted_layers_by_sensitivity = list(sensitivity_profile.keys())
    
    iteration = 0
    print("--- Starting Automated Mixed-Precision Search ---")
    
    while True:
        iteration += 1
        # Apply current precision mapping
        apply_precision_config(model, clean_weights, config)
        
        # Evaluate current config
        current_acc = evaluate_model(model, test_loader, device)
        INT4_count = sum(1 for p in config.values() if p == "INT4")
        fp32_count = sum(1 for p in config.values() if p == "FP32")
        
        print(f"Iter {iteration:2d} | Accuracy: {current_acc:.2f}% | INT4: {INT4_count:2d} layers, FP32: {fp32_count:2d} layers")
        
        # Check if accuracy criteria met
        if current_acc >= target_acc:
            print(f"\n[SUCCESS] Met target accuracy ({current_acc:.2f}% >= {target_acc:.2f}%)!")
            break
            
        # If accuracy is below target, upgrade the most sensitive remaining INT4 layer to FP32
        layer_to_upgrade = None
        for layer in sorted_layers_by_sensitivity:
            if config[layer] == "INT4":
                layer_to_upgrade = layer
                break
                
        if layer_to_upgrade is None:
            print("\n[WARNING] All layers restored to FP32, target accuracy still not reached.")
            break
            
        print(f"   --> Restoring sensitive layer '{layer_to_upgrade}' to FP32...")
        config[layer_to_upgrade] = "FP32"

    # 4. Save Final Optimal Configuration
    with open("optimal_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("\nSaved optimal configuration to optimal_config.json")
    
    # Print summary of decisions
    print("\n--- Final Layer Precision Assignment ---")
    for layer, prec in config.items():
        print(f"  {layer:<25}: {prec}")

if __name__ == "__main__":
    # Target maximum accuracy drop: 1.0%
    run_mixed_precision_allocator(max_acc_drop=1.0)