import json
import torch
import torch.nn as nn
import os
from model import get_cifar_resnet

def fake_quantize_tensor(tensor, num_bits=4):
    """Applies low-bit quantization mapping to weight tensors."""
    t_min, t_max = tensor.min(), tensor.max()
    if t_max == t_min:
        return tensor
    levels = (2 ** num_bits) - 1
    scale = (t_max - t_min) / float(levels)
    quantized = torch.round((tensor - t_min) / scale)
    return (quantized * scale) + t_min

def export_models_to_onnx():
    device = torch.device("cpu")
    os.makedirs("./exported_models", exist_ok=True)
    
    # 1. Load FP32 Checkpoint
    model = get_cifar_resnet(num_classes=10)
    model.load_state_dict(torch.load("./models/resnet18_cifar10_fp32.pth", map_location=device, weights_only=True))
    model.eval()

    # Create dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # 2. Export FP32 Baseline Model to ONNX
    fp32_path = "./exported_models/resnet18_fp32.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo = False
    )
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"FP32 ONNX Model exported: {fp32_path} ({fp32_size:.2f} MB)")

    # 3. Apply Optimal Mixed-Precision Configuration
    with open("optimal_config.json", "r") as f:
        config = json.load(f)

    for name, module in model.named_modules():
        if name in config:
            if config[name] in ["INT8", "INT4"]:
                module.weight.data = fake_quantize_tensor(module.weight.data, num_bits=4)

    # 4. Export Mixed-Precision Model to ONNX
    mixed_path = "./exported_models/resnet18_mixed.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        mixed_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo =False
    )
    mixed_size = os.path.getsize(mixed_path) / (1024 * 1024)
    print(f"Mixed-Precision ONNX Model exported: {mixed_path} ({mixed_size:.2f} MB)")

if __name__ == "__main__":
    export_models_to_onnx()