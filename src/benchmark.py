import os
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

def benchmark_model(model_path, num_runs=100):
    """Measures average inference time (latency) in milliseconds."""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    # Warmup runs
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_time_ms

def run_benchmarks():
    fp32_path = "./exported_models/resnet18_fp32.onnx"
    prep_path = "./exported_models/resnet18_fp32_prep.onnx"
    int8_path = "./exported_models/resnet18_true_int8.onnx"
    
    print("1. Pre-processing ONNX graph to resolve shape annotations...")
    quant_pre_process(
        input_model_path=fp32_path, 
        output_model_path=prep_path,
        save_as_external_data=False  # Forces ONNX to keep everything in one file
)

    print("2. Compiling ONNX model to True Physical INT8...")
    quantize_dynamic(
        model_input=prep_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8
    )
    
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    
    print(f"\n--- Storage Footprint ---")
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB (Reduction: {fp32_size/int8_size:.1f}x)")
    
    print("\n3. Running CPU Latency Benchmarks (Averaged over 100 runs)...")
    fp32_latency = benchmark_model(fp32_path)
    int8_latency = benchmark_model(int8_path)
    
    print(f"\n--- CPU Latency (ms per image) ---")
    print(f"FP32 Latency: {fp32_latency:.2f} ms")
    print(f"INT8 Latency: {int8_latency:.2f} ms (Speedup: {fp32_latency/int8_latency:.2f}x)")

if __name__ == "__main__":
    run_benchmarks()