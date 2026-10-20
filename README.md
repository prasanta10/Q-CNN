# Q-CNN: Mixed-Precision Quantization Engine

Q-CNN is a model-quantization project built around **ResNet-18**. It explores low-bit quantization, layer-wise sensitivity, greedy bit-width allocation, and ONNX-based inference benchmarking to study trade-offs among model footprint, prediction quality, and inference latency.

> **Project focus:** quantization and deployment-oriented evaluation of a CNN. Benchmark results depend on the model checkpoint, dataset, hardware, runtime, and measurement procedure.

## Highlights

- **ResNet-18 quantization:** experiments with 4-bit asymmetric *fake quantization* to study the effect of reduced numerical precision.
- **Layer-wise sensitivity analysis:** identifies layers more affected by low-precision quantization.
- **Greedy bit-width allocation:** explores mixed-precision configurations using a sensitivity-oriented search.
- **ONNX benchmarking:** exports models for runtime evaluation and comparison.
- **Latency investigation:** examines CPU dynamic-dequantization overhead, showing that a smaller model does not necessarily run faster on every backend.

## Reported result

The project reports a **4.0× reduction in model footprint**, from **42.6 MB to 10.7 MB**, for the evaluated model artifact.

This is a reported model-footprint comparison, not a guarantee of 4× lower inference latency. Fake quantization simulates low-precision arithmetic during the forward pass and does not, by itself, guarantee weights are stored as packed 4-bit values. Actual storage and latency depend on the export format and runtime implementation.

## Method overview

1. **Load model and evaluation data.** Use the ResNet-18 model and the dataset/configuration used by the experiment.
2. **Establish a baseline.** Record baseline model footprint and evaluation results before quantization.
3. **Apply quantization.** Configure asymmetric fake quantization at the selected precision.
4. **Analyze sensitivity.** Compare layer-wise behavior to identify layers sensitive to reduced precision.
5. **Allocate precision.** Use the greedy bit-width search to evaluate a mixed-precision configuration.
6. **Export and benchmark.** Export the selected model to ONNX and measure quality, model size, and runtime.
7. **Inspect trade-offs.** Compare results using consistent settings and investigate cases where quantization overhead affects latency.

## Repository layout

```text
Q-CNN/
├── data/               # Local datasets or evaluation data
├── exported_models/    # Exported model artifacts
├── models/             # Model code and/or checkpoints
├── quant_env/          # Local Python environment (do not commit)
└── optimal_config.json # Selected quantization configuration
```

The precise contents may vary. Avoid committing datasets, checkpoints, exported binaries, or virtual environments unless needed for collaboration. If `data/` and `quant_env/` should remain local, add them to `.gitignore`.

## Setup

The exact install and run commands depend on the scripts and dependency files in the repository. A typical Python environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

Install dependencies using the dependency file included in the repository, if present:

```bash
pip install -r requirements.txt
```

If the repository does not yet have a `requirements.txt`, create one with the package versions from the working environment. Keep environment-specific files and downloaded datasets out of source control.

### Running an experiment

Use the project's existing evaluation, quantization, and export scripts. Confirm the actual entry-point filenames and arguments in the source code before documenting specific commands. For reproducibility, record:

- model/checkpoint and dataset
- quantization method and bit-width configuration
- evaluation metric and result
- exact artifact measured and its size
- runtime, device, batch size, and input shape
- whether the result uses fake quantization or genuinely packed low-bit weights

## Evaluation and benchmarking

Keep baseline and quantized runs consistent when comparing results.

| Dimension | What to record |
|---|---|
| Model quality | Accuracy or task-appropriate metric |
| Footprint | Exact file/artifact measured and its size in MB |
| Latency | Measurement method, warm-up, repetitions, and summary statistic |
| Throughput | Inputs per second, batch size, and input shape |
| Runtime | ONNX Runtime version, execution provider, and settings |
| Hardware | CPU/GPU model and available memory |
| Quantization | Weight/activation precision, granularity, scale/zero-point scheme, and fake vs. packed quantization |

Do not compare latency across different devices or runtimes without noting those differences. CPU dequantization overhead, kernel support, and memory movement can offset expected gains from low-bit representations.

## Scope and limitations

- The implementation is focused on ResNet-18 and the quantization configurations supported by this repository.
- The reported footprint reduction is not a universal guarantee for other models or deployment backends.
- Fake quantization is useful for studying quantization effects, but hardware acceleration requires a compatible low-bit representation and supported kernels.
- Reproduction depends on preserving the configuration, checkpoint, dataset, software versions, and benchmarking procedure.

## Potential extensions

- Document exact calibration and evaluation datasets.
- Add reproducible baseline-versus-quantized result tables.
- Compare fake quantization with packed INT4 inference where supported.
- Measure quality/latency trade-offs across multiple bit-width configurations.
- Record per-layer precision choices and sensitivity scores.
- Benchmark additional execution providers or hardware backends.

---

This README reflects the project's stated scope and reported results. Update setup commands, run examples, dataset details, and benchmark tables once the actual scripts and experiment outputs are finalized.
