# AIDevelopementToolkit

AIDevelopementToolkit provides a production-ready utility suite for AI development workflows.
The package is designed to help teams standardize logging, experiment tracking, preprocessing, PyTorch model management, ONNX export, and distributed training.

## Why AIDevelopementToolkit?

- Purpose-built for PyTorch-based research and engineering workflows.
- Shared utilities for logging, MLflow integration, data preparation, and model lifecycle management.
- Designed to keep examples in `examples/` while documentation stays concise and easy to navigate.
- Supports single-GPU, multi-GPU single-node, and multi-node distributed training patterns.

## Core Package Areas

- `logging_utils`: formatted logger configuration, MLflow experiment helpers and file I/O for JSON/YAML/CSV (also supporting S3 buckets!)
- `data_utils`: preprocessing functions, classification and clustering metrics.
- `torch_utils`: checkpoint saving/loading, ONNX export and runtime validation, early stopping class and distributed PyTorch DataParallel utilities.
- `general_utils`: deterministic seed setting.

## Installation

```bash
pip install aidevelopementtoolkit
```

Or install from source:

```bash
git clone https://github.com/DanieleBertagnoli/AIDevelopementToolkit.git
cd AIDevelopementToolkit
pip install -e .
```

## Examples and Documentation

Real usage examples are kept in the `examples/` folder. The script `examples/mnist_training.py` contains a full example on how the package shall be used.

This includes `examples/run_distributed.sh`, which shows commands for:

- single GPU on a single node,
- multiple GPUs on a single node,
- multiple GPUs across multiple nodes.

For full API documentation, visit the generated site:

https://DanieleBertagnoli.github.io/AIDevelopementToolkit/

## Getting Started

Import the package in your Python code and use the example scripts for integration patterns. The docs focus on package scope and purpose, while `examples/` provides runnable workflows.