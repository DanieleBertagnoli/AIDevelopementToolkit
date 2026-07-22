# AIDevelopementToolkit

AIDevelopementToolkit is a production-ready Python package that provides reusable utilities for AI development workflows.
The library is designed to help teams standardize training pipelines, logging, experiment tracking, preprocessing, and PyTorch model management without locking users into a single framework.

## Scope

This package is intended for machine learning practitioners and engineering teams who need:

- a lightweight foundation for structured logging and experiment tracking,
- reusable utilities for data preprocessing and metric computation,
- robust checkpointing, model export, and inference tooling for PyTorch,
- distributed training helpers to support multi-GPU workflows,
- automatic artifact logging through MLflow,
- and file I/O helpers for JSON/YAML/CSV data interchange.

## Main Features

- `logging_utils`: standard logger configuration, MLflow integration, and file-based artifact helpers.
- `data_utils`: data preprocessing and evaluation metrics with padded sequence support.
- `torch_utils`: model checkpointing, ONNX export, inference helpers, early stopping, and distributed training support.
- `general_utils`: deterministic seed setting for PyTorch, NumPy, and Python.

## Installation

```sh
pip install aidevelopementtoolkit
```

Or install from source:

```sh
git clone https://github.com/DanieleBertagnoli/AIDevelopementToolkit.git
cd AIDevelopementToolkit
pip install -e .
```

## What’s Included

- `aidevelopementtoolkit.logging_utils`: colorized console logger, file I/O wrappers, MLflow helpers, and Plotly visualization logging.
- `aidevelopementtoolkit.data_utils`: sequence-aware preprocessing and classification/clustering metrics.
- `aidevelopementtoolkit.torch_utils`: PyTorch checkpointing, ONNX export, runtime validation, model summaries, early stopping, and distributed process management.
- `aidevelopementtoolkit.general_utils`: seed initialization for reproducible experiments.

## Usage Guidance

This repository keeps usage examples in the `examples/` folder.
Review `examples/mnist_training.py` and `examples/run_distributed.sh` for real-world integration patterns and best practices.

The `examples/run_distributed.sh` script includes launcher commands for:
- single GPU, single node;
- multiple GPUs, single node;
- multiple GPUs across multiple nodes.

## Documentation

Complete package documentation is available at:

https://DanieleBertagnoli.github.io/AIDevelopementToolkit/

## Authors

Daniele Bertagnoli