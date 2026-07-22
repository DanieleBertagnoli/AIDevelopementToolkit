"""Torch utilities for model management, distributed training, and early stopping.

The `aidevelopementtoolkit.torch_utils` package provides reusable PyTorch
utilities for model creation, checkpoint management, ONNX support, early stopping,
and distributed training.

This package is intended for use in training workflows that require:
- reliable model save/load semantics,
- export and validation of PyTorch models to ONNX,
- distributed training orchestration via `torch.distributed`,
- early stopping based on validation metrics.

Modules
-------
distributed_torch_utils
    Helpers for distributed PyTorch training and process group management.

EarlyStopper
    Early stopping logic to monitor validation metrics and stop training.

model_utils
    Model saving, loading, and architecture utility functions.
"""
