"""
models/__init__.py — Model registry.

Supported model names: "cnn", "lstm", "resnet", "transformer"

Usage:
    from models import get_model_module
    module = get_model_module("cnn")
    model  = module.create_model(num_classes=35)
"""

import importlib
from types import ModuleType

_REGISTRY = {
    "cnn":         "models.cnn",
    "lstm":        "models.lstm",
    "resnet":      "models.resnet",
    "transformer": "models.transformer",
}


def get_model_module(name: str) -> ModuleType:
    """Return the module object for the requested model name."""
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return importlib.import_module(_REGISTRY[name])


def list_models():
    return list(_REGISTRY.keys())
