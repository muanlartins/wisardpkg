"""Python-side ports of recent weightless neural network architectures.

These models are convenience wrappers / pure-Python implementations
that complement the C++ core in ``wisardpkg``. Each module documents
the reference paper, repo, and reproduction caveats.

Quick reference:

| Class | Paper | Backend | Optional dependency |
|-------|-------|---------|---------------------|
| ``BTHOWeN`` | Susskind et al., PACT 2022 (arXiv:2203.01479) | ``wisardpkg`` C++ (BloomWisard + H3 + Gaussian thermometer) | none |
| ``DWN``     | Bacellar et al., ICML 2024 (arXiv:2410.11112) | PyTorch (CPU port of paper's CUDA kernels) | ``torch`` |
| ``ULEEN``   | Susskind et al., ACM TACO 2023 (doi:10.1145/3629522) | PyTorch (CPU port of paper's libtorch H3 extension) | ``torch`` |

BTHOWeN is always available since it only uses the C++ core. DWN and
ULEEN require the optional ``torch`` extra (``pip install wisardpkg[torch]``);
attempting to import them without torch raises ``ImportError`` at access time.
"""
# BTHOWeN is always importable — no torch dependency
from .bthowen import BTHOWeN  # noqa: F401

# Lazy torch-dependent imports
import importlib as _importlib


def __getattr__(name):
    if name == "DWN":
        return _importlib.import_module("wisardpkg.models.dwn").DWNClassifier
    if name == "DWNClassifier":
        return _importlib.import_module("wisardpkg.models.dwn").DWNClassifier
    if name == "ULEEN":
        return _importlib.import_module("wisardpkg.models.uleen").ULEENClassifier
    if name == "ULEENClassifier":
        return _importlib.import_module("wisardpkg.models.uleen").ULEENClassifier
    raise AttributeError(f"module 'wisardpkg.models' has no attribute {name!r}")


__all__ = ["BTHOWeN", "DWN", "DWNClassifier", "ULEEN", "ULEENClassifier"]
