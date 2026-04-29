"""wisardpkg — A library of WiSARD models in C++ with Python bindings,
plus pure-Python ports of recent weightless architectures.

The C++ core (BloomWisard, Wisard, ClusWisard, RegressionWisard, all the
binarization techniques, mappings, etc.) lives in the ``_native`` C++
extension. We re-export everything at package level so existing user code
works unchanged:

    >>> import wisardpkg as wp
    >>> clf = wp.Wisard(addressSize=4)            # C++ class, unchanged
    >>> th = wp.GaussianThermometer(thermoSize=8) # C++ class, unchanged

The ``wisardpkg.models`` subpackage adds Python-side ports of three recent
weightless architectures that build on top of the C++ core (or are pure
PyTorch implementations of architectures the core does not expose
directly):

    >>> from wisardpkg.models import BTHOWeN              # always available
    >>> from wisardpkg.models import DWN, ULEEN           # requires `pip install wisardpkg[torch]`

See ``wisardpkg/models/__init__.py`` and the individual module docstrings
for the architectural details and references.
"""
from ._native import *  # noqa: F401, F403  — re-export the C++ core
from ._native import __version__  # noqa: F401

# Convenience: lazy access to wisardpkg.models without forcing torch import
import importlib as _importlib


def __getattr__(name):
    # PEP 562 — attribute lookup on the module
    if name == "models":
        return _importlib.import_module("wisardpkg.models")
    raise AttributeError(f"module 'wisardpkg' has no attribute {name!r}")
