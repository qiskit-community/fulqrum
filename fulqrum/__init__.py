# Fulqrum - Top Hat
# Copyright (C) 2024, IBM

"""Fulqrum Top Hat core"""

# Grab version from version.py
try:
    from .version import version as __version__
    from .version import openmp
except ImportError:
    __version__ = "0.0.0"
    openmp = False

from .core.qubit_operator import QubitOperator
