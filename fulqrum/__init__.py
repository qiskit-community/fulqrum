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

from .core import QubitOperator, Subspace, SubspaceHamiltonian


def about():
    """About Fulqrum"""
    print("")
    print("███████ ██    ██ ██       ██████  ██████  ██    ██   ████    ████")
    print("██      ██    ██ ██      ██    ██ ██   ██ ██    ██  ██  ██  ██  ██")
    print("█████   ██    ██ ██      ██    ██ ██████  ██    ██ ██    ████    ██")
    print("██      ██    ██ ██      ██ ▄▄ ██ ██   ██ ██    ██ ██    ████    ██")
    print("██       ██████  ███████  ██████  ██   ██  ██████  ██    ████    ██")
    print("")
    print("Copyright (C) 2024, IBM Quantum")
    print("Paul D. Nation and Hwajung Kang")
    print(f"Fulqrum version: {__version__}")
    print(f"OpenMP enabled: {openmp}")
