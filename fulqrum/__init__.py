# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum"""

# Grab version from version.py
try:
    from .version import version as __version__
    from .version import openmp
except ImportError:
    __version__ = "0.0.0"
    openmp = False

from .core import QubitOperator, Subspace, SubspaceHamiltonian, FermionicOperator, Bitset


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
    print("Paul D. Nation, Abdullah Saki, and Hwajung Kang")
    print(f"Fulqrum version: {__version__}")
    print(f"OpenMP enabled: {openmp}")
