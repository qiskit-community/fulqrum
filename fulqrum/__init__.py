# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum"""

__version__ = "0.0.5"


from .core import (
    QubitOperator,
    Subspace,
    SubspaceHamiltonian,
    FermionicOperator,
    Bitset,
)


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
