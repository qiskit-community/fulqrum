# Fulqrum
# Copyright (C) 2024, IBM
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)

Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

I = np.array([[1, 0], [0, 1]], dtype=complex)

Z = np.array([[1, 0], [0, -1]], dtype=complex)

G = np.array([[1, 0], [0, 0]], dtype=complex)

O = np.array([[0, 0], [0, 1]], dtype=complex)

P = np.array([[0, 0], [1, 0]], dtype=complex)

M = np.array([[0, 1], [0, 0]], dtype=complex)

str_matrix_convert = {"I": I, "Z": Z, "0": G, "1": O, "X": X, "Y": Y, "+": P, "-": M}


def kron_str(operator_string):
    """Performs the kron over a string consisting of qubit operators"""
    mat = str_matrix_convert[operator_string[0].upper()]
    for kk in range(1, len(operator_string)):
        mat = np.kron(mat, str_matrix_convert[operator_string[kk].upper()])
    return mat
