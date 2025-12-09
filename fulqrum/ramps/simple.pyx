# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.subspace cimport Subspace
from fulqrum.core.bitset cimport Bitset


def ramps_simple_refinement(QubitOperator H, Subspace S,
                            Bitset start, unsigned int max_recursion=4):
    return 0
