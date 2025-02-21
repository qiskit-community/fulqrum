# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector

from fulqrum.core.qubit_operator cimport QubitOperator

import numpy as np
cimport numpy as np

include "../core/includes/base_header.pxi"
include "../core/includes/strings_header.pxi"
include "../core/includes/bitstrings_header.pxi"



def str_to_vec_test(string input):
    """Test the conversion of a string to a bit vector
    
    Parameters:
        input (str): Input string
    
    Returns:
        ndarray: Array of type uintp
    """
    cdef vector[unsigned char] vec
    cdef size_t bit_len = input.size()
    cdef size_t kk
    cdef size_t[::1] out = np.empty(bit_len, dtype=np.uintp)
    vec.reserve(bit_len)
    string_to_vec(input.c_str(), &vec[0], bit_len)
    for kk in range(bit_len):
        out[kk] = vec[kk]
    return np.asarray(out)


def find_col_vec_test(QubitOperator op, string input):

    cdef vector[unsigned char] vec
    cdef vector[unsigned char] temp
    cdef size_t bit_len = input.size()
    cdef size_t kk
    cdef size_t[::1] out = np.empty(bit_len, dtype=np.uintp)
    cdef const unsigned char * row_data = <unsigned char *>input.c_str()
    vec.resize(bit_len)
    temp.resize(bit_len)
    string_to_vec(input.c_str(), &vec[0], bit_len)
    temp = vec
    get_column_vec(&vec[0], &temp[0], bit_len,
                   &op.oper.terms[0].indices[0],
                   &op.oper.terms[0].values[0],
                   op.oper.terms[0].indices.size())
    for kk in range(bit_len):
        out[kk] = temp[kk]
    return np.asarray(out)
