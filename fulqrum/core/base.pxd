# Fulqrum 
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp cimport bool
from libcpp.vector cimport vector

cdef struct s_OperatorTerm:
    double complex coeff
    vector[size_t] indices
    vector[char] values

ctypedef s_OperatorTerm OperatorTerm


cdef bool diagonal_term(OperatorTerm * term)

