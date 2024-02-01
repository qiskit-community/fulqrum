# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef pair[size_t, unsigned char] size_uchar_pair

cdef struct s_OperatorTerm:
    double complex coeff
    vector[size_uchar_pair] operators

ctypedef s_OperatorTerm OperatorTerm


cdef bool diagonal_term(OperatorTerm * term)

