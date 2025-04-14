# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

include "includes/base_header.pxi"

cdef class FermionicOperator:
    cdef FermionicOperator_t oper
    cpdef void append(self, FermionicOperator other)