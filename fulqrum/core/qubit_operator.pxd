# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

include "includes/base_header.pxi"

cdef class QubitOperator:
    cdef QubitOperator_t oper
    cdef unsigned int _iter_index

    cpdef void append(self, QubitOperator other)
    cpdef int is_diagonal(self)