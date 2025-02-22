# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

from libcpp.vector cimport vector


cdef class Subspace():
    cdef vector[unsigned char] subspace
    cdef vector[size_t] bin_counts
    cdef vector[size_t] bin_ranges
    cdef size_t num_qubits
    cdef public size_t num_bins
    cdef public bin_width
    cdef public size_t size
