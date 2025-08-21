# cython: c_string_type=unicode, c_string_encoding=UTF-8
from .bitset cimport bitset_t


cdef class BitsetView:
    cdef bitset_t bits

    cdef void assign_bits(self, bitset_t)
