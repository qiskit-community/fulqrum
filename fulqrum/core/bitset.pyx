# cython: c_string_type=unicode, c_string_encoding=UTF-8
from .bitset cimport bitset_t, to_string
from libcpp cimport string

import numpy as np
import numbers
from collections.abc import Iterable
from fulqrum.exceptions import FulqrumError

include "includes/bitset_utils_header.pxi"



cdef class Bitset:

    def __cinit__(self, str bitstring = ''):
        cdef string temp = bitstring
        self.bits = bitset_t(temp, 0, temp.size())

    def __dealloc__(self):
        self.bits = bitset_t()

    def __len__(self):
        return self.bits.size()

    def __repr__(self):
        cdef string s
        to_string(self.bits, s)
        return f"<Bitset: {s}>"

    def size(self):
        return self.bits.size()

    def __getitem__(self, size_t idx):
        return self.bits[idx]

    def __eq__(self, Bitset other):
        return self.bits == other.bits

    def __neq__(self, Bitset other):
        return self.bits != other.bits

    def num_blocks(self):
        return self.bits.num_blocks()

    def to_string(self):
        cdef string s
        to_string(self.bits, s)
        return s

    def to_int(self):
        cdef string s
        to_string(self.bits, s)
        return int(s, 2)

    def bin_width_int(self, unsigned int bin_width):
        if bin_width > self.bits.size():
            raise FulqrumError("bin_width is larger than number of bits")
        cdef size_t out
        bin_int(self.bits, bin_width, out)
        return out

    def flip(self, object bits):
        cdef size_t[::1] int_array
        if isinstance(bits, numbers.Integral):
            int_array = np.asarray([bits], dtype=np.uintp)
        elif isinstance(bits, Iterable):
            int_array = np.asarray(bits, dtype=np.uintp)
        else:
            raise Exception("What the hell")

        flip_bits(self.bits, &int_array[0], int_array.shape[0])

