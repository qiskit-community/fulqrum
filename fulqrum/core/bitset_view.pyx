# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from cython.operator cimport dereference as deref
from .bitset cimport bitset_t, to_string
from .bitset_view cimport BitsetView
from libcpp.string cimport string

from fulqrum.exceptions import FulqrumError

include "includes/bitset_utils_header.pxi"


cdef class BitsetView:
        
    cdef void assign_bits(self, bitset_t bitset):
        self.bits = bitset

    def __len__(self):
        return self.bits.size()

    def __repr__(self):
        cdef string s
        to_string(self.bits,  s)
        return f"<BitsetView: {s}>"

    def size(self):
        """Number of bits in Bitset

        Returns:
            int: Number of bits
        """
        return self.bits.size()

    def __getitem__(self, size_t idx):
        return (self.bits)[idx]

    def __eq__(self, BitsetView other):
        return self.bits == other.bits

    def __neq__(self, BitsetView other):
        return self.bits != other.bits

    def num_blocks(self):
        return self.bits.num_blocks()

    def to_string(self):
        """Convert Bitset to string

        Returns:
            str: String representation of Bitset
        """
        cdef string s
        to_string(self.bits, s)
        return s

    def to_int(self):
        """Convert Bitset to Python integer

        Returns:
            int: Integer value for Bitset
        """
        cdef string s
        to_string(self.bits, s)
        return int(s, 2)
