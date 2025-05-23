# cython: c_string_type=unicode, c_string_encoding=UTF-8
from cython.operator cimport dereference as deref
from .bitset cimport bitset_t, to_string
from .bitset_view cimport BitsetView
from libcpp.string cimport string

from fulqrum.exceptions import FulqrumError

include "includes/bitset_utils_header.pxi"


cdef class BitsetView:
        
    cdef void bit_ptr(self, bitset_t * ptr):
        self.bits = ptr

    def __len__(self):
        return self.bits.size()

    def __repr__(self):
        cdef string s
        to_string(deref(self.bits),  s)
        return f"<BitsetView: {s}>"

    def size(self):
        """Number of bits in Bitset

        Returns:
            int: Number of bits
        """
        return self.bits.size()

    def __getitem__(self, size_t idx):
        return self.bits.at(idx)

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
        to_string(deref(self.bits), s)
        return s

    def to_int(self):
        """Convert Bitset to Python integer

        Returns:
            int: Integer value for Bitset
        """
        cdef string s
        to_string(deref(self.bits), s)
        return int(s, 2)

    def bin_width_int(self, unsigned int bin_width):
        """Compute the integer for a given bin-width

        Parameters:
            bin_width (int): Bin-width to compute

        Returns:
            int: Computed bin-width
        """
        if bin_width > self.bits.size():
            raise FulqrumError("bin_width is larger than number of bits")
        cdef size_t out
        bin_int(deref(self.bits), bin_width, out)
        return out
