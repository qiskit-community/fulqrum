# cython: c_string_type=unicode, c_string_encoding=UTF-8
from .bitset cimport bitset_t, to_string
from libcpp cimport string

import numpy as np
import numbers
from collections.abc import Iterable
from fulqrum.exceptions import FulqrumError

from fulqrum.core.qubit_operator cimport QubitOperator

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
        """Number of bits in Bitset

        Returns:
            int: Number of bits
        """
        return self.bits.size()

    def __getitem__(self, size_t idx):
        return self.bits[idx]

    def __eq__(self, Bitset other):
        return self.bits == other.bits

    def __neq__(self, Bitset other):
        return self.bits != other.bits

    def num_blocks(self):
        """Number of blocks (int64) used to store Bitset

        Returns:
            int: Number of blocks
        """
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

    def bin_width_int(self, unsigned int bin_width):
        """Compute the integer for a given bin-width

        Parameters:
            bin_width (int): Bin-width to compute

        Returns:
            int: Computed bin-width
        """
        if bin_width > self.bits.size():
            raise FulqrumError("bin_width is larger than number of bits")
        cdef size_t out = 0
        bin_int(self.bits, bin_width, out)
        return out

    def flip(self, object bits):
        """Flip one or more bits inplace

        Parameters:
            bits (int or array_like): Indices to flip
        """
        cdef unsigned int[::1] int_array
        if isinstance(bits, numbers.Integral):
            int_array = np.asarray([bits], dtype=np.uint32)
        elif isinstance(bits, Iterable):
            int_array = np.asarray(bits, dtype=np.uint32)
        else:
            raise FulqrumError('bits arg is not a valid type')

        flip_bits(self.bits, &int_array[0], int_array.shape[0])

    def offdiag_flip(self, QubitOperator op):
        """Flip bits corresponding to off-diagonal operators in a single Hamiltonian term

        Parameters:
            op (QubitOperator): QubitOperator with a single-term

        Returns:
            Bitset: Bitset with off-diagonal bits flipped

        Raises:
            FulqrumError: Operator must have a single-term

            FulqrumError: Size of Bitset and QubitOperator do not match
        """
        if op.num_terms > 1:
            raise FulqrumError("Operator must contain a single-term only")
        if self.size() != op.width:
            raise FulqrumError('Bitset and Operator must have same size')
        cdef Bitset out = Bitset()
        out.bits = self.bits
        get_column_bitset(out.bits,
                          &op.oper.terms[0].indices[0],
                          &op.oper.terms[0].values[0],
                          op.oper.terms[0].indices.size())
        return out


    def ladder_int(self, unsigned int[::1] inds, unsigned int ladder_width=3):
        """Compute the ladder integer of a bitset for the given indices

        Parameters:
            inds (ndarray): Unsigned int indices to use
            ladder_width (int): Number of bits to consider, default = 3

        Notes:
            If the number of indices is less than the ladder_width then
            that is the new ladder_width
        """
        ladder_width = min(inds.shape[0], ladder_width)
        return bitset_ladder_int(self.bits, &inds[0], ladder_width)

