# cython: c_string_type=unicode, c_string_encoding=UTF-8
from .bitset cimport bitset_t, to_string
from libcpp cimport string

cdef class Bitset:
    
    cdef bitset_t bits

    def __cinit__(self, string bitstring):
        self.bits = bitset_t(bitstring, 0, bitstring.size())

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

    def as_string(self):
        cdef string s
        to_string(self.bits, s)
        return s
