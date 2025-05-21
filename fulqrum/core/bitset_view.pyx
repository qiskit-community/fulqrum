# cython: c_string_type=unicode, c_string_encoding=UTF-8
from cython.operator cimport dereference as deref
from .bitset cimport bitset_t, to_string
from .bitset_view cimport BitsetView
from libcpp.string cimport string


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
        return self.bits.size()

    def __getitem__(self, size_t idx):
        return self.bits.at(idx)

    def __eq__(self, BitsetView other):
        return self.bits == other.bits

    def __neq__(self, BitsetView other):
        return self.bits != other.bits

    def num_blocks(self):
        return self.bits.num_blocks()
