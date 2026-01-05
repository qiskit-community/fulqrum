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
from fulqrum.core.bitset_hashmap cimport BitsetHashMapWrapper
from fulqrum.core.bitset cimport bitset_t

cdef class PyBitsetHashMap:
    cdef BitsetHashMapWrapper* c_map

    def __cinit__(self):
        self.c_map = new BitsetHashMapWrapper()

    def __dealloc__(self):
        del self.c_map

    def insert_unique(self, bitstring: str, value: int):
        cdef bitset_t bs = bitset_t(bitstring, 0, len(bitstring))
        self.c_map.insert(bs, <size_t>value)

    def get(self, bitstring: str) -> int:
        cdef bitset_t bs = bitset_t(bitstring, 0, len(bitstring))
        return self.c_map.get(bs)
    
    def size(self) -> int:
        return self.c_map.size()
