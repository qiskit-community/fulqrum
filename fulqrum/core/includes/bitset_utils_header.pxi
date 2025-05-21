# Fulqrum
# Copyright (C) 2024, IBM
from fulqrum.core.bitset cimport bitset_t


cdef extern from "../src/bitset_utils.hpp":
    
    void bin_int(bitset_t& b, unsigned int len, size_t& res)