# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cdef unordered_map[char, char] STR_TO_IND = {90: 0, 48: 1, 49: 2, 88: 3,
                                             89: 4, 45: 5, 43: 6}

cdef unordered_map[char, string] IND_TO_STR = {0: 'Z', 1: '0', 2: '1', 3: 'X', 
                                               4: 'Y', 5: '-', 6: '+'}
