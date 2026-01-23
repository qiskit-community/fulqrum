# This code is a part of Fulqrum.
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
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cdef unordered_map[unsigned char, unsigned char] STR_TO_IND = {90: 0, 48: 1, 49: 2, 88: 3,
                                                               89: 4, 45: 5, 43: 6}

cdef unordered_map[unsigned char, string] IND_TO_STR = {0: 'Z', 1: '0', 2: '1', 3: 'X', 
                                                        4: 'Y', 5: '-', 6: '+'}
