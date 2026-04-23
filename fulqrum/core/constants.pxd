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


cdef extern from "src/constants.hpp":
    cdef const size_t MAX_SIZE_T
    cdef const double ATOL
    cdef const double RTOL
    ctypedef unsigned int width_t # The actual type here does not matter as the compiler will just figure it out

