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
import numpy as np

if sizeof(width_t) == 4:
    np_width_t = np.uint32
elif sizeof(width_t) == 2:
    np_width_t = np.uint16
elif sizeof(width_t) == 8:
    np_width_t = np.uint64
else:
    raise Exception("Unknown width_t type")
