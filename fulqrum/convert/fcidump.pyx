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

"""PySCF conversion utilities"""
from libcpp.vector cimport vector

from pathlib import Path
import time
import numpy as np
import logging
logger = logging.getLogger(__name__)


cdef class FCIDumpData():
    def __cinit__(self, string filename):
        self.data = parse_fcidump(filename)

    @property
    def NORB(self):
        return self.data.NORB

    @property
    def NELEC(self):
        return self.data.NELEC
    
    @property
    def ISYM(self):
        return self.data.ISYM
    
    @property
    def MS2(self):
        return self.data.MS2

    @property
    def UHF(self):
        return self.data.UHF

    @property
    def ECORE(self):
        return self.data.ECORE

    @property
    def H1(self):
        cdef double[::1] arr = <double [:self.data.H1.size()]>self.data.H1.data()
        return np.asarray(arr)

    @property
    def H2(self):
        cdef double[::1] arr = <double [:self.data.H2.size()]>self.data.H2.data()
        return np.asarray(arr)

    @property
    def ORBSYM(self):
        cdef int[::1] arr = <int [:self.data.H2.size()]>self.data.ORBSYM.data()
        return np.asarray(arr)


def read_fcidump(filename: str | Path):
    cdef string string_name = str(filename)
    cdef FCIDumpData out = FCIDumpData(filename)
    return out
