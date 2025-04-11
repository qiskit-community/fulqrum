# Fulqrum
# Copyright (C) 2024, IBM
from libcpp.vector cimport vector

include "base_header.pxi"

cdef extern from "../src/jordanwigner.hpp":
    void jw_term(FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term) nogil
