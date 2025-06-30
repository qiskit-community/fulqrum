# Fulqrum
# Copyright (C) 2024, IBM

include "base_header.pxi"

cdef extern from "../src/grouping.hpp":

    void offdiag_term_sort(QubitOperator_t& oper) nogil

    