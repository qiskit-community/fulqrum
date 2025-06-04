# Fulqrum
# Copyright (C) 2024, IBM

include "base_header.pxi"

cdef extern from "../src/fermi.hpp":
    void jw_term(FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)
