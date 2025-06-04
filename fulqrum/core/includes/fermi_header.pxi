# Fulqrum
# Copyright (C) 2024, IBM

include "base_header.pxi"

cdef extern from "../src/fermi.hpp":
    void jw_term(FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)

    void extended_jw_transform(FermionicOperator_t& fermi, QubitOperator_t& out,
                               size_t num_terms)