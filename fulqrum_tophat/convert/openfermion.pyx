# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.string cimport string
from fulqrum_tophat.core.qubit_operator cimport QubitOperator
from fulqrum_tophat.core.qubit_operator cimport STR_TO_IND as QUBIT_STR_TO_IND
from fulqrum_tophat.core.fermi_operator cimport FermionicOperator
from fulqrum_tophat.core.fermi_operator cimport STR_TO_IND as FERMI_STR_TO_IND
from fulqrum_tophat.core.base cimport OperatorTerm, size_uchar_pair

cdef const OperatorTerm EmptyOperatorTerm


cdef inline size_t size_max(size_t x, size_t y):
    """Return the max of two size_t variables
    """
    if x > y:
        return x
    return y


@cython.boundscheck(False)
def OpenFermion_to_QubitOperator(object op):
    """Convert an OpenFermion QubitOperator to a Fulqrum QubitOperator

    Paramters:
        op (openfermion.QubitOperator): Input operator

    Returns:
        fulqrum.QubitOperator: Output operator
    """
    cdef OperatorTerm term
    cdef str string, s
    cdef complex coeff, val
    cdef size_t num_qubits, max_idx = 0
    cdef tuple key, pair
    # Determine number of qubits used in operator
    for key in op.terms.keys():
        for pair in key:
            max_idx = size_max(max_idx, pair[0])
    num_qubits = max_idx+1
    # Build the actual operator
    cdef QubitOperator out = QubitOperator(num_qubits)
    for key, val in op.terms.items():
        term = EmptyOperatorTerm
        term.coeff = val
        for pair in key:
            term.operators.push_back(size_uchar_pair(pair[0], FERMI_STR_TO_IND[pair[1]]))
        out.terms.push_back(term)
    return out
