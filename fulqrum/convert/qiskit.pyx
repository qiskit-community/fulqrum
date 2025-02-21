# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.string cimport string
from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.qubit_operator cimport STR_TO_IND as QUBIT_STR_TO_IND
from fulqrum.core.base cimport OperatorTerm

cdef const OperatorTerm EmptyOperatorTerm


@cython.boundscheck(False)
def SparsePauliOp_to_QubitOperator(object op):
    """Convert a Qiskit SparsePauliOp to QubitOperator

    Parameters:
        op (SparsePauliOp): Input operator

    Returns:
        QubitOperator: Converted operator
    """
    cdef OperatorTerm term
    cdef str string, s
    cdef complex coeff
    cdef size_t idx = 0
    cdef QubitOperator out = QubitOperator(op.num_qubits)
    for string, coeff in op.label_iter():
        term = EmptyOperatorTerm
        term.coeff = coeff
        idx = 0
        for s in string[::-1]:
            if s != 'I':
                term.indices.push_back(idx)
                term.indices.push_back(QUBIT_STR_TO_IND[s])
            idx += 1
        out.terms.push_back(term)
    out.sorted = True
    return out

