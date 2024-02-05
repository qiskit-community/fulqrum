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
                term.operators.push_back(size_uchar_pair(idx, QUBIT_STR_TO_IND[s]))
            idx += 1
        out.terms.push_back(term)
    out.sorted = True
    return out


@cython.boundscheck(False)
def FermionicOp_to_FermionicOperator(object op):
    """Convert a Qiskit FermionicOp to FermionicOperator

    Parameters:
        op (FermionicOp): Input operator

    Returns:
        QubitOperator: FermionicOperator
    """
    cdef OperatorTerm term
    cdef FermionicOperator out = FermionicOperator(op.num_spin_orbitals)
    cdef tuple qiskit_term, item
    cdef size_t kk
    for qiskit_term in op.terms():
        term = EmptyOperatorTerm
        term.coeff = qiskit_term[1]
        for item in qiskit_term[0]:
            term.operators.push_back(size_uchar_pair(item[1], FERMI_STR_TO_IND[item[0]]))
        out.terms.push_back(term)
    return out
