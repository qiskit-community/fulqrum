# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from fulqrum.core.qubit_operator cimport QubitOperator



@cython.boundscheck(False)
def sparsepauli_to_fulqrum(object op):
    """Convert a Qiskit SparsePauliOp to QubitOperator

    Parameters:
        op (SparsePauliOp): Input operator

    Returns:
        QubitOperator: Converted operator
    """
    cdef str label
    cdef complex coeff
    cdef QubitOperator out = QubitOperator(op.num_qubits)
    for label, coeff in op.label_iter():
        out += QubitOperator.from_label(label, coeff)
    return out
