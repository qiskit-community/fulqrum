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
cimport cython
from ..core.qubit_operator cimport QubitOperator
from ..core.fermi_operator cimport FermionicOperator


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

@cython.boundscheck(False)
def qiskit_nature_fermi_op_to_fulqrum(object op):
    """Convert a Qiskit Nature's FermionicOp to Fulqrum FermionicOperator

    Parameters:
        op (FermionicOp): Input operator from qiskit-nature

    Returns:
        FermionicOperator: Converted operator
    """
    cdef str label, term, item, qubit
    cdef complex coeff
    cdef size_t num_qubits
    cdef list tmp, term_split
    cdef size_t qubit_int
    
    num_qubits = op.num_spin_orbitals

    if num_qubits % 2 != 0:
        raise ValueError(
            "Fermionic Operators with odd number of modes are not supported yet."
        )
    
    cdef FermionicOperator out = FermionicOperator(num_qubits)

    for term, coeff in op.items():
        term_split = term.split()
        tmp = []
        for item in term_split:
            symbol, qubit = item.split("_")
            qubit_int = int(qubit)
            if qubit_int > (num_qubits-1):
                raise ValueError(
                    f"Qubit index ({qubit}) cannot be greater than "
                    f"num spin orbitals {num_qubits}."
                )
            tmp.append(f'{symbol}:{int(qubit)}')
        
        label = " ".join(tmp)
        out += FermionicOperator.from_label(num_qubits, label, coeff)
    
    return out
