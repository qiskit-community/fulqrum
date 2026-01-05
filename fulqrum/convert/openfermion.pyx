# This code is a Qiskit project.
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
# cython: c_string_type=unicode, c_string_encoding=UTF-8, language_level=3
cimport cython
from fulqrum.core.qubit_operator cimport QubitOperator
from fulqrum.core.fermi_operator cimport FermionicOperator

cdef inline size_t size_max(size_t x, size_t y):
    """Return the max of two size_t variables
    """
    if x > y:
        return x
    return y

cdef inline dict qubit_reorder_map(size_t num_qubits):
    """OpenFermion has a differnt qubit ordering |... b2 a2 b1 a1 b0 a0>.
        Qiskit and fulqrum use another ordering |..b2 b1 b0 ... a2 a1 a0>.
        This function creates map between these two.
    """
    cdef size_t half_len
    cdef list data, b_half, a_half, interleaved
    cdef dict mapper = {}

    half_len = num_qubits // 2
    data = list(range(num_qubits))[::-1]
    b_half = data[:half_len]
    a_half = data[half_len:]
    
    interleaved = []
    for i in range(half_len):
        interleaved.append(b_half[i])
        interleaved.append(a_half[i])

    for old, new in zip(interleaved, data):
        mapper[new] = old
    
    return mapper

@cython.boundscheck(False)
def openfermion_qubit_op_to_fulqrum(object op):
    """Convert a Openfermion QubitOperator to Fulqrum QubitOperator

    Parameters:
        op (openfermion.ops.QubitOperator): Input operator

    Returns:
        QubitOperator: Converted operator
    """
    cdef str label
    cdef complex coeff
    cdef size_t max_idx = 0
    cdef int num_qubits
    cdef tuple paulis
    cdef list chars, data, b_half, a_half, interleaved
    cdef dict mapper
    
    # Determine number of qubits used in operator
    for key in op.terms.keys():
        for pair in key:
            max_idx = size_max(max_idx, pair[0])
    num_qubits = max_idx+1

    if num_qubits % 2 != 0:
        raise ValueError(
            f"Number of qubits must be even in a QubitOperator. "
            f"number of qubits: {num_qubits} is odd."
        )

    cdef QubitOperator out = QubitOperator(num_qubits)

    mapper = qubit_reorder_map(num_qubits)

    for paulis, coeff in op.terms.items():
        chars = ["I"] * num_qubits

        for idx, pauli in paulis:
            idx_new = mapper[idx]
            chars[idx_new] = pauli
        
        label = ''.join(chars)
        out += QubitOperator.from_label(label[::-1], coeff)
    
    return out

@cython.boundscheck(False)
def openfermion_fermi_op_to_fulqrum(object op):
    """Convert a Openfermion FermionOperator to Fulqrum FermionicOperator

    Parameters:
        op (FermionOperator): Input operator

    Returns:
        FermionicOperator: Converted operator
    """
    cdef str label
    cdef complex coeff
    cdef size_t num_qubits, qubit, half_len, max_idx = 0
    cdef tuple terms
    cdef list data, b_half, a_half, interleaved, tmp
    cdef dict mapper
    
    # Determine number of qubits used in operator
    for key in op.terms.keys():
        for qubit, _ in key:
            max_idx = size_max(max_idx, qubit)
    num_qubits = max_idx+1

    if num_qubits % 2 != 0:
        raise ValueError(
            "Fermionic Operators with odd number of modes are not supported yet."
        )
    
    cdef FermionicOperator out = FermionicOperator(num_qubits)

    mapper = qubit_reorder_map(num_qubits)

    for terms, coeff in op.terms.items():
        tmp = []
        for qubit, symbol in terms:
            if symbol == 1:
                symbol = "+"
            else:
                symbol = "-"

            qubit_new = mapper[qubit]
            
            tmp.append(f'{symbol}:{qubit_new}')
        
        label = " ".join(tmp)
        out += FermionicOperator.from_label(num_qubits, label, coeff)
    
    return out
