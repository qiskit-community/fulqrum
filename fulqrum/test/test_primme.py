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
# pylint: disable=no-name-in-module
import numpy as np
import fulqrum as fq
import primme
from qiskit.transpiler import CouplingMap

# Build 16-qubit coupling map
cmap = CouplingMap.from_grid(4, 4)
num_qubits = cmap.size()

# Generate Hamiltonian
H = fq.QubitOperator(num_qubits, [])
touched_edges = set({})
coeffs = [-1 / 2, -1 / 2, -1]
for edge in cmap.get_edges():
    if edge[::-1] not in touched_edges:  # Only add edge once to Hamiltonian
        touched_edges.add(edge)
        H += fq.QubitOperator(
            num_qubits,
            [("XX", edge, coeffs[0]), ("YY", edge, coeffs[1]), ("ZZ", edge, coeffs[2])],
        )
# Subspace
counts = []
for kk in range(2**num_qubits):
    counts.append(bin(kk)[2:].zfill(num_qubits))
S = fq.Subspace([counts])

# Subspace Hamiltonian and guess vec
Hsub = fq.SubspaceHamiltonian(H, S)
v0 = np.ones((S.size(), 1), dtype=Hsub.dtype)


def test_primme_matrix_free():
    """Validate PRIMME matrix-free solving"""
    evals, _ = primme.eigsh(Hsub, k=1, which="SA", v0=v0)
    assert np.allclose(evals, np.array([-24.0]))


def test_primme_csrfast():
    """Validate PRIMME csrfast solving"""
    evals, _ = primme.eigsh(Hsub.to_csr_linearoperator_fast(), k=1, which="SA", v0=v0)
    assert np.allclose(evals, np.array([-24.0]))
