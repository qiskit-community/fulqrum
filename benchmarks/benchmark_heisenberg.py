import time
import argparse

import numpy as np
import scipy.sparse.linalg as spla
import fulqrum as fq
from qiskit.transpiler import CouplingMap

import logging
logging.basicConfig(level=logging.INFO)



def main(tol=None):
    # Build 1541-qubit coupling map
    cmap = CouplingMap.from_heavy_square(23)
    num_qubits = cmap.size()

    # Generate Hamiltonian
    H = fq.QubitOperator(num_qubits, [])
    touched_edges = set({})
    coeffs = [1/2, 1/2, 1]
    for edge in cmap.get_edges():
        if edge[::-1] not in touched_edges:
            H += fq.QubitOperator(num_qubits, [("XX", edge, coeffs[0]), 
                                            ("YY", edge, coeffs[1]), 
                                            ("ZZ", edge, coeffs[2])])
            touched_edges.add(edge)

    # 1 million Pseudo counts
    counts = {}
    for kk in range(int(1e6)):
        counts[bin(kk)[2:].zfill(num_qubits)] = 1

    # Solve eigenproblem (can substitute scipy.sparse.linalg.eigsh)
    S = fq.Subspace(counts)
    Hsub = fq.SubspaceHamiltonian(H, S)
    st = time.perf_counter()
    evals, _ = spla.eigsh(Hsub, k=1, which='SA', tol=0 if not tol else tol, v0=np.ones(len(S),dtype=complex))
    return (time.perf_counter() - st), evals[0]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', nargs='?', const=0, type=float)
    args = parser.parse_args()
    print(main(args.tol))
