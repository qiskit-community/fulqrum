import time
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import fulqrum as fq

import logging
logging.basicConfig(level=logging.INFO)


def full_subspace(num_qubits):
    out = {}
    for val in range(2**num_qubits):
        out[bin(val)[2:].zfill(num_qubits)] = None
    return out


def main(tol=None):
    path = Path(__file__).parent / "data/h2o.json"
    fop = fq.FermionicOperator.from_json(path)
    op = fop.extended_jw_transformation()
    new_op = op.combine_repeated_terms()

    dist = full_subspace(new_op.width)

    S = fq.Subspace(dist)
    Hsub = fq.SubspaceHamiltonian(new_op, S)

    st = time.perf_counter()
    evals, _ = spla.eigsh(Hsub, k=1, which='SA',tol=0 if not tol else tol, v0=np.ones(len(S),dtype=complex))
    return (time.perf_counter() - st), evals[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tol', nargs='?', const=0, type=float)
    args = parser.parse_args()
    print(main(args.tol))
