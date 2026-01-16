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

import time
import pytest
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import qiskit_addon_fulqrum as fq

ANS = -84.20635059311753

def full_subspace(num_qubits):
    out = {}
    for val in range(2**num_qubits):
        out[bin(val)[2:].zfill(num_qubits)] = None
    return out


def test_h2o_full(benchmark):
    @benchmark
    def result():
        path = Path(__file__).parent / "data/h2o.json"
        fop = fq.FermionicOperator.from_json(path)
        jw_start = time.perf_counter()
        op = fop.extended_jw_transformation()
        jw_end = time.perf_counter()

        dist = full_subspace(op.width)

        S = fq.Subspace([list(dist.keys())])
        Hsub = fq.SubspaceHamiltonian(op, S)

        solver_start = time.perf_counter()
        evals, _ = spla.eigsh(
            Hsub,
            k=1,
            which="SA",
            tol=0,
            v0=np.ones(len(S), dtype=float),
        )
        solver_end = time.perf_counter()
        return evals[0], jw_end - jw_start, solver_end - solver_start

    benchmark.extra_info["jw_time"] = result[1]
    benchmark.extra_info["eigen_time"] = result[2]
    assert np.abs((result[0] - ANS)/ANS) < 1e-14
