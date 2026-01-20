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
import primme
import qiskit_addon_fulqrum as fq

from benchmarks.config import Configuration

TOL = Configuration.options["solver"]["tol"]

ANS = -84.20635059311753


def full_subspace(num_qubits):
    out = {}
    for val in range(2**num_qubits):
        out[bin(val)[2:].zfill(num_qubits)] = None
    return out


def test_h2o_full_csr_fast(benchmark):
    @benchmark
    def result():
        path = Path(__file__).parent / "data/h2o.json"
        fop = fq.FermionicOperator.from_json(path)
        jw_start = time.perf_counter()
        op = fop.extended_jw_transformation()
        jw_end = time.perf_counter()

        dist = full_subspace(op.width)

        subspace_start = time.perf_counter()
        S = fq.Subspace([list(dist.keys())])
        subspace_end = time.perf_counter()

        hsub_start = time.perf_counter()
        Hsub = fq.SubspaceHamiltonian(op, S)
        hsub_end = time.perf_counter()

        oper_start = time.perf_counter()
        A = Hsub.to_csr_linearoperator_fast()
        oper_end = time.perf_counter()

        diag = Hsub.diagonal_vector()
        min_idx = np.where(diag == np.min(diag))[0][0]

        v0 = np.zeros((len(S), 1), dtype=float)
        v0[min_idx] = 1

        solver_start = time.perf_counter()
        evals, _ = primme.eigsh(
            A, k=1, which="SA", tol=TOL, v0=v0, method="PRIMME_DEFAULT_MIN_MATVECS"
        )
        solver_end = time.perf_counter()
        return (
            evals[0],
            jw_end - jw_start,
            subspace_end - subspace_start,
            hsub_end - hsub_start,
            oper_end - oper_start,
            solver_end - solver_start,
        )

    benchmark.extra_info["jw_time"] = result[1]
    benchmark.extra_info["subspace_time"] = result[2]
    benchmark.extra_info["hsub_time"] = result[3]
    benchmark.extra_info["operator_time"] = result[4]
    benchmark.extra_info["eigen_time"] = result[5]
    assert True


def test_h2o_full_matrix_free(benchmark):
    @benchmark
    def result():
        path = Path(__file__).parent / "data/h2o.json"
        fop = fq.FermionicOperator.from_json(path)
        jw_start = time.perf_counter()
        op = fop.extended_jw_transformation()
        jw_end = time.perf_counter()

        dist = full_subspace(op.width)

        subspace_start = time.perf_counter()
        S = fq.Subspace([list(dist.keys())])
        subspace_end = time.perf_counter()

        hsub_start = time.perf_counter()
        Hsub = fq.SubspaceHamiltonian(op, S)
        hsub_end = time.perf_counter()

        diag = Hsub.diagonal_vector()
        min_idx = np.where(diag == np.min(diag))[0][0]

        v0 = np.zeros((len(S), 1), dtype=float)
        v0[min_idx] = 1

        solver_start = time.perf_counter()
        evals, _ = primme.eigsh(
            Hsub, k=1, which="SA", tol=TOL, v0=v0, method="PRIMME_DEFAULT_MIN_MATVECS"
        )
        solver_end = time.perf_counter()
        return (
            evals[0],
            jw_end - jw_start,
            subspace_end - subspace_start,
            hsub_end - hsub_start,
            solver_end - solver_start,
        )

    benchmark.extra_info["jw_time"] = result[1]
    benchmark.extra_info["subspace_time"] = result[2]
    benchmark.extra_info["hsub_time"] = result[3]
    benchmark.extra_info["eigen_time"] = result[4]
    assert True
