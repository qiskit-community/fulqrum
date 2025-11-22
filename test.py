import time
import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import fulqrum as fq
import primme

path = "data/ch4_dimer.json"
fop = fq.FermionicOperator.from_json(path)
op = fop.extended_jw_transformation()

num_samples = 1_000_000


counts = {"000000000000000011111111000000000000000011111111":1,
"000000000000000111110111000000000000000011111111":1,
"000000000000000111111011000000000000000011111111":1,
"000000000000000011111111000000000000000111110111":1,
"000000000000000111110111000000000000000111110111":1,
"000000000000000111111011000000000000000111110111":1,
"000000000000000011111111000000000000000111111011":1,
"000000000000000111110111000000000000000111111011":1,
"000000000000000111111011000000000000000111111011":1}

S = fq.Subspace(counts)

Hsub = fq.SubspaceHamiltonian(op, S)


diag, off = op.split_diagonal()

diag_vec = Hsub.diagonal_vector()
min_idx = np.where(diag_vec == diag_vec.min())[0]
min_energy = diag_vec[min_idx]



atol = 1e-14 # threshold for non-degenerate energy
split = np.abs(diag_vec - min_energy)
sort_idx = np.argsort(split)
min_nondegen_split_idx = np.where(split[sort_idx] > atol)[0][0]
min_split = split[sort_idx[min_nondegen_split_idx]]

worst_amps = off.worst_case_offdiag_group_amplitudes()

max_energy_contribution = abs(worst_amps**2/min_split)/abs(min_energy)

energy_atol = 1e-6

good_groups = np.where(max_energy_contribution > energy_atol)[0]


off_group_ptrs = off.group_ptrs()

new_op = diag.copy()
for idx in good_groups:
    start = off_group_ptrs[idx]
    stop = off_group_ptrs[idx+1]
    for kk in range(start, stop):
        new_op += off[kk]
    break # first good group only


Hsub2 = fq.SubspaceHamiltonian(new_op, S)

M = Hsub2.to_csr_linearoperator_fast(verbose=True)

# Set good initial state
x0 = 1e-8*np.ones((S.size(),1), dtype=Hsub2.dtype)
x0[min_idx] = 1

st = time.perf_counter()
evals, evecs = primme.eigsh(M, k=1, which='SA', tol=1e-8, v0=x0)
print('solve time', time.perf_counter() - st)

print('eval:', evals[0])