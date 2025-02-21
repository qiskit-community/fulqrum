

# fulqrum
Operator methods for quantum subpspace eigenproblems

This is very much a work in progress, and not suitable for human or animal consumption.


## Building files locally

In order to run the unittests locally, it is necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
where you can add any number of additional env flags such as `FULQRUM_OPENMP=1`.


## Installing

To enable OpenMP one must have an OpenMP 4.0+ enabled compiler and install with:

```bash
FULQRUM_OPENMP=1 pip install .
```

### OpenMP on OSX

On OSX one must install GCC using homebrew:

```bash
brew install gcc
```

Then installation with openmp can be accomplished using a call like:

```bash
FULQRUM_OPENMP=1 CC=gcc-14 CXX=g++14 pip install .
```


## Example: 1541 qubit spin-lattice

This takes six minutes on my desktop:

```python

import time
import numpy as np
from qiskit.transpiler import CouplingMap

import scipy.sparse.linalg as spla
import primme

import fulqrum as fq

# Build 1541-qubit coupling map
cmap = CouplingMap.from_heavy_square(23)
num_qubits = cmap.size()

# Generate Hamiltonian
H = fq.QubitOperator(num_qubits, [])
touched_edges = set({})
for edge in cmap.get_edges():
    if edge[::-1] not in touched_edges:
        coeffs = np.random.random(size=3)
        H += fq.QubitOperator(num_qubits, [("XX", edge, coeffs[0]), 
                                           ("YY", edge, coeffs[1]), 
                                           ("ZZ", edge, coeffs[2])])
        touched_edges.add(edge)

# 1 million Pseudo counts
counts = {}
for kk in range(int(1e6)):
    counts[bin(kk)[2:].zfill(num_qubits)] = 1

# Solve eigenproblem (can substitute scipy.sparse.linalg.eigsh)
st = time.perf_counter()
S = fq.Subspace(counts)
Hsub = fq.SubspaceHamiltonian(H, S)
evals, _ = primme.eigsh(Hsub, k=1, which='SA', method='PRIMME_DEFAULT_MIN_MATVECS')
```
