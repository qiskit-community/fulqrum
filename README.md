
![fulqrum_logo](https://github.ibm.com/user-attachments/assets/889bc5f1-93e7-46c4-91e1-aa64ab8c3391)


# fulqrum
Operator methods for quantum subpspace eigenproblems.

Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subpspace eigenproblems over extended alphabets for those of us without access to high-performance computing (HPC) resources.  To accomplish this, Fulqrum utilizes a novel matrix-free method for performing the matrix-vector computation that is at the core of all sparse eigensolving methods.

Working over extended (i.e. non-Pauli) alphabets allows Fulqrum to work for both Bosonic and Fermionic problems.  Fermionic problems can be cast into Bosonic ones in a one-to-one manner using an extended Jordan-Wigner transformation, and the properties of extended operators can be used to further reduce the computational costs.


## Installation


## Requirements

Outside of standard packages, currently Fulqrum requires the Boost library and OpenMP v3+ . 

If using `conda` then adding Boost can be done using:

```bash
conda install boost
```

and the required include files should be automatically found on Linux and OSX.


## Building files locally

In order to run the unittests locally, it is only necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
you can add also use env flags such as `CC=clang CXX=clang++` or set the target architecture using `FULQRUM_ARCH=znver4` (or whatever you have)

## Installation on Linux

Installation on Linux is simple:

```bash
pip install .
```

### Installation on OSX

On OSX, to get OpenMP, one should install llvm using homebrew:

```bash
brew install llvm
```

Then installation of Fulqrum with openmp can be accomplished using a call like:

```bash
CC=clang CXX=clang++ pip install .
```

### Installation on Windows [Currently not working]

I have no idea how to set env vars on Windows, so I just do:

```bash
python setup.py install
```

## Examples

> [!CAUTION]
> The SciPy sparse `eigs` and `eigsh` solvers can conflict with the OpenMP used by Fulqrum if the underlying blas library is based on OpenBlas.  In these cases one should try to set the env variable: `OPENBLAS_NUM_THREADS=1` to begin with.  Increasing this number higher than this might increase the overall runtime on some machines.


### 1541 qubit spin-lattice

```python
import time
import primme
from qiskit.transpiler import CouplingMap
import fulqrum as fq

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
evals, _ = primme.eigsh(Hsub, k=1, which='SA', method='PRIMME_DEFAULT_MIN_MATVECS')
```
