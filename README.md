
![fulqrum](https://github.ibm.com/ibm-q-research/fulqrum/assets/152294/37fa23d5-4cad-4dde-bb0b-aa5f13c7fa56)

# fulqrum
Operator methods for quantum subpspace eigenproblems.

Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subpspace eigenproblems over extended alphabets for those of us without access to high-performance computing (HPC) resources.  To accomplish this, Fulqrum utilizes a novel matrix-free method for performing the matrix-vector computation that is at the core of all sparse eigensolving methods.

Working over extended (i.e. non-Pauli) alphabets allows Fulqrum to work for both Bosonic and Fermionic problems.  Fermionic problems can be cast into Bosonic ones in a one-to-one manner using an extended Jordan-Wigner transformation, and the properties of extended operators can be used to further reduce the computational costs.

This is very much a work in progress, and not suitable for human or animal consumption.


## Installation

> [!IMPORTANT]
> For some reason clang gives markedly better performance than gcc. Vendor specific compilers also give added performance, if available.

## Requirements

Outside of standard packages, currently Fulqrum requires the Boost library and OpenMP v3+ . 

If using `conda` then adding Boost can be done using:

```bash
conda install boost
```

and the required include files should be automatically found.


## Building files locally

In order to run the unittests locally, it is only necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
you can add also use env flags such as `FULQRUM_ARCH=znver4` (or whatever you arch is) if you like.

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

### Installation on Windows

I have no idea how to set env vars on Windows, so I just do:

```bash
python setup.py install
```

## Examples

### 1541 qubit spin-lattice

| Processor  | Platform    | Time (sec)  | Time w/grouping (sec)  | Time w/bitset (sec)  |
| :--------: | :---------: | :---------: | :--------------------: | :------------------: |
| AMD 7900   | Linux       | 335         | 187                    | 72                   |
| Intel 256v | Linux       | 937         | 521                    | 186                  |
| M1         | OSX         | 1569        | 876                    | 262                  |
| M1 Pro     | OSX         | 1170        | 652                    | 179                  |

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
