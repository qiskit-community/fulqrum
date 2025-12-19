
![fulqrum_logo](https://github.ibm.com/user-attachments/assets/889bc5f1-93e7-46c4-91e1-aa64ab8c3391)


# fulqrum
Operator methods for quantum subpspace eigenproblems.

Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subpspace eigenproblems over extended alphabets for those of us without access to high-performance computing (HPC) resources.  To accomplish this, Fulqrum utilizes a novel matrix-free method for performing the matrix-vector computation that is at the core of all sparse eigensolving methods.

Working over extended (i.e. non-Pauli) alphabets allows Fulqrum to work for both Bosonic and Fermionic problems.  Fermionic problems can be cast into Bosonic ones in a one-to-one manner using an extended Jordan-Wigner transformation, and the properties of extended operators can be used to further reduce the computational costs.


## Installation


### Requirements

Outside of standard packages, currently Fulqrum requires the Boost library and OpenMP v3+ . 

If using `conda` then adding Boost can be done using:

```bash
conda install boost
```

and the required include files should be automatically found on Linux and OSX.

### Git submodule

Fulqrum uses [`qiskit-addon-sqd-hpc`](https://github.com/Qiskit/qiskit-addon-sqd-hpc) as a Git submodule. Therefore, we can do either of the following to get actual and required files from that repo:

Clone Fulqrum with `--recurse-submodules` flag, and it will also clone files from the submodule.
```bash
git clone --recurse-submodules https://github.ibm.com/ibm-q-research/fulqrum.git
```

OR

After cloning Fulqrum normally, do a submodule update.
```bash
git clone https://github.ibm.com/ibm-q-research/fulqrum.git
cd fulqrum
git submodule update --init --recursive
```

### Building files locally

In order to run the unittests locally, it is only necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
you can add also use env flags such as `CC=clang CXX=clang++` or set the target architecture using `FULQRUM_ARCH=znver4` (or whatever you have)

### Installation on Linux

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

## Notes on new `Supspace` input format
Two modes of input to subspace is supported:
1. **Half-string mode:** In this mode, the input `subspace_strs` to the `Subspace()` is a length-2 
`tuple[list[str], list[str]]`, where the first element represents _alpha_ strings, and the second
one represents _beta_ strings. Internally, `fulqrum` will sort each list and perform a Cartesian product between the two lists to construct the full subspace. A full bitstring will be represented by `beta half str + alpha half str`, and the final length of subspace bitstring will be `len(beta half string) + len(alpha half string)`. For example, `[['100', '001, '010'], ['110', '000']]` is a valid input `subspace_strs` in half-string mode. The final subspace size will be `2 * 3 = 6`, with each bitstrings with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications. For instance, many popular chemistry packages such as PySCF takes alpha and beta strings separately and extends the subspace by taking Cartesian product. This mode enables PySCF-like funcationality and keeps the memory requirment in check.

2. **Full-string mode:** In this mode, `subspace_strs` has to be length-1 `tuple[list[str]]`. In this mode, no Cartesian product is taken internally, and the supplied _full_ bitstrings are used as is after sorting.

**Note:** There is no special flag to denote half-string vs. full-string mode. `Fulqrum` will compute the lenght of the input `subspace_strs` and detect half-string or full-string mode automatically to construct the subspace. If `subspace_strs` has a different length other than 1 or 2 or its element(s) is not `list[str]`, it will throw an `TypeError`.

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
counts = []
for kk in range(int(1e6)):
    counts.append(bin(kk)[2:].zfill(num_qubits))

# Solve eigenproblem (can substitute scipy.sparse.linalg.eigsh)
S = fq.Subspace([counts])
Hsub = fq.SubspaceHamiltonian(H, S)
evals, _ = primme.eigsh(Hsub, k=1, which='SA', method='PRIMME_DEFAULT_MIN_MATVECS')
```
