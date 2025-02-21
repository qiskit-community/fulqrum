
![fulqrum](https://github.com/user-attachments/assets/33a42162-2545-426a-ab7e-653030643e7b)

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
