############
Installation
############

Requirements
============

Fulqrum is a mix of C++ and Python, glued together by a Cython interface.  

Python
~~~~~~

.. csv-table::
   :header: "Package", "Minimum version", "Description"
   :widths: 15, 10, 30

   "boost", 1.85, "We use the ``dynamic_bitset`` types for storing bit-strings"
   "NumPy", 1.25, "Used as array interface between Python and C++"
   "Cython", 3.0.5, "The glue between C++ and Python"
   "psutil", "any", "Get system information such as free memory"
   "orjson", "any", "IO for saving operators"

C++
~~~~

Fulqrum is based on C++17 and requires having OpenMP 3.0+.
Getting OpenMP is straightforward on Linux and Windows, but does require installing
LLVM via Homebrew on OSX.


Build files locally
===================

For testing we only require building the source files locally using:

.. code-block:: bash

    python setup.py build_ext --inplace

you can add also use env flags on Linux and OSX for such things as specifying the compiler, 
e.g. ``CC=clang CXX=clang++`` or setting the target architecture like ``FULQRUM_ARCH=znver4``.


Installation on Linux
=====================

Installation on Linux is simple:

.. code-block:: bash

    pip install .


Installation on OSX
===================

On OSX, to get OpenMP, one should install llvm using homebrew:

.. code-block:: bash

    brew install llvm

Then installation of Fulqrum with openmp can be accomplished using a call like:

.. code-block:: bash

    CC=clang CXX=clang++ pip install .


Installation on Windows
=======================

I have no idea how to set env vars on Windows, so I just do:

.. code-block:: bash

    python setup.py install
