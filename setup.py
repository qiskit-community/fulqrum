# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum : A sophisticated take on quantum operators"""

import os
import sys
import setuptools

import numpy as np
from Cython.Build import cythonize


with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

PACKAGES = setuptools.find_packages()
PACKAGE_DATA = {
    "fulqrum/core": ["*.pxd"],
    "fulqrum/core/src": ["*.hpp"],
}

CYTHON_EXTS = [
    "subspace",
    "bitset",
    "bitset_view",
    "qubit_operator",
    "fermi_operator",
    "spmv",
    "csr",
    "csrlike",
    "qiskit",
    "openfermion",
    "integrals",
    "matrix",
    "sqd",
    "simple"
]
CYTHON_MODULES = [
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.convert",
    "fulqrum.convert",
    "fulqrum.convert",
    "fulqrum.utils",
    "fulqrum.core",
    "fulqrum.ramps"
]
CYTHON_SOURCE_DIRS = [
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/convert",
    "fulqrum/convert",
    "fulqrum/convert",
    "fulqrum/utils",
    "fulqrum/core",
    "fulqrum/ramps"
]

# Add openmp flags
OPTIONAL_FLAGS = []
OPTIONAL_ARGS = []

if sys.platform == "win32":
    OPTIONAL_FLAGS = ["/openmp:llvm"]
else:
    OPTIONAL_FLAGS = ["-fopenmp"]
    OPTIONAL_ARGS.append("-fopenmp")

if os.getenv("FULQRUM_ARCH", False) and sys.platform != "win32":
    if sys.platform == "darwin":
        # This is needed to set the flag for ARM processors on OSX
        # M1 = apple-a14, M2 = apple-a15, M3 = apple-a16, M4 = apple-m4
        OPTIONAL_FLAGS.append("-mcpu=" + os.getenv("FULQRUM_ARCH"))
    else:
        OPTIONAL_FLAGS.append("-march=" + os.getenv("FULQRUM_ARCH"))
        OPTIONAL_FLAGS.append("-mtune=" + os.getenv("FULQRUM_ARCH"))

INCLUDE_DIRS = [np.get_include()] + [
    "qiskit-addon-sqd-hpc/include",
]
# Extra link args
LINK_FLAGS = []
# If on Win and not in MSYS2 (i.e. Visual studio compile)
if sys.platform == "win32" and os.environ.get("MSYSTEM", None) is None:
    COMPILER_FLAGS = ["/O2", "/std:c++17"]
# Everything else
else:
    COMPILER_FLAGS = ["-O3", "-std=c++17", "-ffast-math"]

EXT_MODULES = []
# Add Cython Extensions
for idx, ext in enumerate(CYTHON_EXTS):
    mod = setuptools.Extension(
        CYTHON_MODULES[idx] + "." + ext,
        sources=[CYTHON_SOURCE_DIRS[idx] + "/" + ext + ".pyx"],
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILER_FLAGS + OPTIONAL_FLAGS,
        extra_link_args=LINK_FLAGS + OPTIONAL_ARGS,
        language="c++",
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS", None),
        ],
    )
    EXT_MODULES.append(mod)



setuptools.setup(
    install_requires=REQUIREMENTS,
    package_data=PACKAGE_DATA,
    packages=PACKAGES,
    ext_modules=cythonize(
        EXT_MODULES, language_level=3, force=True, compiler_directives={"embedsignature": True}
    )
)
