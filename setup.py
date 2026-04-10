# This code is a part of Fulqrum.
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

"""Fulqrum : A generalized quantum subspace eigensolver"""

import os
import sys
import setuptools
from setuptools.command.build_ext import build_ext

import numpy as np
from Cython.Build import cythonize

n_parallel_threads = int(os.environ.get("FQ_BUILD_PARALLEL", "1"))
using_inplace = "--inplace" in sys.argv


class ParallelBuildExt(build_ext):
    def build_extensions(self):
        self.parallel = n_parallel_threads
        build_ext.build_extensions(self)


ROOT = "fulqrum"

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

PACKAGES = setuptools.find_packages()
PACKAGE_DATA = {
    f"{ROOT}/core": ["*.pxd"],
    f"{ROOT}/core/src": ["*.hpp"],
    f"{ROOT}/include": ["*.hpp"],
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
    "simple",
]

CYTHON_MODULES = [
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.core",
    f"{ROOT}.convert",
    f"{ROOT}.convert",
    f"{ROOT}.convert",
    f"{ROOT}.utils",
    f"{ROOT}.core",
    f"{ROOT}.ramps",
]
CYTHON_SOURCE_DIRS = [
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/core",
    f"{ROOT}/convert",
    f"{ROOT}/convert",
    f"{ROOT}/convert",
    f"{ROOT}/utils",
    f"{ROOT}/core",
    f"{ROOT}/ramps",
]

# Add openmp flags
OPTIONAL_FLAGS = []
OPTIONAL_ARGS = []

if sys.platform == "win32":
    OPTIONAL_FLAGS = ["/openmp:llvm"]
else:
    OPTIONAL_FLAGS = ["-fopenmp"]
    OPTIONAL_ARGS.append("-fopenmp")

if os.getenv("FQ_ARCH", False) and sys.platform != "win32":
    if sys.platform == "darwin":
        # This is needed to set the flag for ARM processors on OSX
        # M1 = apple-a14, M2 = apple-a15, M3 = apple-a16, M4 = apple-m4
        OPTIONAL_FLAGS.append("-mcpu=" + os.getenv("FQ_ARCH"))
    else:
        OPTIONAL_FLAGS.append("-march=" + os.getenv("FQ_ARCH"))
        OPTIONAL_FLAGS.append("-mtune=" + os.getenv("FQ_ARCH"))

INCLUDE_DIRS = [np.get_include()] + [
    "third-party",
    "qiskit-addon-sqd-hpc/include",
]
# Extra link args
LINK_FLAGS = []
# If on Win and not in MSYS2 (i.e. Visual studio compile)
if sys.platform == "win32" and os.environ.get("MSYSTEM", None) is None:
    COMPILER_FLAGS = ["/O2", "/std:c++17"]
# Everything else
else:
    COMPILER_FLAGS = ["-O3", "-std=c++17"]

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
        EXT_MODULES,
        nthreads=0 if using_inplace else n_parallel_threads,  # to avoid race condition
        language_level=3,
        force=True,
        compiler_directives={"embedsignature": True},
    ),
    cmdclass={"build_ext": ParallelBuildExt},
)
