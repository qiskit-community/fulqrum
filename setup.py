# Fulqrum
# Copyright (C) 2024, IBM

"""Fulqrum : A sophisticated take on quantum operators
"""

import os
import sys
import setuptools

import numpy as np
from Cython.Build import cythonize

MAJOR = 0
MINOR = 0
MICRO = 1

VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

PACKAGES = setuptools.find_packages()
PACKAGE_DATA = {
    "fulqrum/core": ["*.pxd"],
    "fulqrum/core/src": ["*.hpp"],
}
DOCLINES = __doc__.split("\n")
DESCRIPTION = DOCLINES[0]
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as readme:
    LONG_DESCRIPTION = readme.read()

CYTHON_EXTS = ["qubit_operator", "subspace", "spmv", "string_funcs"]
CYTHON_MODULES = [
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.core",
    "fulqrum.test",
]
CYTHON_SOURCE_DIRS = [
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/core",
    "fulqrum/test",
]

# Add openmp flags
OPTIONAL_FLAGS = []
OPTIONAL_ARGS = []
WITH_OMP = False
for _arg in sys.argv:
    if _arg == "--openmp":
        WITH_OMP = True
        sys.argv.remove(_arg)
        break
if WITH_OMP or os.getenv("FULQRUM_OPENMP", False):
    WITH_OMP = True
    if sys.platform == "win32":
        OPTIONAL_FLAGS = ["/openmp:llvm"]
    else:
        OPTIONAL_FLAGS = ["-fopenmp"]
        OPTIONAL_ARGS.append("-fopenmp")

if os.getenv("FULQRUM_ARCH", False) and sys.platform != "win32":
    OPTIONAL_FLAGS.append("-march=" + os.getenv("FULQRUM_ARCH"))

INCLUDE_DIRS = [np.get_include()]
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
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    EXT_MODULES.append(mod)


def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except:  # pylint: disable=bare-except
        git_str = ""
    else:
        if git_str == "+":  # fixes setuptools PEP issues with versioning
            git_str = ""
    return git_str


FULLVERSION = VERSION


def write_version_py(filename="/fulqrum/version.py"):
    cnt = """\
# THIS FILE IS GENERATED FROM FULQRUM SETUP.PY
# pylint: disable=missing-module-docstring
short_version = '%(version)s'
version = '%(fullversion)s'
openmp = %(with_omp)s
"""
    a = open(os.path.dirname(__file__) + filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "fullversion": FULLVERSION,
                "with_omp": str(WITH_OMP),
            }
        )
    finally:
        a.close()


local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, "fulqrum"))  # to retrive _version

# always rewrite _version
if os.path.exists(os.path.dirname(__file__) + "/fulqrum/version.py"):
    os.remove(os.path.dirname(__file__) + "/fulqrum/version.py")

write_version_py()


setuptools.setup(
    name="fulqrum",
    version=VERSION,
    python_requires=">=3.10",
    packages=PACKAGES,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="",
    author="Paul Nation",
    author_email="paul.nation@ibm.com",
    license="IBM",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=REQUIREMENTS,
    package_data=PACKAGE_DATA,
    ext_modules=cythonize(EXT_MODULES, language_level=3),
    include_package_data=True,
    zip_safe=False,
)
