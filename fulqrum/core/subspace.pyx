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
# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.string cimport string
from libcpp cimport bool
from libc.math cimport abs

import itertools
import math
cimport cython
import numpy as np
cimport numpy as np

from ..exceptions import FulqrumError
from .bitset cimport bitset_t, to_string
from .bitset_view cimport BitsetView
from .bitset cimport Bitset
from .bitset_hashmap cimport BitsetHashMapWrapper

include "includes/base_header.pxi"
include "includes/bitset_utils_header.pxi"
include "includes/types.pxi"
include "includes/constants.pxi"


cdef class Subspace():
    """Hashmap representation of a quantum subspace over bit-strings obtained from
    sampling on a quantum computer (or simulator).

    Parameters:
        subspace_strs: Input bitstrings as either length-1 or length-2 tuple
            (or list of list(s)).

            Two modes of input to subspace is supported:
            1. Half-string mode: In this mode, the input `subspace_strs`
            to the `Subspace()` is a length-2  `tuple[list[str], list[str]]`,
            where the first element represents _alpha_ strings, and the second
            one represents _beta_ strings. Internally, `fulqrum` will sort each
            list and perform a Cartesian product between the two lists to construct
            the full subspace. A full bitstring will be represented by
            `beta half str + alpha half str`, and the final length of subspace
            bitstring will be `len(beta half string) + len(alpha half string)`.
            For example, `[['100', '001, '010'], ['110', '000']]` is a valid input
            `subspace_strs` in half-string mode.
            The final subspace size will be `2 * 3 = 6`, with each bitstrings
            with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications.

            2. Full-string mode: In this mode, `subspace_strs` has to be length-1
            `tuple[list[str]]`. In this mode, no Cartesian product is taken internally,
            and the supplied _full_ bitstrings are used as is after sorting.
        
        reserve_multiplier (float): We reserve a capacity for the Hash table that stores the
            subspace bit-strings, typically equal to the number of bit-strings. This
            argument allows a user to reserve more capacity than needed. While it consumes,
            more memory, it reduces collision during Hash table look-up leading to 
            minor speed-up.
            Default: 2.
        
        use_all_bitset_blocks (bool): If `use_all_bitset_blocks=False`, only first block of a
            bitset is used in hashing. If your bitsets are long and rarely share common
            prefixes, setting it to False speeds up execution. However, it likely that
            bitsets for practical cases will share common patterns. In that case, set
            `use_all_bitset_blocks` to True so that the whole bitset is used in the hash
            function. Although hashing n (> 1) blocks is slower than hashing a single block,
            full hashing usually leads to fewer collisions during Hash table look-up.
            Default: `True`.

    Example:

    .. jupyter-execute::

        import fulqrum as fq

        num_qubits = 5
        bitstrings = []
        for val in range(2**num_qubits):
            bitstrings.append(bin(val)[2:].zfill(num_qubits))

        fq.Subspace([bitstrings])

    """
    @cython.boundscheck(False)
    def __cinit__(self, subspace_strs, int reserve_multiplier=2, bool use_all_bitset_blocks=True):
        """
        args:
            subspace_strs: Input bitstrings as either length-1 or length-2 tuple
                (or list of list(s)).

                Two modes of input to subspace is supported:
                1. Half-string mode: In this mode, the input `subspace_strs`
                to the `Subspace()` is a length-2  `tuple[list[str], list[str]]`,
                where the first element represents _alpha_ strings, and the second
                one represents _beta_ strings. Internally, `fulqrum` will sort each
                list and perform a Cartesian product between the two lists to construct
                the full subspace. A full bitstring will be represented by
                `beta half str + alpha half str`, and the final length of subspace
                bitstring will be `len(beta half string) + len(alpha half string)`.
                For example, `[['100', '001, '010'], ['110', '000']]` is a valid input
                `subspace_strs` in half-string mode.
                The final subspace size will be `2 * 3 = 6`, with each bitstrings
                with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications.

                2. Full-string mode: In this mode, `subspace_strs` has to be length-1
                `tuple[list[str]]`. In this mode, no Cartesian product is taken internally,
                and the supplied _full_ bitstrings are used as is after sorting.
                
            reserve_multiplier: We reserve a capacity for the Hash table that stores the
                subspace bitstrings, typically equal to the number of bitstrings. This
                argument allows a user to reserve more capacity than needed. While it consumes,
                more memory, it reduces collision during Hash table look-up leading to 
                minor speed-up.
                Default: 2.

            use_all_bitset_blocks: If `use_all_bitset_blocks=False`, only first block of a
                bitset is used in hashing. If your bitsets are long and rarely share common
                prefixes, setting it to False speeds up execution. However, it likely that
                bitsets for practical cases will share common patterns. In that case, set
                `use_all_bitset_blocks` to True so that the whole bitset is used in the hash
                function. Although hashing n (> 1) blocks is slower than hashing a single block,
                full hashing usually leads to fewer collisions during Hash table look-up.
                Default: `True`.
        """
        if len(subspace_strs) == 0:
            return
        elif len(subspace_strs) == 1:
            iterator = subspace_strs[0]
            iterator.sort()
            num_qubits = len(next(iter(iterator)))
            size = len(iterator)
        elif len(subspace_strs) == 2:
            alpha_strs = subspace_strs[0]
            beta_strs = subspace_strs[1]
            alpha_strs.sort()
            beta_strs.sort()
            iterator = map(
                lambda ab: ab[1] + ab[0],
                itertools.product(alpha_strs, beta_strs)
            )
            num_qubits = len(next(iter(alpha_strs))) + len(next(iter(beta_strs)))
            size = len(alpha_strs) * len(beta_strs)
        else:
            raise ValueError("bitstrings are wrongly formatted")
        
        if num_qubits > (2 ** 16):
            raise ValueError(
                f"Number of qubits must be <= 2 ** 16 ( = {2 ** 16}). "
                f"Current number of qubits = {num_qubits}"
            )

        self.subspace.num_qubits = num_qubits
        self.subspace.size = <size_t>size
        if not use_all_bitset_blocks:
            self.subspace.bitstrings = BitsetHashMapWrapper(use_all_bitset_blocks)
        if reserve_multiplier < 1:
            raise ValueError(
                f"`reserve_multiplier(={reserve_multiplier})` must be >= 1"
            )
        # The +1 is here because insertion would fail for a dim=1 subspace otherwise
        self.subspace.bitstrings.reserve(self.subspace.size * reserve_multiplier + 1)

        cdef size_t idx
        cdef string key
        cdef bitset_t temp_bits

        for idx, key in enumerate(iterator):
            temp_bits = bitset_t(key, 0, self.subspace.num_qubits)
            self.subspace.bitstrings.insert_unique(temp_bits, idx)
    
    def __dealloc__(self):
        # Clear hash table upon deallocation of class
        self.subspace.bitstrings = BitsetHashMapWrapper()

    @cython.boundscheck(False)
    def __getitem__(self, object key):
        if key < 0:
            key = self.subspace.bitstrings.size() + key 
        cdef size_t idx = <size_t>key
        cdef bitset_t bits = self.subspace.bitstrings.get_n_th_bitset(idx)
        cdef Bitset out = Bitset()
        out.bits = bits
        return out

    def __len__(self):
        cdef size_t size = self.subspace.bitstrings.size()
        return size

    def __repr__(self):
        temp_str = f"size={self.subspace.size}, "
        temp_str += f"width={self.subspace.num_qubits}, "
        temp_str += f"num_buckets={self.subspace.bitstrings.num_buckets()}, "
        temp_str += f"use_all_bitset_blocks={self.subspace.bitstrings.use_all_bitset_blocks()}"
        return f"<Subspace: {temp_str}>"

    def size(self):
        """Size (dimensionality) of subspace

        Returns:
            int
        """
        cdef size_t size = self.subspace.bitstrings.size()
        return size

    def copy(self):
        """Return a copy of subspace

        Returns:
            Subspace
        """
        cdef Subspace out
        if self.size() > 1:
            out = Subspace([])
            out.subspace.bitstrings = self.subspace.bitstrings
            out.subspace.num_qubits = self.subspace.num_qubits
            out.subspace.size = self.subspace.size
        # Copying the data of a size=1 subspace segfaults. This gets around that by creating a new
        # subspace.  This is not a big deal since there is a single element
        else:
            out = Subspace([[self[0].to_string()]], use_all_bitset_blocks=self.subspace.bitstrings.use_all_bitset_blocks())
        return out

    # TODO: Move to sqd.pyx
    @cython.boundscheck(False)
    def get_orbital_occupancies(self, double[::1] probs, int norb):
        """Computes orbital occupancies of electrons.

        Args:
            probs (np.ndarray): Probabilities of each basis vector of a wavefunction.
                It must be the absolute value squared of an eigenvector, which represents
                amplitudes of each basis vector (Slated determinant) in a wavefunction.
            norb (int): The number of spatial orbitals.
        
        Returns:
            Orbital occupanies as a length-2 tuple of 1D np.ndarrays.
            It is formatted as ``([a0, ..., aN], [b0, ..., bN])``.
        """
        out = np.zeros(2 * norb, dtype=np.float64)

        cdef double[::1] out_mv = out
        compute_orbital_occupancies(
            self.subspace.bitstrings, self.subspace.size, &probs[0], &out_mv[0]
        )
        
        return np.split(out, 2)
    
    
    @cython.boundscheck(False)
    def get_bitstring_index(self, str bitstring):
        """Return the index of the given bitstring.

        Return value is max(size_t) if bitset not in subspace

        Args:
            bitstring (str): The bitstring as Python ``str`` type.
        
        Returns:
            size_t: Index if the bitstring is found.
            None: If the bitstring is not found.
        """
        cdef size_t index

        cdef bitset_t bitset
        bitset = bitset_t(bitstring, 0, len(bitstring))
        index = self.subspace.bitstrings.get(bitset)
        
        return None if index == MAX_SIZE_T else index


    @cython.boundscheck(False)
    def interpret_vector(self, double_or_complex[::1] vec, double atol=1e-12, int sort=0, int renormalize=True):
        """Convert solution vector into dict of counts and complex amplitudes

        Parameters:
            vec (ndarray): Complex solution vector
            atol (double): Absolute tolerance for truncation, default=1e-12
            sort (int): Sort output dict by integer representation.
            renormalize (bool): Renormalize values such that probabilities sum to one, default = True

        Returns:
            dict: Dictionary with bit-string keys and complex values

        Notes:
            Truncation can be disabled by calling `atol=0`
        """
        cdef size_t kk
        cdef double abs_val
        cdef double reduced_prob = 0
        cdef string s
        # cdef bytes py_bytes
        cdef dict out = {}

        for kk in range(self.subspace.bitstrings.size()):
            abs_val = abs(vec[kk])
            if abs_val <= atol:
                continue
            to_string(self.subspace.bitstrings.get_n_th_bitset(kk), s)

            out[s] = vec[kk]
            reduced_prob += abs_val * abs_val

        if renormalize:
            reduced_prob = math.sqrt(reduced_prob)
            for key in out:
                out[key] /= reduced_prob

        if sort:
            out = {k: v for k, v in sorted(out.items(), key=lambda item: int(item[0], 2))}
        return out
    
    
    def get_n_th_bitstring(self, size_t n):
        """Return n-th bitstring in the Subspace

        Parameters:
            n (size_t): Index of the expected bitstring.

        Returns:
            str: N-th bitstring in the subspace. Note that, both Python
                dictionary and emhash8::HashMap retains the insertion order.
        """
        cdef string s
        to_string(self.subspace.bitstrings.get_n_th_bitset(n), s)
        return s

    
    def get_bitset_index(self, Bitset bitset):
        """Return the index of the given Bitset.

        Return value is max(size_t) if bitset not in subspace

        Returns:
            size_t: Index
        """
        return self.subspace.bitstrings.get(bitset.bits)
    
    
    @cython.boundscheck(False)
    def to_dict(self, str key_type='bitset'):
        """Converts Subspace to a dictionary

        Parameters:
            key_type (str): Type of key data to return, default='bitset'

        Returns:
            dict
        """
        cdef size_t kk
        cdef string s
        cdef Bitset temp_bits
        cdef bitset_t bits
        cdef dict out = {}


        if key_type not in ['bitset', 'str']:
            raise FulqrumError("key_type must be 'bitset' or 'str'")
        
        if key_type == 'bitset':
            for kk in range(self.subspace.bitstrings.size()):
                bits = self.subspace.bitstrings.get_n_th_bitset(kk)
                temp_bits = Bitset()
                temp_bits.bits = bits
                out[temp_bits] = None
        
        elif key_type == 'str':
            for kk in range(self.subspace.bitstrings.size()):
                to_string(self.subspace.bitstrings.get_n_th_bitset(kk), s)
                out[s] = None
        return out
