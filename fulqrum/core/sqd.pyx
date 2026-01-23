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
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t
cimport cython
import numpy as np
cimport numpy as np
from .bitset cimport bitset_t, to_string
cimport qiskit_addon_fulqrum.core.sqd as sqd
from .subspace cimport Subspace

include "includes/base_header.pxi"
include "includes/bitset_utils_header.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def postselect_by_hamming_right_and_left(
    list bitstrings,
    double[::1] weights,
    uint32_t right,
    uint32_t left
) -> tuple[list[str], np.ndarray]:
    """Postselect bitstrings based on desired Hamming weight on right and left halves.

    Args:
        bitstrings (list[str]): List of bitstrings as Python strs.
        weights (np.ndarray): Probability of each bitstring.
        right (uint32_t): Right hamming weight.
        left (uint32_t): Left hamming weight.
    Returns:
        Postselected bitstrings as Python list of strs and probabilities and np.ndarray.
        The new list contains only those bitstrings from the
        original matrix that have the desired Hamming weight on the right
        and left halves, and the new probabilities are constructed by taking
        the original probabilities corresponding to the postselected bitstrings
        and rescaling them to sum to one.
    """
    num_items = len(bitstrings)
    num_qubits = len(bitstrings[0])
    cdef bitset_t temp_bits
    cdef vector[double] wvec, new_wvec
    wvec.reserve(num_items)
    cdef vector[bitset_t] bitvec, new_bitvec
    bitvec.reserve(num_items)

    for bitstring in bitstrings:
        temp_bits = bitset_t(bitstring, 0, num_qubits)
        bitvec.push_back(temp_bits)

    for weight in weights:
        wvec.push_back(weight)

    bitvec_wvec_pair = sqd.postselect_bitstrings_cpp(bitvec, wvec, right, left)
    new_bitvec = bitvec_wvec_pair.first
    new_wvec = bitvec_wvec_pair.second

    cdef list py_bitvec = []
    cdef list py_weights = []
    cdef string s
    cdef double w

    cdef size_t n
    for n in range(new_bitvec.size()):
        w = new_wvec[n]
        if np.isclose(w, 0):
            continue
        to_string(new_bitvec[n], s)
        py_bitvec.append(s)
        py_weights.append(w)

    return py_bitvec, np.asarray(py_weights)


@cython.boundscheck(False)
@cython.wraparound(False)
def subsample(
    list bitstrings, double[::1] weights, int samples_per_batch, uint32_t seed=0
) -> list[str]:
    """Subsamples bitstrings from a list to create a batch.

    A batch will be sampled without replacement from the input ``bitstrings``.

    Args:
        bitstrings (list): Bitstrings as a Python list of strs.
        weights (np.ndarray): A 1D array specifying a probability distribution
            over the bitstrings.
        samples_per_batch (int): The number of samples to draw for each batch.
            Must be > 0.
        seed (uint32_t): A seed to control random behavior.

    Returns:
        A list of bitstrings subsampled from the input bitstrings.
    """
    if samples_per_batch <= 0:
        raise ValueError(f"samples_per_batch (= {samples_per_batch}) must > 0")
    num_qubits = len(bitstrings[0])
    cdef bitset_t temp_bits
    cdef vector[bitset_t] bitvec, subsampled_bitvec
    bitvec.reserve(len(bitstrings))

    cdef vector[double] wvec
    wvec.reserve(len(bitstrings))

    for bitstring in bitstrings:
        temp_bits = bitset_t(bitstring, 0, num_qubits)
        bitvec.push_back(temp_bits)

    for weight in weights:
        wvec.push_back(weight)

    subsampled_bitvec = sqd.subsample_cpp(bitvec, wvec, samples_per_batch, seed)

    cdef list py_bitvec = []
    cdef string s

    cdef int n
    for n in range(samples_per_batch):
        to_string(subsampled_bitvec[n], s)
        py_bitvec.append(s)

    return py_bitvec


@cython.boundscheck(False)
@cython.wraparound(False)
def recover_configurations(
    list bitstrings,
    double[::1] weights,
    double[::1] avg_occupancies_a,
    double[::1] avg_occupancies_b,
    int num_elec_a,
    int num_elec_b,
    uint32_t seed=0
) -> tuple[list[str], np.ndarray]:
    """Refine bitstrings based on average orbital occupancy and target hamming weights.

    This function refines each bit in isolation in an attempt to transform the Hilbert
    space represented by the input ``bitstrings`` into a space closer to that which
    supports the ground state.

    .. note::

        This function makes the assumption that bit ``i`` represents the
        spin-down orbital corresponding to the spin-up orbital in bit ``i + N``
        where ``N`` is the number of spatial orbitals and ``i < N``.

    Args:
        bitstrings (list[str]): Bitstrings as a Python list of strs.
        weights (np.ndarray): A 1D array specifying a probability distribution
            over the bitstrings.
        avg_occupancies_a (np.ndarray): An 1D array holding the mean occupancy
            of the spin-up (alpha) orbitals. The occupancies should be formatted
            as: ``array([occ_a_0, ..., occ_a_N]``.
        avg_occupancies_b (np.ndarray): An 1D array holding the mean occupancy
            of the spin-down (beta) orbitals. and spin-down orbitals, respectively.
            The occupancies should be formatted as: ``array([occ_b_0, ..., occ_b_N]``.
        num_elec_a (int): The number of spin-up (alpha) electrons in the system.
        num_elec_b (int): The number of spin-down (beta) electrons in the system.
        seed (uint32_t): A seed for controlling randomness

    Returns:
        A list bitstrings and an updated probability array.

    References:
        [1]: J. Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_,
             arXiv:2405.05068 [quant-ph].
    """
    cdef int num_qubits
    num_qubits = len(bitstrings[0])
    cdef bitset_t temp_bits
    cdef vector[bitset_t] bitvec, new_bitvec
    bitvec.reserve(len(bitstrings))

    cdef vector[double] wvec, new_wvec, occ_a, occ_b
    wvec.reserve(len(bitstrings))
    occ_a.reserve(num_qubits // 2)
    occ_b.reserve(num_qubits // 2)

    for bitstring in bitstrings:
        temp_bits = bitset_t(bitstring, 0, num_qubits)
        bitvec.push_back(temp_bits)

    for weight in weights:
        wvec.push_back(weight)
    cdef size_t n
    for k in range(num_qubits//2):
        occ_a.push_back(avg_occupancies_a[k])
        occ_b.push_back(avg_occupancies_b[k])

    bitvec_wvec_pair = sqd.recover_configurations_cpp(
        bitvec,
        wvec,
        occ_a,
        occ_b,
        num_elec_a,
        num_elec_b,
        seed
    )

    new_bitvec = bitvec_wvec_pair.first
    new_wvec = bitvec_wvec_pair.second

    cdef list py_bitstrings = []
    cdef list py_weights = []
    cdef string s
    cdef double w

    for n in range(new_bitvec.size()):
        w = new_wvec[n]
        if np.isclose(w, 0):
            continue
        to_string(new_bitvec[n], s)
        py_bitstrings.append(s)
        py_weights.append(w)

    return py_bitstrings, np.asarray(py_weights)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_carryover_full_strs(
    Subspace subspace, double[::1] abs_amps, double threshold
) -> list[tuple[str, float]]:
    """Gets full bitstrings from the subspace above specified threshold.

    These bitstrings can be carried over to next iteration of configuration
    recovery loop.

    Args:
        subsapce (Subspace): Subspace containing the bitstrings.
        
        abs_amps (ndarray): Absolute amplitudes of subspace bitstrings.
            One should use ``np.abs(eigenvector)`` as this argument as
            it will maintain order between subspace bitstrings and the
            absolute amplitudes.
        
        threshold (float): Threshold to filter out bitstrings. Only bitstrings
            with absolute amplitude ``> threshold`` is returned.
    
    Returns:
        A sorted list of lenght-2 tuples where the first element is the carryover
        bitstring and the second element is the absolute amplitude. The list is sorted
        is descending order of absolute amplitudes.
    """
    cdef size_t kk
    cdef string s
    cdef list out = []

    for kk in range(<size_t>subspace.size()):
        if abs_amps[kk] <= threshold:
            continue
        s = subspace.get_n_th_bitstring(kk)
        out.append((s, abs_amps[kk]))
    
    out.sort(key=lambda x: x[1], reverse=True)
    return out
