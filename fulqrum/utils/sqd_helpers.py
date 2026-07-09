# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helpers for the sample-based quantum diagonalization (SQD) loop."""

from collections import OrderedDict


def split_alpha_beta(full_bitstrings):
    """Split full bitstrings into ordered, de-duplicated alpha and beta halves.

    Parameters:
        full_bitstrings (list[str]): Full bitstrings, each of even length ``2 * N``.

    Returns:
        tuple[OrderedDict, OrderedDict]: ``(alpha_halfs, beta_halfs)`` as
        order-preserving sets (dict keys). Insertion order is preserved so a
        caller can keep a stable precedence across iterations. Returns empty
        dicts if ``full_bitstrings`` is empty.
    """
    alpha = OrderedDict()
    beta = OrderedDict()
    if not full_bitstrings:
        return alpha, beta

    n = len(full_bitstrings[0]) // 2
    for bs in full_bitstrings:
        beta[bs[:n]] = 1  # high/left bits
        alpha[bs[n:]] = 1  # low/right bits
    return alpha, beta


def build_spin_separated_half_strings(
    carryover_full_strs,
    batch,
    num_orbitals,
    num_a,
    num_b,
    pool_by_weight=True,
):
    """Build alpha and beta half-string sets for one SQD subspace construction.

    Parameters:
        carryover_full_strs (list[str]): Full carryover bitstrings from the
            previous round.
        batch (list[str]): Full bitstrings recovered/subsampled in this batch.
        num_orbitals (int): Number of spatial orbitals ``N`` (half the qubit
            count).
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        pool_by_weight (bool): If ``True`` (default) offer each observed half to
            whichever spin its Hamming weight matches. If ``False`` route halves
            strictly by the side they appeared on.

    Returns:
        tuple[OrderedDict, OrderedDict]: ``(alpha_halfs, beta_halfs)`` as
        order-preserving sets, Hartree-Fock first.
    """
    alpha_dict = OrderedDict()
    beta_dict = OrderedDict()

    # 1. Hartree-Fock half-strings (built separately per spin).
    alpha_dict["0" * (num_orbitals - num_a) + "1" * num_a] = 1
    beta_dict["0" * (num_orbitals - num_b) + "1" * num_b] = 1

    def _route(full_strs):
        n = num_orbitals
        for bs in full_strs:
            alpha_half = bs[n:]  # low/right bits
            beta_half = bs[:n]  # high/left bits
            if pool_by_weight:
                # Offer each half to whichever spin its Hamming weight matches.
                for half in (alpha_half, beta_half):
                    if half.count("1") == num_a:
                        alpha_dict[half] = 1
                    if half.count("1") == num_b:
                        beta_dict[half] = 1
            else:
                alpha_dict[alpha_half] = 1
                beta_dict[beta_half] = 1

    # 2. Carryover halves, then 3. recovered halves from this batch.
    _route(carryover_full_strs)
    _route(batch)

    return alpha_dict, beta_dict
