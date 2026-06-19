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

"""PySCF conversion utilities"""

from pathlib import Path
import numpy as np
from ..core.fermi_operator import FermionicOperator


def _flat_index2d(i, j, dim):
    if i >= dim or j >= dim:
        raise Exception(f"Indices should be < {dim}")
    return i * dim + j


def _flat_index4d(i, j, k, l, dim):
    if i >= dim or j >= dim or k >= dim or l >= dim:
        raise Exception(f"Indices should be < {dim}")
    return i + j * dim + k * dim * dim + l * dim * dim * dim


def integrals_to_fq_fermionic_op(
    one_body_integrals, two_body_integrals, constant=0, EQ_TOLERANCE=1e-12
) -> FermionicOperator:
    """Convert one- and two-body integrals as numpy arrays into Fulqrum
        fermionic operator.

    Parameters:
        one_body_integrals (np.ndarray): One body integrals.
        two_body_integrals (np.ndarray): Two body integrals. The integrals must
            be in chemist's notation (a_p^ a_q a_r^ a_s) as used by PySCF.
        constant (float or complex): A constant term such as nuclear repulsion energy.
        EQ_TOLERANCE (float): Equality tolerance.

    Returns:
        FermionicOperator: Converted operator.
    """
    two_body_integrals = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order="C")
    # Go to flat arrays in prep for doing calculation in C++
    flat_one_body_integrals = one_body_integrals.ravel()
    flat_two_body_integrals = two_body_integrals.ravel()

    num_qubits = int(2 * np.sqrt(flat_one_body_integrals.shape[0]))
    half_num_qubits = num_qubits // 2

    ob_str = "+-"
    tb_str = "++--"

    qubit_mapping = np.zeros(num_qubits, dtype=np.uint32)
    for kk in range(num_qubits):
        qubit_mapping[kk] = (not kk % 2) * kk // 2 + (kk % 2) * (
            kk // 2 + half_num_qubits
        )

    fop = FermionicOperator(num_qubits)
    if np.abs(constant) > EQ_TOLERANCE:
        fop += FermionicOperator(num_qubits, [("", [], constant)])

    for p in range(half_num_qubits):
        for q in range(half_num_qubits):
            # temp_one_body = one_body_integrals[p, q]
            temp_one_body = flat_one_body_integrals[
                _flat_index2d(p, q, half_num_qubits)
            ]
            if np.abs(temp_one_body) > EQ_TOLERANCE:
                # Populate 1-body coefficients. Require p and q have same spin.
                ii = 2 * p
                jj = 2 * q
                fop += FermionicOperator(
                    num_qubits,
                    [(ob_str, [qubit_mapping[ii], qubit_mapping[jj]], temp_one_body)],
                )

                ii = 2 * p + 1
                jj = 2 * q + 1
                fop += FermionicOperator(
                    num_qubits,
                    [(ob_str, [qubit_mapping[ii], qubit_mapping[jj]], temp_one_body)],
                )
            # Continue looping to prepare 2-body coefficients.
            for r in range(half_num_qubits):
                for s in range(half_num_qubits):
                    temp_two_body = (
                        flat_two_body_integrals[
                            _flat_index4d(p, q, r, s, half_num_qubits)
                        ]
                        / 2.0
                    )
                    if np.abs(temp_two_body) > EQ_TOLERANCE:
                        # Mixed spin
                        ii = 2 * p
                        jj = 2 * q + 1
                        kk = 2 * r + 1
                        ll = 2 * s
                        fop += FermionicOperator(
                            num_qubits,
                            [
                                (
                                    tb_str,
                                    [
                                        qubit_mapping[ii],
                                        qubit_mapping[jj],
                                        qubit_mapping[kk],
                                        qubit_mapping[ll],
                                    ],
                                    temp_two_body,
                                )
                            ],
                        )

                        ii = 2 * p + 1
                        jj = 2 * q
                        kk = 2 * r
                        ll = 2 * s + 1
                        fop += FermionicOperator(
                            num_qubits,
                            [
                                (
                                    tb_str,
                                    [
                                        qubit_mapping[ii],
                                        qubit_mapping[jj],
                                        qubit_mapping[kk],
                                        qubit_mapping[ll],
                                    ],
                                    temp_two_body,
                                )
                            ],
                        )

                        # Same spin
                        ii = 2 * p
                        jj = 2 * q
                        kk = 2 * r
                        ll = 2 * s
                        fop += FermionicOperator(
                            num_qubits,
                            [
                                (
                                    tb_str,
                                    [
                                        qubit_mapping[ii],
                                        qubit_mapping[jj],
                                        qubit_mapping[kk],
                                        qubit_mapping[ll],
                                    ],
                                    temp_two_body,
                                )
                            ],
                        )

                        ii = 2 * p + 1
                        jj = 2 * q + 1
                        kk = 2 * r + 1
                        ll = 2 * s + 1
                        fop += FermionicOperator(
                            num_qubits,
                            [
                                (
                                    tb_str,
                                    [
                                        qubit_mapping[ii],
                                        qubit_mapping[jj],
                                        qubit_mapping[kk],
                                        qubit_mapping[ll],
                                    ],
                                    temp_two_body,
                                )
                            ],
                        )

    return fop


def fcidump_to_fq_fermionic_op(fcidump_path: str | Path) -> FermionicOperator:
    """Load one- and two-body integrals as numpy arrays into Fulqrum
        fermionic operator from FCIDUMP file.

    Parameters:
        fcidump_path (str | Path): The FCIDUMP file.

    Returns:
        FermionicOperator: Converted operator.
    """
    from pyscf import ao2mo, tools

    mf_as = tools.fcidump.to_scf(fcidump_path)
    hcore = mf_as.get_hcore()
    num_spatial_orbitals = hcore.shape[0]
    eri = ao2mo.restore(1, mf_as._eri, num_spatial_orbitals)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()

    return integrals_to_fq_fermionic_op(
        one_body_integrals=hcore,
        two_body_integrals=eri,
        constant=nuclear_repulsion_energy,
    )
