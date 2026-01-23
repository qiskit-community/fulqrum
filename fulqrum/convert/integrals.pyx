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
cimport cython

from pathlib import Path
import numpy as np

from ..core.fermi_operator cimport FermionicOperator
from .openfermion import openfermion_fermi_op_to_fulqrum


@cython.boundscheck(False)
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
    import openfermion

    # PySCF generated two-body integrals use chemist notation
    # a_p^ a_q a_r^ a_s whereas openfermion use physicist
    # notation a_p^ a_q^ a_r a_s. Thus, we need to convert ordering.
    # See: https://github.com/quantumlib/OpenFermion-PySCF/blob/e669cec3cc3f0b11170c8293249197d71bede82c/openfermionpyscf/_run_pyscf.py#L93-L94
    two_body_integrals = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order='C')
    of_interaction_op = openfermion.hamiltonians.generate_hamiltonian(
        one_body_integrals=one_body_integrals,
        two_body_integrals=two_body_integrals,
        constant=constant,
        EQ_TOLERANCE=EQ_TOLERANCE
    )
    of_fermion_op = openfermion.transforms.get_fermion_operator(of_interaction_op)

    return openfermion_fermi_op_to_fulqrum(of_fermion_op)


@cython.boundscheck(False)
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
    # TODO: Ask Kevin Sung or Mario Motta about the correct/need for below line
    eri = ao2mo.restore(1, mf_as._eri, num_spatial_orbitals)
    nuclear_repulsion_energy = mf_as.mol.energy_nuc()

    return integrals_to_fq_fermionic_op(
        one_body_integrals=hcore,
        two_body_integrals=eri,
        constant=nuclear_repulsion_energy
    )
