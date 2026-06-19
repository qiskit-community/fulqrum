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
# pylint: disable=no-name-in-module
import pytest

import numpy as np
from fulqrum.convert import (
    openfermion_fermi_op_to_fulqrum,
    openfermion_qubit_op_to_fulqrum,
    integrals_to_fq_fermionic_op,
)


def test_openfermion_qubit_op_to_fulqrum():
    "Test OpenFermion QubitOperator to Fulqrum QubitOperator conversion"
    openfermion = pytest.importorskip("openfermion")

    labels = ["Z5 Y3 X2", "Z0 X3", "", "Y2 X4 Z0"]
    coeffs = [0.5, -1.5, 1.0, 0.5 + 0.6j]

    qubit_reorder_map = {5: 5, 4: 2, 3: 4, 2: 1, 1: 3, 0: 0}
    openf_qubit_op = openfermion.QubitOperator()
    for label, coeff in zip(labels, coeffs):
        openf_qubit_op += coeff * openfermion.QubitOperator(label)

    fulqrum_qubit_op = openfermion_qubit_op_to_fulqrum(openf_qubit_op)

    for idx in range(fulqrum_qubit_op.num_terms):
        label = labels[idx].split()
        label = [(item[0], qubit_reorder_map[int(item[1:])]) for item in label]
        label = sorted(label, key=lambda x: x[1])
        assert fulqrum_qubit_op[idx].operators == label
        assert fulqrum_qubit_op[idx].coefficients()[0] == coeffs[idx]

    assert fulqrum_qubit_op.width == 6


def test_openfermion_qubit_op_to_fulqrum_value_error():
    "Test OpenFermion QubitOperator to Fulqrum QubitOperator conversion"
    openfermion = pytest.importorskip("openfermion")
    labels = ["Z4 Y3 X2", "Z0 X3", "", "Y2 X4 Z0"]
    coeffs = [0.5, -1.5, 1.0, 0.5 + 0.6j]

    openf_qubit_op = openfermion.QubitOperator()
    for label, coeff in zip(labels, coeffs):
        openf_qubit_op += coeff * openfermion.QubitOperator(label)

    with pytest.raises(ValueError) as msg:
        openfermion_qubit_op_to_fulqrum(openf_qubit_op)

    assert str(msg.value) == (
        "Number of qubits must be even in a QubitOperator. number of qubits: 5 is odd."
    )


def test_openfermion_fermi_op_to_fulqrum():
    openfermion = pytest.importorskip("openfermion")
    terms = ["", "4^ 3^ 9 1", "3 1^"]
    coeffs = [1, 1.0 + 2.0j, -1.7]
    qubit_order_map = {9: 9, 8: 4, 7: 8, 6: 3, 5: 7, 4: 2, 3: 6, 2: 1, 1: 5, 0: 0}
    openf_fermi_op = openfermion.FermionOperator()
    for term, coeff in zip(terms, coeffs):
        openf_fermi_op += openfermion.FermionOperator(term, coeff)

    fulqrum_fermi_op = openfermion_fermi_op_to_fulqrum(openf_fermi_op)

    for idx in range(fulqrum_fermi_op.num_terms):
        single_terms = []
        for elem in terms[idx].split():
            if "^" in elem:
                single_terms.append(("+", qubit_order_map[int(elem[:-1])]))
            else:
                single_terms.append(("-", qubit_order_map[int(elem)]))

        single_terms = sorted(single_terms, key=lambda x: x[1])
        assert (
            fulqrum_fermi_op[idx].coefficients()[0] == coeffs[idx]
            or fulqrum_fermi_op[idx].coefficients()[0] == -coeffs[idx]
        )
        assert fulqrum_fermi_op[idx].operators == single_terms


def test_openfermion_fermi_op_to_fulqrum_value_error():
    openfermion = pytest.importorskip("openfermion")
    terms = ["", "4^ 3^ 8 1", "3 1^"]
    coeffs = [1, 1.0 + 2.0j, -1.7]

    openf_fermi_op = openfermion.FermionOperator()
    for term, coeff in zip(terms, coeffs):
        openf_fermi_op += openfermion.FermionOperator(term, coeff)

    with pytest.raises(ValueError) as msg:
        openfermion_fermi_op_to_fulqrum(openf_fermi_op)

    assert str(msg.value) == (
        "Fermionic Operators with odd number of modes are not supported yet."
    )


def test_integrals_to_fq_fermionic_op():
    """Direct conversion from PySCF integrals to fermi ops matches OpenFermion"""
    openfermion = pytest.importorskip("openfermion")
    pyscf = pytest.importorskip("pyscf")
    import pyscf.mcscf as mc

    def old_integrals_to_fq_fermionic_op(
        one_body_integrals, two_body_integrals, constant=0, EQ_TOLERANCE=1e-12
    ):
        """This is the old pathway"""
        two_body_integrals = np.asarray(
            two_body_integrals.transpose(0, 2, 3, 1), order="C"
        )
        of_interaction_op = openfermion.hamiltonians.generate_hamiltonian(
            one_body_integrals=one_body_integrals,
            two_body_integrals=two_body_integrals,
            constant=constant,
            EQ_TOLERANCE=EQ_TOLERANCE,
        )
        of_fermion_op = openfermion.transforms.get_fermion_operator(of_interaction_op)
        return openfermion_fermi_op_to_fulqrum(of_fermion_op)

    atoms = [
        [["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        [["Li", (0, 0, 0)], ["H", (1.0, 0, 0)]],
        [["H", (0, 0, 0)], ["H", (1.0, 0, 0)]],
    ]
    bases = ["sto3g", "sto6g"]

    for atom in atoms:
        for basis in bases:
            mol = pyscf.gto.Mole()
            mol.build(atom=atom, basis=basis)
            n_frozen = 0
            active_space = range(n_frozen, mol.nao_nr())
            # Get molecular integrals
            scf = pyscf.scf.RHF(mol).run()
            num_orbitals = len(active_space)
            n_electrons = int(sum(scf.mo_occ[active_space]))
            num_elec_a = (n_electrons + mol.spin) // 2
            num_elec_b = (n_electrons - mol.spin) // 2
            cas = mc.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
            mo = cas.sort_mo(active_space, base=0)
            hcore, nuclear_repulsion_energy = cas.get_h1cas(
                mo
            )  # hcore: one-body integrals
            eri = pyscf.ao2mo.restore(
                1, cas.get_h2cas(mo), num_orbitals
            )  # eri: two-body integrals
            # Build operators
            # The way truncation works in the old path is not exactly the same as the new path
            # The old path seems to have some addtional truncation somewhere.  As such we set the
            # tolerance value so that these extra small get cut and we do not throw an error
            fop = integrals_to_fq_fermionic_op(
                one_body_integrals=hcore, two_body_integrals=eri, EQ_TOLERANCE=1e-10
            )
            fop2 = old_integrals_to_fq_fermionic_op(
                one_body_integrals=hcore, two_body_integrals=eri
            )
            # A brute force way to check for operator equality
            assert fop.size() == fop2.size()
            num_touched = 0
            for op in fop:
                found = False
                for op2 in fop2:
                    if (
                        abs(op.coeff - op2.coeff) < 1e-12
                        and op.operators == op2.operators
                    ):
                        found = True
                        num_touched += 1
                        break
                assert found
            assert num_touched == fop.size()


@pytest.mark.skip(reason="Not implemented as it uses already tested functions")
def test_fcidump_to_fq_fermionic_op():
    """The function uses PySCF to extract one- and two-body integrals and then
    calls  ``integrals_to_fq_fermionic_op()``. Therefore, we are skipping
    explicitly testing this function for now.

    Test Idea: Search fcidump for a small molecule such as
        H2O, solve it for full subspace, and compare expected accuracy with
        computed accuracy. If integral conversion is correct, then accuracies
        must match.
    """
    pass
