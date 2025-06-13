# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""
import numpy as np
from pathlib import Path
from qiskit.transpiler import CouplingMap
from fulqrum import QubitOperator, FermionicOperator


def test_grouping1():
    """Test grouping"""
    H = QubitOperator(2, [])
    for item in ["XY", "XI", "IY", "YY", "IZ", "II", "Z0"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    # three diag terms, two terms of weight two, and one of each others
    ans = np.array([0, 0, 0, 1, 2, 3, 3], dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping2():
    """Test grouping all diag terms gives all zero groups"""
    H = QubitOperator(3, [])
    for item in ["III", "ZZ1", "Z0Z", "IZI", "ZI0"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    ans = np.zeros(5, dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping3():
    """Test all operators have same offdiag structure"""
    H = QubitOperator(4, [])
    for item in ["XIIY", "+ZZ-", "Y01X", "-00+"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    ans = np.ones(4, dtype=np.int32)
    assert np.allclose(ans, H.groups())


def test_grouping_split():
    """Test sorting is passes onto operators that are split into diag/offdiag"""
    H = QubitOperator(4, [])
    for item in ["XIII", "ZYII", "ZIZI", "01+Z", "IIII", "+Z00"]:
        H += QubitOperator.from_label(item)
    H.offdiag_term_grouping()
    diag, offdiag = H.split_diagonal()
    diag_ans = np.zeros(len(diag))
    offdiag_ans = np.array([1, 1, 2, 3])
    assert np.allclose(diag.groups(), diag_ans)
    assert np.allclose(offdiag.groups(), offdiag_ans)


def test_empty_operator_pointers():
    """Test grouping and pointers do not fail for empty operator"""
    H = QubitOperator(5)
    assert H.num_groups == 0
    assert np.allclose(H.group_ptrs(), np.zeros(0))


def test_basic_group_pointers():
    """Test simple term group pointers"""
    H = QubitOperator(5)
    H += QubitOperator.from_label("IIIIX")
    H += QubitOperator.from_label("IIIXI")
    H += QubitOperator.from_label("IIYII")
    H += QubitOperator.from_label("I-III")
    H += QubitOperator.from_label("+IIII")
    assert H.num_groups == 5
    assert np.allclose(H.group_ptrs(), np.arange(5 + 1))


def test_basic_group_pointers2():
    """Test repeat elements in front"""
    H = QubitOperator(5)
    H += QubitOperator.from_label("IIIIX")
    H += QubitOperator.from_label("IIIIY")
    H += QubitOperator.from_label("IIIXI")
    H += QubitOperator.from_label("IIYII")
    H += QubitOperator.from_label("I-III")
    H += QubitOperator.from_label("+IIII")
    assert H.num_groups == 5
    # repeat elements in front
    assert np.allclose(H.group_ptrs(), np.array([0, 2, 3, 4, 5, 6]))


def test_basic_group_pointers3():
    """Test repeat elements in back"""
    H = QubitOperator(5)
    H += QubitOperator.from_label("IIIIX")
    H += QubitOperator.from_label("IIIXI")
    H += QubitOperator.from_label("IIYII")
    H += QubitOperator.from_label("I-III")
    H += QubitOperator.from_label("+IIII")
    H += QubitOperator.from_label("XIIII")
    assert H.num_groups == 5
    # repeat elements in front
    assert np.allclose(H.group_ptrs(), np.array([0, 1, 2, 3, 4, 6]))


def test_basic_group_pointers4():
    """Test repeat terms with id term"""
    H = QubitOperator(5)
    H += QubitOperator.from_label("IIIIX")
    H += QubitOperator.from_label("+IIII")
    H += QubitOperator.from_label("XIIII")
    H += QubitOperator.from_label("YIIII")
    H += QubitOperator.from_label("IIIII")
    assert H.num_groups == 3
    # Id term gets moved to front since group is 0 by definition
    # next is IIIIX term, then last 3 terms combined
    assert np.allclose(H.group_ptrs(), np.array([0, 1, 2, 5]))


def test_square_group_pointers():
    """Build 1600-qubit square coupling map and validate grouping pointers"""
    cmap = CouplingMap.from_grid(40, 40)
    num_qubits = cmap.size()

    # Generate Hamiltonian
    H = QubitOperator(num_qubits, [])
    touched_edges = set({})
    coeffs = [1 / 2, 1 / 2, 1]
    for edge in cmap.get_edges():
        if edge[::-1] not in touched_edges:
            H += QubitOperator(
                num_qubits,
                [
                    ("XX", edge, coeffs[0]),
                    ("YY", edge, coeffs[1]),
                    ("ZZ", edge, coeffs[2]),
                ],
            )
            touched_edges.add(edge)

    h_ptrs = H.group_ptrs()
    H_diag, H_off = H.split_diagonal()
    assert H_diag.num_terms == h_ptrs[1]
    hdiag_ptrs = H_diag.group_ptrs()
    assert hdiag_ptrs.shape[0] == 2
    assert hdiag_ptrs[1] == h_ptrs[1]
    assert np.allclose(H_diag.groups(), np.zeros(H_diag.num_terms))
    hoff_ptrs = H_off.group_ptrs()
    # XX and YY have same off-diagonal structure so this is True here
    assert H_off.num_terms // 2 == (hoff_ptrs.shape[0] - 1)
    assert np.allclose(np.diff(hoff_ptrs), 2 * np.ones(H_off.num_terms // 2))


def test_square_group_pointers_h2_example():
    """Test Fermionic grouping and pointers example using H2"""
    path = Path(__file__).parent / "data/h2.json"
    fop = FermionicOperator.from_json(path)
    op = fop.extended_jw_transformation()
    op_ptrs = op.group_ptrs()
    diag, off = op.split_diagonal()
    assert diag.num_terms == op_ptrs[1]
    assert off.num_terms == op_ptrs[2] - op_ptrs[1]
    # All the elements in the off-diagonal component have the same diagonal structure
    assert np.allclose(off.group_ptrs(), np.array([0, 4]))
