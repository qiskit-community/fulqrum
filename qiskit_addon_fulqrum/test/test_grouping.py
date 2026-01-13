# This code is a Qiskit project.
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
"""Test basic core functionality"""
import numpy as np
from pathlib import Path
from qiskit.transpiler import CouplingMap
import qiskit_addon_fulqrum as fq
from qiskit_addon_fulqrum import QubitOperator, FermionicOperator


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
    offdiag_ans = np.array([1, 2, 3, 3])
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


def test_basic_group_pointers5():
    """Test combination of terms for grouping"""
    op = fq.QubitOperator.from_label("IIII")
    op += fq.QubitOperator.from_label("XIIX")
    op += fq.QubitOperator.from_label("YIIX")
    op += fq.QubitOperator.from_label("IYYI")
    op += fq.QubitOperator.from_label("ZIII")
    op += fq.QubitOperator.from_label("XYYX")
    op += fq.QubitOperator.from_label("IIIX")
    assert op.num_groups == 5
    op.offdiag_term_grouping()
    assert np.allclose(op.group_ptrs(), np.array([0, 2, 3, 5, 6, 7]))


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


def test_group_ladder_indices1():
    """Validate that group ladder indices routine works as it should for ladders"""
    op = fq.QubitOperator.from_label("I+I+")
    op += fq.QubitOperator.from_label("+II+")
    op += fq.QubitOperator.from_label("+III")
    op += fq.QubitOperator.from_label("-III")
    op += fq.QubitOperator.from_label("I++I")
    op += fq.QubitOperator.from_label("I--I")
    op += fq.QubitOperator.from_label("----")
    op += fq.QubitOperator.from_label("--I-")
    op.set_type(2)
    op.group_term_sort_by_ladder_int()
    inds_list = op.group_offdiag_indices()
    assert np.allclose(inds_list[0], np.array([0, 2], dtype=np.uint32))
    assert np.allclose(inds_list[1], np.array([3], dtype=np.uint32))
    assert np.allclose(inds_list[2], np.array([0, 3], dtype=np.uint32))
    assert np.allclose(inds_list[3], np.array([1, 2], dtype=np.uint32))
    assert np.allclose(inds_list[4], np.array([0, 2, 3], dtype=np.uint32))
    assert np.allclose(inds_list[5], np.array([0, 1, 2, 3], dtype=np.uint32))


def test_group_ladder_indices2():
    """Validate that group ladder indices routine works as it should for Paulis"""
    op = fq.QubitOperator.from_label("IXIY")
    op += fq.QubitOperator.from_label("XIIX")
    op += fq.QubitOperator.from_label("YIII")
    op += fq.QubitOperator.from_label("XIII")
    op += fq.QubitOperator.from_label("IXXI")
    op += fq.QubitOperator.from_label("IYYI")
    op += fq.QubitOperator.from_label("YYYY")
    op += fq.QubitOperator.from_label("YYIY")
    op.offdiag_term_grouping()
    inds_list = op.group_offdiag_indices()
    assert np.allclose(inds_list[0], np.array([0, 2], dtype=np.uint32))
    assert np.allclose(inds_list[1], np.array([3], dtype=np.uint32))
    assert np.allclose(inds_list[2], np.array([0, 3], dtype=np.uint32))
    assert np.allclose(inds_list[3], np.array([1, 2], dtype=np.uint32))
    assert np.allclose(inds_list[4], np.array([0, 2, 3], dtype=np.uint32))
    assert np.allclose(inds_list[5], np.array([0, 1, 2, 3], dtype=np.uint32))


def test_group_terms_ladder_int1():
    """Test that sorting groups by ladder ints does what it should"""
    op = fq.QubitOperator.from_label("III+")
    op += fq.QubitOperator.from_label("-II+")
    op += fq.QubitOperator.from_label("+II-")
    op += fq.QubitOperator.from_label("+II+")
    op += fq.QubitOperator.from_label("++-I")
    op += fq.QubitOperator.from_label("---I")
    op += fq.QubitOperator.from_label("IZZI")
    op.set_type(2)
    op.group_term_sort_by_ladder_int()
    assert np.allclose(op.group_ptrs(), [0, 1, 2, 5, 7])
    assert np.allclose(
        op.ladder_ints(),
        np.array([np.iinfo(np.uint32).max, 1, 1, 2, 3, 0, 6], dtype=np.uint32),
    )


def test_group_terms_ladder_int_width1():
    """Test that sorting groups by ladder ints obeys ladder_width"""
    op = fq.QubitOperator.from_label("++++++")
    op += fq.QubitOperator.from_label("------")
    op += fq.QubitOperator.from_label("IIIZZI")
    op.set_type(2)
    op.group_term_sort_by_ladder_int(3)
    assert np.allclose(
        op.ladder_ints(), np.array([np.iinfo(np.uint32).max, 0, 7], dtype=np.uint32)
    )


def test_group_terms_ladder_int_width2():
    """Test that sorting groups by ladder ints obeys ladder_width"""
    op = fq.QubitOperator.from_label("++++++")
    op += fq.QubitOperator.from_label("------")
    op += fq.QubitOperator.from_label("IIIZZI")
    op.set_type(2)
    op.group_term_sort_by_ladder_int(2)
    assert np.allclose(
        op.ladder_ints(), np.array([np.iinfo(np.uint32).max, 0, 3], dtype=np.uint32)
    )


def test_group_terms_ladder_int_width3():
    """Test that sorting groups by ladder ints obeys ladder_width"""
    op = fq.QubitOperator.from_label("++++++")
    op += fq.QubitOperator.from_label("------")
    op += fq.QubitOperator.from_label("IIIZZI")
    op.set_type(2)
    op.group_term_sort_by_ladder_int(1)
    assert np.allclose(
        op.ladder_ints(), np.array([np.iinfo(np.uint32).max, 0, 1], dtype=np.uint32)
    )


def test_group_terms_ladder_int_width4():
    """Test that sorting groups by ladder ints obeys ladder_width"""
    op = fq.QubitOperator.from_label("_+II")
    op += fq.QubitOperator.from_label("++II")
    op += fq.QubitOperator.from_label("IIIZ")
    op.set_type(2)
    op.group_term_sort_by_ladder_int(3)
    assert np.allclose(
        op.ladder_ints(), np.array([np.iinfo(np.uint32).max, 1, 3], dtype=np.uint32)
    )


def test_group_ladder_bin_starts1():
    """Verify that ladder bin starts show correct locations of elements based on int values"""
    # group 2
    op = fq.QubitOperator.from_label("-II+")  # int = 1
    op += fq.QubitOperator.from_label("+II-")  # int = 2
    op += fq.QubitOperator.from_label("+II+")  # int = 3
    # group 1
    op += fq.QubitOperator.from_label("II+-")  # int = 2
    op += fq.QubitOperator.from_label("II-+")  # int = 1
    # group 3
    op += fq.QubitOperator.from_label("I++-")  # int = 6
    op += fq.QubitOperator.from_label("I---")  # int = 0
    op += fq.QubitOperator.from_label("I+++")  # int = 7
    op.set_type(2)
    op.offdiag_term_grouping()
    op.group_term_sort_by_ladder_int()

    assert np.allclose(op.terms_by_group(1).ladder_ints(), [1, 2])
    assert np.allclose(op.terms_by_group(2).ladder_ints(), [1, 2, 3])
    assert np.allclose(op.terms_by_group(3).ladder_ints(), [0, 6, 7])


def test_group_ladder_bin_starts2():
    """Verify that ladder bin starts correct for ladder_width=1"""
    op = fq.QubitOperator.from_label("-II+")  # int = 1
    op += fq.QubitOperator.from_label("-ZZ+")  # int = 1
    op += fq.QubitOperator.from_label("+II-")  # int = 0
    op += fq.QubitOperator.from_label("-ZZ-")  # int = 0
    op += fq.QubitOperator.from_label("+0I-")  # int = 0
    op.set_type(2)
    op.group_term_sort_by_ladder_int(1)
    group_ladder_starts = op.group_ladder_bin_starts()
    assert np.allclose(group_ladder_starts, [0, 3, 5])


def test_group_ladder_bin_starts3():
    """Verify that ladder bin starts yield correct numbers for 2 groups"""
    # group 2
    op = fq.QubitOperator.from_label("-II+")  # int = 1,
    op += fq.QubitOperator.from_label("-ZZ+")  # int = 1,
    op += fq.QubitOperator.from_label("+II-")  # int = 2,
    op += fq.QubitOperator.from_label("-ZZ-")  # int = 0,
    op += fq.QubitOperator.from_label("+0I-")  # int = 2,
    op += fq.QubitOperator.from_label("+01+")  # int = 3,
    # group 1
    op += fq.QubitOperator.from_label("II-+")  # int = 1
    op += fq.QubitOperator.from_label("II++")  # int = 3
    op += fq.QubitOperator.from_label("ZZ--")  # int = 0
    op += fq.QubitOperator.from_label("ZZ+-")  # int = 2
    op += fq.QubitOperator.from_label("Z0+-")  # int = 2
    op += fq.QubitOperator.from_label("Z0-+")  # int = 1
    op += fq.QubitOperator.from_label("ZI++")  # int = 3
    op.set_type(2)
    op.offdiag_term_grouping()
    op.group_term_sort_by_ladder_int()
    assert np.allclose(op.terms_by_group(1).ladder_ints(), [0, 1, 1, 2, 2, 3, 3])
    assert np.allclose(op.terms_by_group(2).ladder_ints(), [0, 1, 1, 2, 2, 3])


def test_group_ladder_bin_starts4():
    """bin starts yield correct numbers for 2 groups that switch order"""
    # group 1
    op = fq.QubitOperator.from_label("-II+")  # int = 1,
    op += fq.QubitOperator.from_label("-ZZ+")  # int = 1,
    op += fq.QubitOperator.from_label("+II-")  # int = 2
    op += fq.QubitOperator.from_label("-ZZ-")  # int = 0,
    op += fq.QubitOperator.from_label("+0I-")  # int = 2,
    op += fq.QubitOperator.from_label("+01+")  # int = 3,
    # group 0
    op += fq.QubitOperator.from_label("IIZ+")  # int = 1
    op += fq.QubitOperator.from_label("II0+")  # int = 1
    op += fq.QubitOperator.from_label("ZZI-")  # int = 0
    op += fq.QubitOperator.from_label("ZZ1-")  # int = 0
    op += fq.QubitOperator.from_label("Z01-")  # int = 0
    op += fq.QubitOperator.from_label("Z00+")  # int = 1
    op += fq.QubitOperator.from_label("ZII+")  # int = 1
    op.set_type(2)
    op.group_term_sort_by_ladder_int(3)
    group_ladder_starts = op.group_ladder_bin_starts()
    num_terms = np.diff(group_ladder_starts)

    grp0_num_terms = num_terms[:8]
    grp1_num_terms = num_terms[8:]
    assert np.allclose(grp0_num_terms, [3, 4, 0, 0, 0, 0, 0, 0])
    assert np.allclose(grp1_num_terms, [1, 2, 2, 1, 0, 0, 0, 0])


def test_group_ladder_bin_starts5():
    """Validate bin starts index correct term"""
    # group 1
    op = fq.QubitOperator.from_label("-II+")  # int = 1,
    op += fq.QubitOperator.from_label("-ZZ+")  # int = 1,
    op += fq.QubitOperator.from_label("+II-")  # int = 2
    op += fq.QubitOperator.from_label("-ZZ-")  # int = 0,
    op += fq.QubitOperator.from_label("+0I-")  # int = 2,
    op += fq.QubitOperator.from_label("+01+")  # int = 3,
    # group 0
    op += fq.QubitOperator.from_label("IIZ+")  # int = 1
    op += fq.QubitOperator.from_label("II0+")  # int = 1
    op += fq.QubitOperator.from_label("ZZI-")  # int = 0
    op += fq.QubitOperator.from_label("ZZ1-")  # int = 0
    op += fq.QubitOperator.from_label("Z01-")  # int = 0
    op += fq.QubitOperator.from_label("Z00+")  # int = 1
    op += fq.QubitOperator.from_label("ZII+")  # int = 1
    op.set_type(2)
    op.group_term_sort_by_ladder_int(3)
    group_ladder_starts = op.group_ladder_bin_starts()
    ptr_size = 2**3  #  2**ladder_width
    idx = 1 * ptr_size + 3  # group 1 + int value = 3
    # should find "+01+"
    assert op[group_ladder_starts[idx]].operators == [
        ("+", 0),
        ("1", 1),
        ("0", 2),
        ("+", 3),
    ]


def test_group_ladder_bin_starts6():
    """Validate bin starts pick out correct collection of terms by int"""
    # group 1
    op = fq.QubitOperator.from_label("-II+")  # int = 1,
    op += fq.QubitOperator.from_label("-ZZ+")  # int = 1,
    op += fq.QubitOperator.from_label("+II-")  # int = 2
    op += fq.QubitOperator.from_label("-ZZ-")  # int = 0,
    op += fq.QubitOperator.from_label("+0I-")  # int = 2,
    op += fq.QubitOperator.from_label("+01+")  # int = 3,
    # group 0
    op += fq.QubitOperator.from_label("IIZ+")  # int = 1
    op += fq.QubitOperator.from_label("II0+")  # int = 1
    op += fq.QubitOperator.from_label("ZZI-")  # int = 0
    op += fq.QubitOperator.from_label("ZZ1-")  # int = 0
    op += fq.QubitOperator.from_label("Z01-")  # int = 0
    op += fq.QubitOperator.from_label("Z00+")  # int = 1
    op += fq.QubitOperator.from_label("ZII+")  # int = 1
    op.set_type(2)
    op.offdiag_term_grouping()
    op.group_term_sort_by_ladder_int()
    group_ladder_starts = op.group_ladder_bin_starts()
    ptr_size = 2**3 + 1
    idx = 0 * ptr_size + 1  # group 0 + int value = 1
    correct_terms = [
        [("+", 0), ("Z", 1)],
        [("+", 0), ("0", 1)],
        [("+", 0), ("0", 1), ("0", 2), ("Z", 3)],
        [("+", 0), ("Z", 3)],
    ]
    for kk in range(group_ladder_starts[idx], group_ladder_starts[idx + 1]):
        assert op[kk].operators in correct_terms
