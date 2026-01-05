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
import numpy as np
import qiskit_addon_fulqrum as fq


def test_worst_case1():
    """Worst case amplitude single type=1 group"""
    H = fq.QubitOperator(3, [])
    for item in ["IIX", "ZZX", "Z0X", "IZX", "ZIX"]:
        H += fq.QubitOperator.from_label(item)

    worst_amps = H.worst_case_offdiag_group_amplitudes()
    assert np.allclose(worst_amps, np.asarray([5.0]))


def test_worst_case2():
    """Worst case amplitude two type=1 groups"""
    H = fq.QubitOperator(3, [])
    for item in ["IIX", "ZZX", "Z0X", "IZX", "ZIX"]:
        H += fq.QubitOperator.from_label(item)

    for item in ["XIX", "YZX", "Y0X", "YZX"]:
        H += fq.QubitOperator.from_label(item)

    worst_amps = H.worst_case_offdiag_group_amplitudes()
    assert np.allclose(worst_amps, np.asarray([5.0, 4.0]))


def test_worst_case3():
    """Worst case amplitude two type=1 groups with a negative coeff"""
    H = fq.QubitOperator(3, [])
    for item in ["IIX", "ZZX", "Z0X", "IZX", "ZIX"]:
        H += fq.QubitOperator.from_label(item)

    H += fq.QubitOperator(
        3, [("XIX", range(3), 1.0), ("YZX", range(3), -1.0), ("X0X", range(3), 1.0)]
    )

    worst_amps = H.worst_case_offdiag_group_amplitudes()
    assert np.allclose(worst_amps, np.asarray([5.0, 3.0]))


def test_worst_case4():
    """Worst case amplitude two type=1 with 3 groups"""
    H = fq.QubitOperator(
        3, [("IXI", range(3), -1.0), ("YZZ", range(3), -1.0), ("X0X", range(3), -1.0)]
    )
    worst_amps = H.worst_case_offdiag_group_amplitudes()
    assert np.allclose(worst_amps, np.asarray([1.0, 1.0, 1.0]))


def test_worst_case5():
    """Worst case amplitude type=2 operator one group"""
    H = fq.QubitOperator(3, [("++I", range(3), -1.0), ("++Z", range(3), -1.0)])
    H.set_type(2)
    worst_amps = H.worst_case_offdiag_group_amplitudes()
    assert np.allclose(worst_amps, np.asarray([2.0]))


def test_worst_case6():
    """Worst case amplitude type=2 operator two groups"""
    H = fq.QubitOperator(3, [("++I", range(3), -1.0), ("+-Z", range(3), -1.0)])
    H += fq.QubitOperator(3, [("---", range(3), 2.0), ("+++", range(3), -1.0)])
    H.set_type(2)
    worst_amps = H.worst_case_offdiag_group_amplitudes()
    np.allclose(worst_amps, np.asarray([1.0, 2.0]))
