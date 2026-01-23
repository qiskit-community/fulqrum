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
import fulqrum as fq


def test_real_phase1():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("IIII")
    assert op.real_phases()[0] == 1


def test_real_phase2():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("IYII")
    assert op.real_phases()[0] == 0


def test_real_phase3():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("XZ01")
    assert op.real_phases()[0] == 1


def test_real_phase4():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("IYIY")
    assert op.real_phases()[0] == -1


def test_real_phase5():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("IYYY")
    assert op.real_phases()[0] == 0


def test_real_phase6():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("YYYY")
    assert op.real_phases()[0] == 1


def test_real_phase7():
    """Test real_phase of operator term"""
    op = fq.QubitOperator.from_label("YYYYYY")
    assert op.real_phases()[0] == -1
