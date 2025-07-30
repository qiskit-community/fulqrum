# Fulqrum
# Copyright (C) 2024, IBM
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
