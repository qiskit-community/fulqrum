# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test projector functionality"""
import numpy as np
import fulqrum as fq


def test_bitstring_term_proj_validation1():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("+ZII")
    bits = fq.Bitset("0001")
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation2():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("III1")
    bits = fq.Bitset("0001")
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation3():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("III1")
    bits = fq.Bitset("0000")
    assert np.allclose(op.projector_oper_validation(bits), [0])


def test_bitstring_term_proj_validation4():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("III0")
    bits = fq.Bitset("0001")
    assert np.allclose(op.projector_oper_validation(bits), [0])


def test_bitstring_term_proj_validation5():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("0II0")
    bits = fq.Bitset("0000")
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation6():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("0II0")
    bits = fq.Bitset("0000")
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation7():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("1I1I")
    bits = fq.Bitset("1010")
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation8():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("1IZI")
    bits = fq.Bitset("0010")
    assert np.allclose(op.projector_oper_validation(bits), [0])


def test_bitstring_term_proj_validation9():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("1" + "Z" * 99)
    bits = fq.Bitset("1" * 100)
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation10():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("I1" + "Z" * 98)
    bits = fq.Bitset("01" * 50)
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation11():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("I1" + "Z" * 97 + "0")
    bits = fq.Bitset("10" * 50)
    assert np.allclose(op.projector_oper_validation(bits), [0])


def test_bitstring_term_proj_validation12():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("1" + "I" * 43)
    bits = fq.Bitset("1" * 44)
    assert np.allclose(op.projector_oper_validation(bits), [1])


def test_bitstring_term_proj_validation13():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("0" + "I" * 43)
    bits = fq.Bitset("1" * 44)
    assert np.allclose(op.projector_oper_validation(bits), [0])


def test_bitstring_term_proj_validation14():
    """Test projector validation of term and bitstring"""
    op = fq.QubitOperator.from_label("1II1")
    bits = fq.Bitset("1100")
    assert np.allclose(op.projector_oper_validation(bits), [0])
