# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""

from fulqrum_tophat.convert.qiskit import FermionicOp_to_FermionicOperator
from qiskit_nature.second_q.operators import FermionicOp


def test_simple_conversion():
    """Test simple conversion from FermionicOp to FermionicOperator"""
    fo = FermionicOp({"-_3 -_2 +_3 +_1": 1, "-_0 +_1": -1j}, num_spin_orbitals=5)
    new_op = FermionicOp_to_FermionicOperator(fo)
    assert new_op.num_terms == 2
    assert new_op[0].coeff == 1
    assert new_op[0].operators == [("-", 3), ("-", 2), ("+", 3), ("+", 1)]
    assert new_op[1].coeff == -1j
    assert new_op[1].operators == [("-", 0), ("+", 1)]
