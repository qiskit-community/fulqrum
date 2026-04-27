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
"""Test operator IO functionality"""

import os
from pathlib import Path
import fulqrum as fq

_path = Path(__file__).parent / "data/lih.json"
FOP = fq.FermionicOperator.from_json(_path)
OP = FOP.extended_jw_transformation()


def test_fermionic_json():
    """Test round-trip of fermionic to json"""
    FOP.to_json("lih.json", overwrite=True)
    new_fop = fq.FermionicOperator.from_json("lih.json")
    assert FOP.width == new_fop.width
    assert FOP.num_terms == new_fop.num_terms
    try:
        os.remove("lih.json")
    except FileNotFoundError:
        pass


def test_fermionic_xz():
    """Test round-trip of fermionic to xz"""
    FOP.to_json("lih.json.xz", overwrite=True)
    new_fop = fq.FermionicOperator.from_json("lih.json.xz")
    assert FOP.width == new_fop.width
    assert FOP.num_terms == new_fop.num_terms
    try:
        os.remove("lih.json.xz")
    except FileNotFoundError:
        pass


def test_qubit_json():
    """Test round-trip of qubitoperator to json"""
    OP.to_json("lih_op.json", overwrite=True)
    new_op = fq.QubitOperator.from_json("lih_op.json")
    assert OP.width == new_op.width
    assert OP.num_terms == new_op.num_terms
    try:
        os.remove("lih_op.json")
    except FileNotFoundError:
        pass


def test_qubit_xz():
    """Test round-trip of qubitoperator to xz"""
    OP.to_json("lih_op.json.xz", overwrite=True)
    new_op = fq.QubitOperator.from_json("lih_op.json.xz")
    assert OP.width == new_op.width
    assert OP.num_terms == new_op.num_terms
    try:
        os.remove("lih_op.json.xz")
    except FileNotFoundError:
        pass
