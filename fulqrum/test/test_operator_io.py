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
    assert FOP.size() == new_fop.size()
    try:
        os.remove("lih.json")
    except FileNotFoundError:
        pass


def test_fermionic_xz():
    """Test round-trip of fermionic to xz"""
    FOP.to_json("lih.json.xz", overwrite=True)
    new_fop = fq.FermionicOperator.from_json("lih.json.xz")
    assert FOP.width == new_fop.width
    assert FOP.size() == new_fop.size()
    try:
        os.remove("lih.json.xz")
    except FileNotFoundError:
        pass
    try:
        os.remove("lih.json")
    except FileNotFoundError:
        pass


def test_qubit_json():
    """Test round-trip of qubitoperator to json"""
    OP.to_json("lih_op.json", overwrite=True)
    new_op = fq.QubitOperator.from_json("lih_op.json")
    assert OP.width == new_op.width
    assert OP.size() == new_op.size()
    try:
        os.remove("lih_op.json")
    except FileNotFoundError:
        pass


def test_qubit_xz():
    """Test round-trip of qubitoperator to xz"""
    OP.to_json("lih_op.json.xz", overwrite=True)
    new_op = fq.QubitOperator.from_json("lih_op.json.xz")
    assert OP.width == new_op.width
    assert OP.size() == new_op.size()
    try:
        os.remove("lih_op.json.xz")
    except FileNotFoundError:
        pass
    try:
        os.remove("lih_op.json")
    except FileNotFoundError:
        pass


def test_fermionic_to_qubit_method_in_json():
    """Test round-trip of fermionic to json"""
    OP.to_json("lih_jw.json", overwrite=True)
    assert OP.type == 2
    new_op = fq.QubitOperator.from_json("lih_jw.json")
    assert new_op.type == 2
    try:
        os.remove("lih_jw.json")
    except FileNotFoundError:
        pass


def test_qubit_json_preserves_real_phase(tmp_path):
    """Test round-trip of qubitoperator to json keeps real phases"""
    op = fq.QubitOperator(
        4,
        [
            ("YY", [0, 1], 1.0),
            ("XY", [0, 1], 1.0),
            ("XX", [0, 1], 1.0),
            ("YYY", [0, 1, 2], 1.0),
            ("YYYY", [0, 1, 2, 3], 1.0),
        ],
    )
    filename = tmp_path / "phase_op.json"
    op.to_json(str(filename), overwrite=True)
    new_op = fq.QubitOperator.from_json(str(filename))
    assert list(new_op.real_phases()) == [-1, 0, 1, 0, 1]
    assert list(new_op.real_phases()) == list(op.real_phases())
    assert new_op.is_real() == op.is_real()
    assert list(new_op.offdiag_structures()) == list(op.offdiag_structures())
