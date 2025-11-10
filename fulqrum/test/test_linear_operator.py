# Fulqrum
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test method(s) in linear operator that are not covered in other tests"""
from pathlib import Path
import pytest

from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian


_path = Path(__file__).parent / "data/h2o.json"
FOP = FermionicOperator.from_json(_path)
NEW_OP = FOP.extended_jw_transformation()


def test_get_n_th_bitstring():
    """Test get_n_th_bitstring() method returns the correct bitstring
    Both Python and Fulqrum's emhash8::HashMap dictionaries retain
    the insertion order.
    """
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    keys = list(full_dist.keys())
    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(NEW_OP, S)

    for n in range(len(S)):
        bs = Hsub.get_n_th_bitstring(n)
        assert bs == keys[n]


def test_get_n_th_bitstring_out_of_range():
    """Test get_n_th_bitstring() raises for an index that is >= subspace size."""
    full_dist = {}
    for kk in range(2**14):
        full_dist[bin(kk)[2:].zfill(14)] = None

    S = Subspace(full_dist)
    Hsub = SubspaceHamiltonian(NEW_OP, S)

    with pytest.raises(ValueError):
        Hsub.get_n_th_bitstring(len(S))
