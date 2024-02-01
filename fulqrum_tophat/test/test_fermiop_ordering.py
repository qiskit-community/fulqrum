# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# pylint: disable=no-name-in-module
"""Test basic core functionality"""

from fulqrum_tophat import FermionicOperator


def test_normal_empty():
    """Test normal ordering of empty operator returns empty operator"""
    N = 5
    fo = FermionicOperator(N)
    fo_normal = fo.normal_ordering()
    assert fo_normal.num_terms == 0


def test_normal_simple():
    """Test normal ordering of (-+) goes to 1 - (+-)"""
    N = 5
    fo = FermionicOperator(N, [("-", 2), ("+", 2)])
    fo_normal = fo.normal_ordering()
    # Should have two terms now
    assert fo_normal.num_terms == 2
    # First term is identity
    assert fo_normal[0].num_terms == 1
    assert fo_normal[0].operators == []
    # second term is -1*(+-)
    assert fo_normal[1].num_terms == 1
    assert fo_normal[1].operators == [("+", 2), ("-", 2)]
    assert fo_normal[1].coeff == -1


def test_normal_simple():
    """Test normal ordering of (+-) goes (+-)"""
    N = 5
    fo = FermionicOperator(N, [("+", 2), ("-", 2)])
    fo_normal = fo.normal_ordering()
    # Should still have one term
    assert fo_normal.num_terms == 1
    assert fo_normal[0].operators == [("+", 2), ("-", 2)]
    assert fo_normal[0].coeff == 1


def test_normal_simple2():
    """Test normal ordering moves larger index to right, picking up minus sign"""
    N = 5
    fo = FermionicOperator(N, [("-", 4), ("-", 2)])
    fo_normal = fo.normal_ordering()

    assert fo_normal[0].operators == [("-", 2), ("-", 4)]
    assert fo_normal[0].coeff == -1


def test_normal_simple3():
    """Test normal ordering keeps lower indices to the left"""
    N = 5
    fo = FermionicOperator(N, [("-", 0), ("-", 2)])
    fo_normal = fo.normal_ordering()

    assert fo_normal[0].operators == [("-", 0), ("-", 2)]
    assert fo_normal[0].coeff == 1


def test_normal_simple4():
    """Test normal ordering gives 4 terms"""
    N = 5
    fo = FermionicOperator(N, [("+", 2), ("-", 4), ("-", 2)])
    fo_normal = fo.normal_ordering()

    assert fo_normal[0].operators == [("+", 2), ("-", 2), ("-", 4)]
    assert fo_normal[0].coeff == -1


def test_normal_simple5():
    """Test normal ordering gives empty operator for repeated operation and index"""
    fo = FermionicOperator(2, [("-", 0), ("-", 0)])
    fo_normal = fo.normal_ordering()
    assert fo_normal.num_terms == 0

    fo = FermionicOperator(2, [("+", 0), ("+", 0)])
    fo_normal = fo.normal_ordering()
    assert fo_normal.num_terms == 0


def test_normal_simple6():
    """Test normal ordering gives empty operator with stuff in between"""
    fo = FermionicOperator(5, [("-", 0), ("-", 4), ("+", 1), ("-", 0)])
    fo_normal = fo.normal_ordering()
    assert fo_normal.num_terms == 0


def test_normal_simple7():
    """Test expansion of terms gives correct number of output terms"""
    fo = FermionicOperator(
        5, [("-", 2), ("+", 2), ("-", 1), ("+", 1), ("-", 0), ("+", 0)]
    )
    fo_normal = fo.normal_ordering()
    assert fo_normal.num_terms == 8


def test_normal_simple8():
    """Large index - gets moved to far right"""
    fo = FermionicOperator(5, [("-", 4), ("+", 0), ("-", 0)])
    fo_normal = fo.normal_ordering()
    assert fo_normal.operators[-1] == ("-", 4)


def test_normal_simple9():
    """Large index + gets moved middle"""
    fo = FermionicOperator(5, [("+", 4), ("+", 0), ("-", 0)])
    fo_normal = fo.normal_ordering()
    assert fo_normal.operators[1] == ("+", 4)
    assert fo_normal.coeff == -1


def test_index_simple():
    """Test index ordering works for empty operator"""
    fo = FermionicOperator(5)
    fo_index = fo.normal_ordering()
    assert fo_index.num_terms == 0


def test_index_simple2():
    """Test index ordering does not move +/- ops around if same indices"""
    fo = FermionicOperator(5, [("-", 2), ("+", 2)])
    fo_index = fo.index_ordering()
    assert fo_index.operators == [("-", 2), ("+", 2)]


def test_index_simple3():
    """Test index ordering preserves the ordering if already index order"""
    fo = FermionicOperator(5, [("+", 1), ("+", 2), ("-", 4)])
    fo_index = fo.index_ordering()
    assert fo_index.operators == [("+", 1), ("+", 2), ("-", 4)]


def test_index_simple4():
    """Test index ordering single-flip works and changes sign"""
    fo = FermionicOperator(5, [("+", 2), ("-", 1)])
    fo_index = fo.index_ordering()
    assert fo_index.operators == [("-", 1), ("+", 2)]
    assert fo_index.coeff == -1


def test_index_simple5():
    """Test index ordering multi-flip works and changes sign"""
    fo = FermionicOperator(5, [("-", 4), ("-", 2), ("-", 3), ("+", 1)])
    fo_index = fo.index_ordering()
    assert fo_index.operators == [("+", 1), ("-", 2), ("-", 3), ("-", 4)]
    assert fo_index.coeff == -1


def test_index_simple6():
    """Test index ordering kills repeated indices and +/- operators if it can"""
    fo = FermionicOperator(5, [("-", 4), ("-", 2), ("-", 3), ("-", 2)])
    fo_index = fo.index_ordering()
    assert fo_index.num_terms == 0


def test_normal_index_combo():
    """Validate that normal ordering followed by index ordering does what it should"""
    fo = FermionicOperator(5, [("-", 2), ("-", 4), ("+", 2), ("-", 3), ("-", 1)])
    fo_index = fo.normal_ordering().index_ordering()
    assert fo_index[0].operators == [("-", 1), ("-", 3), ("-", 4)]
    assert fo_index[0].coeff == 1
    assert fo_index[1].operators == [("-", 1), ("+", 2), ("-", 2), ("-", 3), ("-", 4)]
    assert fo_index[1].coeff == -1
