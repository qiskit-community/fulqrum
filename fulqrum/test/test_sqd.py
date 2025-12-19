import pytest
import numpy as np
from fulqrum.core.subspace import Subspace
from fulqrum.core.sqd import (
    postselect_by_hamming_right_and_left,
    subsample,
    recover_configurations,
    get_carryover_full_strs,
)


def test_postselect_by_hamming_right_and_left():
    bitstrings = ["11101001", "01111110", "01111100", "01011100"]
    probs = np.array([0.1, 0.2, 0.4, 0.3])
    bitstrings_post, probs_post = postselect_by_hamming_right_and_left(
        bitstrings, probs, right=2, left=3
    )
    expected_bitstrings = ["11101001", "01111100"]
    expected_probs = np.array([0.2, 0.8])
    assert bitstrings_post == expected_bitstrings
    np.testing.assert_allclose(probs_post, expected_probs)


def test_subsample():
    # 4 qubit full list of bitstring
    bitstrings = [
        "0000",
        "0001",
        "0010",
        "0011",
        "0100",
        "0101",
        "0110",
        "0111",
        "1000",
        "1001",
        "1010",
        "1011",
        "1100",
        "1101",
        "1110",
        "1111"
    ]

    uniform_probs = np.array(
        [1 / len(bitstrings) for _ in bitstrings]
    )

    samples_per_batch = 2
    subsampled_batch = subsample(
        bitstrings=bitstrings,
        weights=uniform_probs,
        samples_per_batch=samples_per_batch,
        seed=0
    )

    assert len(subsampled_batch) == samples_per_batch
    
    with pytest.raises(ValueError):
        samples_per_batch = 0
        subsampled_batch = subsample(
            bitstrings=bitstrings,
            weights=uniform_probs,
            samples_per_batch=samples_per_batch,
            seed=0
        )

    with pytest.raises(ValueError):
        samples_per_batch = -1
        subsampled_batch = subsample(
            bitstrings=bitstrings,
            weights=uniform_probs,
            samples_per_batch=samples_per_batch,
            seed=0
        )
    
    bitstrings = [
        "0000",
        "0001",
        "0010",
        "0011",
    ]

    probs = np.array([1.0, 0, 0, 0])

    samples_per_batch = 1
    subsampled_batch = subsample(
        bitstrings=bitstrings,
        weights=probs,
        samples_per_batch=samples_per_batch,
        seed=0
    )

    assert len(subsampled_batch) == samples_per_batch


@pytest.mark.parametrize("num_bits", [4, 258, 10000])
def test_recover_configurations_0s_to_1s(num_bits):
    bitstrings = ["0" * num_bits]
    probs = np.array([1.0])
    occs_a = np.array([1.0] * (num_bits // 2))
    occs_b = np.array([1.0] * (num_bits // 2))
    num_a = num_bits // 2
    num_b = num_bits // 2
    expected = ["1" * num_bits]
    expected_probs = np.array([1.0])
    bs_rec, probs_rec = recover_configurations(
        bitstrings, probs, occs_a, occs_b, num_elec_a=num_a, num_elec_b=num_b
    )

    assert bs_rec == expected
    np.testing.assert_allclose(probs_rec, expected_probs)


def test_recover_configurations_mixed_to_1s():
    bitstrings = ["0000", "0110"]
    probs = np.array([0.5, 0.5])
    occs_a = np.array([1.0, 1.0])
    occs_b = np.array([1.0, 1.0])
    num_a = 2
    num_b = 2
    expected = ["1111"]
    expected_probs = np.array([1.0])
    bs_rec, probs_rec = recover_configurations(
        bitstrings, probs, occs_a, occs_b, num_elec_a=num_a, num_elec_b=num_b
    )

    assert bs_rec == expected
    np.testing.assert_allclose(probs_rec, expected_probs)


@pytest.mark.parametrize("num_bits", [4, 258, 10000])
def test_recover_configurations_1s_to_0s(num_bits):
    bitstrings = ["1" * num_bits]
    probs = np.array([1.0])
    occs_a = np.array([0.0] * (num_bits // 2))
    occs_b = np.array([0.0] * (num_bits // 2))
    num_a = 0
    num_b = 0
    expected = ["0" * num_bits]
    expected_probs = np.array([1.0])
    bs_rec, probs_rec = recover_configurations(
        bitstrings, probs, occs_a, occs_b, num_elec_a=num_a, num_elec_b=num_b
    )

    assert bs_rec == expected
    np.testing.assert_allclose(probs_rec, expected_probs)


def test_recover_configurations_mismatch_orbitals():
    bitstrings = ["1111"] # bN ... b0 / aN ... a0
    probs = np.array([1.0])
    occs_a = np.array([0.0, 0.0]) # a0 ... aN
    occs_b = np.array([0.0, 1.0]) # b0 ... bN
    num_a = 0
    num_b = 1
    expected_bs = ["1000"] # bN ... b0 / aN ... a0
    expected_probs = np.array([1.0])

    bs_rec, probs_rec = recover_configurations(
        bitstrings, probs, occs_a, occs_b, num_elec_a=num_a, num_elec_b=num_b
    )

    assert bs_rec == expected_bs
    np.testing.assert_allclose(probs_rec, expected_probs)


def test_get_carryover_full_strs():
    bitstrings = [['011101', '101011', '110110']]
    S = Subspace(bitstrings)
    abs_amps = np.array([0.1, 0.3, 0.05]) # Test only. abs_amps ** 2 != 1

    carryover = get_carryover_full_strs(subspace=S, abs_amps=abs_amps, threshold=0.08)
    assert carryover == [('101011', 0.3), ('011101', 0.1)]

    carryover = get_carryover_full_strs(subspace=S, abs_amps=abs_amps, threshold=0.20)
    assert carryover == [('101011', 0.3)]

    carryover = get_carryover_full_strs(subspace=S, abs_amps=abs_amps, threshold=0.04)
    assert carryover == [('101011', 0.3), ('011101', 0.1), ('110110', 0.05)]

    carryover = get_carryover_full_strs(subspace=S, abs_amps=abs_amps, threshold=0.40)
    assert carryover == []
