/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#include "doctest.h"
#include <complex>
#include <vector>
#include "fulqrum.hpp"


typedef std::complex<double> complex;



TEST_CASE("Check off-diag weight for identity op is zero") {
    QubitOperator op = QubitOperator::from_label("IIIII");
    std::vector<unsigned int> ans = {0};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for diag op is zero") {
    QubitOperator op = QubitOperator::from_label("IIZII");
    std::vector<unsigned int> ans = {0};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for simple off-weight 1 op") {
    QubitOperator op = QubitOperator::from_label("IIYII");
    std::vector<unsigned int> ans = {1};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for simple off-weight op") {
    QubitOperator op = QubitOperator::from_label("+ZYZX");
    std::vector<unsigned int> ans = {3};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for simple multi-term") {
    QubitOperator op = QubitOperator::from_label("0IYI1") + QubitOperator::from_label("+ZYZX");
    std::vector<unsigned int> ans = {1, 3};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for simple multi-term 2") {
    unsigned int N = 5;
    QubitOperator op = QubitOperator(N, {{"Y", {2}, 1}}) + QubitOperator(N, {{"-X", {0, 2}, 5}});
    std::vector<unsigned int> ans = {1, 2};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Check off-diag weight for simple multi-term 3") {
    unsigned int N = 5;
    QubitOperator op = QubitOperator(N, {{"Y", {2}, 1}}) + QubitOperator(N, {{"-XY", {4, 0, 2}, complex(-1, 1)}});
    std::vector<unsigned int> ans = {1, 3};
    CHECK(op.offdiag_weights() == ans);
}


TEST_CASE("Test off-diagonal weight sorting") {
    QubitOperator op = QubitOperator::from_label("IXI");
    op += QubitOperator::from_label("YXX");
    op += QubitOperator::from_label("X1Y");
    op += QubitOperator::from_label("ZI0");
    op.offdiag_weight_sort();
    std::vector<unsigned int> ans = {0, 1, 2, 3};
    CHECK(op.offdiag_weights() == ans);
}

