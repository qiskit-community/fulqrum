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


TEST_CASE("Test off-diagonal weight pointers 1") {
    QubitOperator op = QubitOperator::from_label("IIII+");
    op += QubitOperator::from_label("III+I");
    op += QubitOperator::from_label("II+II");
    op += QubitOperator::from_label("I+III");
    op += QubitOperator::from_label("+IIII");
    std::vector<std::size_t> ans = {0, 5};
    CHECK(op.offdiag_weight_ptrs() == ans);
}


TEST_CASE("Test off-diagonal weight pointers 2") {
    QubitOperator op = QubitOperator::from_label("IIIII");
    op += QubitOperator::from_label("IIZII");
    op += QubitOperator::from_label("IZZZI");
    op += QubitOperator::from_label("I+III");
    op += QubitOperator::from_label("++III");
    std::vector<std::size_t> ans = {3, 4, 5};
    CHECK(op.offdiag_weight_ptrs() == ans);
    CHECK(op[0].offdiag_weight == 0);
    CHECK(op[1].offdiag_weight == 0);
    CHECK(op[2].offdiag_weight == 0);
    CHECK(op[3].offdiag_weight == 1);
    CHECK(op[4].offdiag_weight == 2);
}


TEST_CASE("Test off-diagonal weight pointers all diag op") {
    QubitOperator op = QubitOperator::from_label("IIIII");
    op += QubitOperator::from_label("IIZII");
    op += QubitOperator::from_label("IZZZI");
    std::vector<std::size_t> ans = {};
    CHECK(op.offdiag_weight_ptrs() == ans);
}