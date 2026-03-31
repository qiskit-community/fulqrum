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


TEST_CASE("Test empty QubitOperator") {
    QubitOperator_t op = QubitOperator(5);
    CHECK(op.num_terms() == 0);
}


TEST_CASE("Test empty QubitOperator 2") {
    QubitOperator_t op = QubitOperator(5, {});
    CHECK(op.num_terms() == 0);
}


TEST_CASE("Test identity operator") {
    QubitOperator_t op = QubitOperator(5, {{}});
    CHECK(op.num_terms() == 1);
    CHECK(op[0].coeff == complex(1,0));
}


TEST_CASE("Test setting coeff for single identity operator") {
    QubitOperator_t op = QubitOperator(5, {{"", {}, complex(1,2)}});
    CHECK(op.num_terms() == 1);
    CHECK(op[0].coeff == complex(1,2));
}


TEST_CASE("Validate QubitOperator inplace multiplication") {
    QubitOperator_t op = QubitOperator(5, {{}});
    op *= complex(1,2);
    CHECK(op[0].coeff == complex(1,2));
}


TEST_CASE("Test multiplication on left") {
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {4, 3, 2, 1, 0}, complex(0,-1)}});
    QubitOperator_t new_op = 5 * op;
    CHECK(new_op[0].coeff == complex(0,-5));
}


TEST_CASE("Test multiplication on right") {
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {4, 3, 2, 1, 0}, complex(0,-1)}});
    QubitOperator_t new_op = op * 3.4;
    CHECK(new_op[0].coeff == complex(0,-3.4));
}


TEST_CASE("Test simple multi operators") {
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0,1,2,3,4}, 1}});
    std::vector<OpData> ans = {OpData("X", 0), OpData("X", 1), OpData("X", 2), OpData("X", 3), OpData("X", 4)};
    CHECK(op[0].operators() == ans);
}


TEST_CASE("Test simple operator weight") {
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0,1,2,3,4}, 1}});
    CHECK(op[0].weight() == 5);
}


TEST_CASE("Test simple multi-term operator weight") {
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0,1,2,3,4}, 1}});
    op += QubitOperator(5, {{"ZY", {2,3}, 1}});
    CHECK(op[0].weight() == 5);
    CHECK(op[1].weight() == 2);
    std::vector<unsigned int> ans = {5, 2};
    CHECK(op.weights() == ans);
}


TEST_CASE("Test inplace addition of operators") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N);
    for(unsigned int kk=0; kk < N; kk++)
    {
        op += QubitOperator(N, {{"Y", {kk}, 1.0/(N+kk)}});
    }
    for(unsigned int kk=0; kk < N; kk++)
    {
        std::vector<OpData> ans = {OpData("Y", kk)};
        CHECK(op[kk].operators() == ans);
        CHECK(op[kk].coeff == 1.0/(N+kk));
    }
}


TEST_CASE("Test addition of operators") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N);
    for(unsigned int kk=0; kk < N; kk++)
    {
        op = op + QubitOperator(N, {{"Y", {kk}, 1.0/(N+kk)}});
    }
    for(unsigned int kk=0; kk < N; kk++)
    {
        std::vector<OpData> ans = {OpData("Y", kk)};
        CHECK(op[kk].operators() == ans);
        CHECK(op[kk].coeff == 1.0/(N+kk));
    }
}


TEST_CASE("Verify diagonal operator returns true") {
    unsigned int N = 25;
    QubitOperator_t op = QubitOperator(N);
    std::vector<std::string> diag_ops = {"Z", "O", "1"};
    unsigned int kk;
    for(kk=0; kk < N; kk++)
    {
        op +=  QubitOperator(N, {{diag_ops[kk % 3], {kk}, 1.0/(N+kk)}});
    }
    CHECK(op[kk].is_diagonal());
}


TEST_CASE("Verify non-diagonal operator returns false") {
    unsigned int N = 25;
    QubitOperator_t op = QubitOperator(N);
    std::vector<std::string> diag_ops = {"Z", "O", "1"};
    unsigned int kk;
    for(kk=0; kk < N; kk++)
    {
        op +=  QubitOperator(N, {{diag_ops[kk % 3], {kk}, 1.0/(N+kk)}});
    }
    op +=  QubitOperator(N, {{"X", {0}, 1}});
    CHECK(!op[kk].is_diagonal());
}


TEST_CASE("Test simple operator sorting") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N, {{"Z0+XY", {4,0,3,1,2}, 1.0}});
    std::vector<OpData> ans = {OpData("0", 0), OpData("X", 1), OpData("Y", 2), OpData("+", 3), OpData("Z", 4)};
    CHECK(op[0].operators() == ans);
}


TEST_CASE("Test operator subtraction") {
    unsigned int N = 5;
    QubitOperator_t op1 = QubitOperator(N, {{"Z0+XY", {4,0,3,1,2}, 1.0}});
    QubitOperator_t op2 = QubitOperator(N, {{"XYZZ", {0,3,1,2}, 5.0}});
    QubitOperator out = op1 - op2;
    CHECK(out[1].coeff == complex(-5,0));
}


TEST_CASE("Test projector indices") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N, {{"1Z0Z0", {0,1,2,3,4}, 1.0}});
    std::vector<unsigned int> ans = {0,2,4};
    CHECK(op[0].proj_indices == ans);
}


TEST_CASE("Test projector indices are not set") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N, {{"XZYZ+", {0,1,2,3,4}, 1.0}});
    std::vector<unsigned int> ans = {};
    CHECK(op[0].proj_indices == ans);
}


TEST_CASE("Test is_real() 1") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N, {{"++", {2,3}, complex(1, 1e-14)}});
    CHECK(op.is_real() == true);
}


TEST_CASE("Test is_real() 2") {
    unsigned int N = 5;
    QubitOperator_t op = QubitOperator(N, {{"++", {2,3}, complex(1, 1e-11)}});
    CHECK(op.is_real() == false);
}