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
#include <string>
#include "fulqrum.hpp"


typedef std::complex<double> complex;


/**
 * Test basic Fermi functionality
 *
 */

TEST_CASE("Test empty FermionicOperator") {
    FermionicOperator_t fop = FermionicOperator(5);
    CHECK(fop.size() == 0);
}


TEST_CASE("Test identity FermionicOperator") {
    FermionicOperator_t fop = FermionicOperator(5, {{}});
    CHECK(fop.size() == 1);
}


TEST_CASE("Test identity coefficient FermionicOperator") {
    FermionicOperator_t fop = FermionicOperator(5, {{{}, {}, complex(1, 2)}});
    CHECK(fop[0].coeff == complex(1, 2));
}


TEST_CASE("Test in-place multiplication") {
    FermionicOperator_t fop = FermionicOperator(5, {{{}, {}, 1}});
    fop *= complex(1,2);
    CHECK(fop[0].coeff == complex(1, 2));
}


TEST_CASE("Test multiplication on left") {
    FermionicOperator_t fop = FermionicOperator(5, {{{}, {}, 1}});
    fop = fop * complex(1, 2);
    CHECK(fop[0].coeff == complex(1, 2));
}


TEST_CASE("Test multiplication on right") {
    FermionicOperator_t fop = FermionicOperator(5, {{{}, {}, 1}});
    fop = fop * complex(2, 1);
    CHECK(fop[0].coeff == complex(2, 1));
}


TEST_CASE("Test in-place addition") {
    width_t N = 5;
    std::vector<OpData> ans;
    FermionicOperator_t fop = FermionicOperator(N);
    for(width_t kk=0; kk < N; kk++)
    {
        fop += FermionicOperator(N, {{"-", {kk}, 1.0 / (N + kk)}});
    }
    for(width_t kk=0; kk < N; kk++)
    {
        ans = {OpData("-", kk)};
        CHECK(fop[kk].operators() == ans);
        CHECK(fop[kk].coeff == 1.0 / (N + kk));
    }
}


TEST_CASE("Test addition") {
    width_t N = 5;
    std::vector<OpData> ans;
    FermionicOperator_t fop = FermionicOperator(N);
    for(width_t kk=0; kk < N; kk++)
    {
        fop = fop + FermionicOperator(N, {{"+", {kk}, 1.0 / (N + kk)}});
    }
    for(width_t kk=0; kk < N; kk++)
    {
        ans = {OpData("+", kk)};
        CHECK(fop[kk].operators() == ans);
        CHECK(fop[kk].coeff == 1.0 / (N + kk));
    }
}


TEST_CASE("Test simple one-term operator") {
    FermionicOperator_t op = FermionicOperator(5, {{"XXXXX", {0, 1, 2, 3, 4}, 1}});
    std::vector<OpData> ans = {OpData("X", 0), OpData("X", 1), OpData("X", 2), OpData("X", 3), OpData("X", 4)};
    CHECK(op[0].operators() == ans);
}


TEST_CASE("Test repeated indices simple") {
    FermionicOperator_t op = FermionicOperator(5, {{"+++++", {0, 0, 0, 0, 0}, 1}});
    std::vector<OpData> ans = {OpData("+", 0), OpData("+", 0), OpData("+", 0), OpData("+", 0), OpData("+", 0)};
    CHECK(op[0].operators() == ans);
}


TEST_CASE("Test repeated indices") {
    FermionicOperator_t op = FermionicOperator(5, {{"+--+-", {2, 1, 1, 0, 0}, 1}});
    std::vector<OpData> ans = {OpData("+", 0), OpData("-", 0), OpData("-", 1), OpData("-", 1), OpData("+", 2)};
    CHECK(op[0].operators() == ans);
}


TEST_CASE("Test operator subtraction") {
    width_t N = 5;
    FermionicOperator_t op1 = FermionicOperator(N, {{"+", {0}, 1}}) - FermionicOperator(N, {{"-", {0}, 2}});
    FermionicOperator_t op2 = FermionicOperator(N, {{"+", {0}, 1}}) + FermionicOperator(N, {{"-", {0}, -2}});
    CHECK(op1[0].operators() == op2[0].operators());
    CHECK(op1[0].coeff == op2[0].coeff);
    CHECK(op1[1].coeff == op2[1].coeff);
}


/**
 * Test Fermi combine repeated indices
 *
 */

TEST_CASE("Test combine repeated indices for empty operator does not crash") {
    FermionicOperator_t op = FermionicOperator(5);
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
    CHECK(op_deflate.width == 5);
}


TEST_CASE("Test combine repeated indices for identity") {
    FermionicOperator_t op = FermionicOperator(5, {{}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices does nothing for single element 1") {
    FermionicOperator_t op = FermionicOperator(5, {{"-", {2}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("-", 2)};
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices does nothing for single element 2") {
    FermionicOperator_t op = FermionicOperator(5, {{"+", {1}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("+", 1)};
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices does nothing for single element 3") {
    FermionicOperator_t op = FermionicOperator(5, {{"0", {1}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("0", 1)};
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices does nothing for single element 3") {
    FermionicOperator_t op = FermionicOperator(5, {{"1", {3}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("1", 3)};
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 1") {
    FermionicOperator_t op = FermionicOperator(5, {{"--", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0); // operator is NULL
}


TEST_CASE("Combine repeated indices for single pair of elements 2") {
    FermionicOperator_t op = FermionicOperator(5, {{"-+", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("0", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 3") {
    FermionicOperator_t op = FermionicOperator(5, {{"-0", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 4") {
    FermionicOperator_t op = FermionicOperator(5, {{"-1", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("-", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 5") {
    FermionicOperator_t op = FermionicOperator(5, {{"+-", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("1", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 6") {
    FermionicOperator_t op = FermionicOperator(5, {{"++", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 7") {
    FermionicOperator_t op = FermionicOperator(5, {{"+0", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("+", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 8") {
    FermionicOperator_t op = FermionicOperator(5, {{"+1", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 9") {
    FermionicOperator_t op = FermionicOperator(5, {{"0-", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("-", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 10") {
    FermionicOperator_t op = FermionicOperator(5, {{"0+", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 11") {
    FermionicOperator_t op = FermionicOperator(5, {{"00", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("0", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 12") {
    FermionicOperator_t op = FermionicOperator(5, {{"01", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 13") {
    FermionicOperator_t op = FermionicOperator(5, {{"1-", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 14") {
    FermionicOperator_t op = FermionicOperator(5, {{"1+", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("+", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for single pair of elements 15") {
    FermionicOperator_t op = FermionicOperator(5, {{"10", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for single pair of elements 16") {
    FermionicOperator_t op = FermionicOperator(5, {{"11", {0, 0}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("1", 0)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for three elements 1") {
    FermionicOperator_t op = FermionicOperator(5, {{"+-+", {1, 1, 1}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("+", 1)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}


TEST_CASE("Combine repeated indices for three elements 2") {
    FermionicOperator_t op = FermionicOperator(5, {{"11-", {1, 1, 1}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    CHECK(op_deflate.size() == 0);
}


TEST_CASE("Combine repeated indices for three elements 3") {
    FermionicOperator_t op = FermionicOperator(5, {{"-+-", {1, 1, 1}, 1}});
    FermionicOperator_t op_deflate = op.combine_repeat_indices();
    std::vector<OpData> ans = {OpData("-", 1)};
    CHECK(op_deflate.size() == 1);
    CHECK(op_deflate[0].operators() == ans);
}