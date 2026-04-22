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
#include "fulqrum.hpp"
#include <complex>
#include <vector>

typedef std::complex<double> complex;

TEST_CASE("Test empty QubitOperator")
{
    QubitOperator_t op = QubitOperator(5);
    CHECK(op.num_terms() == 0);
}

TEST_CASE("Test empty QubitOperator 2")
{
    QubitOperator_t op = QubitOperator(5, {});
    CHECK(op.num_terms() == 0);
}

TEST_CASE("Test identity operator")
{
    QubitOperator_t op = QubitOperator(5, {{}});
    CHECK(op.num_terms() == 1);
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test setting coeff for single identity operator")
{
    QubitOperator_t op = QubitOperator(5, {{"", {}, complex(1, 2)}});
    CHECK(op.num_terms() == 1);
    CHECK(op[0].coeff == complex(1, 2));
}

TEST_CASE("Validate QubitOperator inplace multiplication")
{
    QubitOperator_t op = QubitOperator(5, {{}});
    op *= complex(1, 2);
    CHECK(op[0].coeff == complex(1, 2));
}

TEST_CASE("Test multiplication on left")
{
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {4, 3, 2, 1, 0}, complex(0, -1)}});
    QubitOperator_t new_op = 5 * op;
    CHECK(new_op[0].coeff == complex(0, -5));
}

TEST_CASE("Test multiplication on right")
{
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {4, 3, 2, 1, 0}, complex(0, -1)}});
    QubitOperator_t new_op = op * 3.4;
    CHECK(new_op[0].coeff == complex(0, -3.4));
}

TEST_CASE("Test simple multi operators")
{
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0, 1, 2, 3, 4}, 1}});
    std::vector<OpData> ans = {
        OpData("X", 0), OpData("X", 1), OpData("X", 2), OpData("X", 3), OpData("X", 4)};
    CHECK(op[0].operators() == ans);
}

TEST_CASE("Test simple operator weight")
{
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0, 1, 2, 3, 4}, 1}});
    CHECK(op[0].weight() == 5);
}

TEST_CASE("Test simple multi-term operator weight")
{
    QubitOperator_t op = QubitOperator(5, {{"XXXXX", {0, 1, 2, 3, 4}, 1}});
    op += QubitOperator(5, {{"ZY", {2, 3}, 1}});
    CHECK(op[0].weight() == 5);
    CHECK(op[1].weight() == 2);
    std::vector<width_t> ans = {5, 2};
    CHECK(op.weights() == ans);
}

TEST_CASE("Test inplace addition of operators")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N);
    for(unsigned int kk = 0; kk < N; kk++)
    {
        op += QubitOperator(N, {{"Y", {kk}, 1.0 / (N + kk)}});
    }
    for(unsigned int kk = 0; kk < N; kk++)
    {
        std::vector<OpData> ans = {OpData("Y", kk)};
        CHECK(op[kk].operators() == ans);
        CHECK(op[kk].coeff == 1.0 / (N + kk));
    }
}

TEST_CASE("Test addition of operators")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N);
    for(unsigned int kk = 0; kk < N; kk++)
    {
        op = op + QubitOperator(N, {{"Y", {kk}, 1.0 / (N + kk)}});
    }
    for(unsigned int kk = 0; kk < N; kk++)
    {
        std::vector<OpData> ans = {OpData("Y", kk)};
        CHECK(op[kk].operators() == ans);
        CHECK(op[kk].coeff == 1.0 / (N + kk));
    }
}

TEST_CASE("Verify diagonal operator returns true")
{
    width_t N = 25;
    QubitOperator_t op = QubitOperator(N);
    std::vector<std::string> diag_ops = {"Z", "O", "1"};
    unsigned int kk;
    for(kk = 0; kk < N; kk++)
    {
        op += QubitOperator(N, {{diag_ops[kk % 3], {kk}, 1.0 / (N + kk)}});
    }
    CHECK(op.is_diagonal());
}

TEST_CASE("Verify non-diagonal operator returns false")
{
    width_t N = 25;
    QubitOperator_t op = QubitOperator(N);
    std::vector<std::string> diag_ops = {"Z", "O", "1"};
    unsigned int kk;
    for(kk = 0; kk < N; kk++)
    {
        op += QubitOperator(N, {{diag_ops[kk % 3], {kk}, 1.0 / (N + kk)}});
    }
    op += QubitOperator(N, {{"X", {0}, 1}});
    CHECK(!op[kk].is_diagonal());
}

TEST_CASE("Test simple operator sorting")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N, {{"Z0+XY", {4, 0, 3, 1, 2}, 1.0}});
    std::vector<OpData> ans = {
        OpData("0", 0), OpData("X", 1), OpData("Y", 2), OpData("+", 3), OpData("Z", 4)};
    CHECK(op[0].operators() == ans);
}

TEST_CASE("Test operator subtraction")
{
    width_t N = 5;
    QubitOperator_t op1 = QubitOperator(N, {{"Z0+XY", {4, 0, 3, 1, 2}, 1.0}});
    QubitOperator_t op2 = QubitOperator(N, {{"XYZZ", {0, 3, 1, 2}, 5.0}});
    QubitOperator out = op1 - op2;
    CHECK(out[1].coeff == complex(-5, 0));
}

TEST_CASE("Test projector indices")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N, {{"1Z0Z0", {0, 1, 2, 3, 4}, 1.0}});
    std::vector<width_t> ans = {0, 2, 4};
    CHECK(op[0].proj_indices == ans);
}

TEST_CASE("Test projector indices are not set")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N, {{"XZYZ+", {0, 1, 2, 3, 4}, 1.0}});
    std::vector<width_t> ans = {};
    CHECK(op[0].proj_indices == ans);
}

TEST_CASE("Test is_real() 1")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N, {{"++", {2, 3}, complex(1, 1e-14)}});
    CHECK(op.is_real() == true);
}

TEST_CASE("Test is_real() 2")
{
    width_t N = 5;
    QubitOperator_t op = QubitOperator(N, {{"++", {2, 3}, complex(1, 1e-11)}});
    CHECK(op.is_real() == false);
}

TEST_CASE("Test operator iteration")
{
    width_t N = 6;
    QubitOperator_t op = QubitOperator(N, {{"XX", {2, 3}, 1}});
    op += QubitOperator(N, {{"YXXY", {0, 2, 3, 5}, 1}});
    op += QubitOperator(N, {{"ZZZZZZ", {0, 1, 2, 3, 4, 5}, 1}});
    std::size_t kk = 0;
    for(auto item : op)
    {
        CHECK(item.operators() == op[kk].operators());
        kk += 1;
    }
}

TEST_CASE("Test operator from label")
{
    QubitOperator_t op = QubitOperator::from_label("YXIIII");
    std::vector<OpData> ans = {OpData("X", 4), OpData("Y", 5)};
    CHECK(op[0].operators() == ans);
}

TEST_CASE("Test operator from label for all id string")
{
    QubitOperator_t op = QubitOperator::from_label("IIIIII");
    std::vector<OpData> ans = {};
    CHECK(op[0].operators() == ans);
}

TEST_CASE("Test term off-diagonal weight")
{
    QubitOperator_t op = QubitOperator::from_label("IIIIII");
    op += QubitOperator::from_label("IIXIII");
    op += QubitOperator::from_label("01XII+");
    op += QubitOperator::from_label("ZZZZZZ");
    CHECK(op[0].offdiag_weight == 0);
    CHECK(op[1].offdiag_weight == 1);
    CHECK(op[2].offdiag_weight == 2);
    CHECK(op[3].offdiag_weight == 0);
}

TEST_CASE("Test operator splitting")
{
    QubitOperator_t op = QubitOperator(5);
    std::vector<std::string> ops = {"X", "Z", "0", "Y", "1"};
    for(auto item : ops)
    {
        op += QubitOperator(5, {{item, {0}, 1.0}});
    }
    auto [diag, off] = op.split_diagonal();
    CHECK(diag.size() == 3);
    CHECK(off.size() == 2);
}

TEST_CASE("Test operator splitting preserves operator type")
{
    QubitOperator_t op = QubitOperator(5);
    std::vector<std::string> ops = {"X", "Z", "0", "Y", "1"};
    for(auto item : ops)
    {
        op += QubitOperator(5, {{item, {0}, 1.0}});
    }
    op.set_type(2);
    auto [diag, off] = op.split_diagonal();
    CHECK(diag.type == 2);
    CHECK(off.type == 2);
}

TEST_CASE("Test combining terms doesn't crash for an empty operator")
{
    QubitOperator op = QubitOperator(5);
    QubitOperator new_op = op.combine_repeated_terms();
    CHECK(new_op.size() == 0);
}

TEST_CASE("Test QubitOperator combining terms")
{
    QubitOperator op = QubitOperator::from_label("IZYXI");
    op += QubitOperator::from_label("IZYXI");
    op += QubitOperator::from_label("IZYXI");
    op += QubitOperator::from_label("IZYXI");
    op += QubitOperator::from_label("IZYXI");
    op += QubitOperator::from_label("IIIII");
    op += QubitOperator::from_label("I0YXI");
    QubitOperator new_op = op.combine_repeated_terms();
    CHECK(new_op.size() == 3);
    CHECK(new_op[1].coeff == complex(5, 0));
}

TEST_CASE("Test diagonal QubitOperator properties")
{
    QubitOperator op = QubitOperator(3);
    std::vector<std::string> labels = {"III", "ZZ1", "Z0Z", "IZI", "ZI0"};
    for(auto label : labels)
    {
        op += QubitOperator::from_label(label);
    }
    CHECK(op.width == 3);
    CHECK(op.size() == 5);
    CHECK(op.num_terms() == 5);
    CHECK(op.sorted == 0);
    CHECK(op.off_weight_sorted == 0);
    CHECK(op[0].offdiag_weight == 0);
    CHECK(op[0].operators().size() == 0);
    std::vector<OpData> ans1 = {OpData("1", 0), OpData("Z", 1), OpData("Z", 2)};
    CHECK(op[1].operators() == ans1);
    CHECK(op[1].offdiag_weight == 0);
    std::vector<OpData> ans2 = {OpData("Z", 0), OpData("0", 1), OpData("Z", 2)};
    CHECK(op[2].operators() == ans2);
    CHECK(op[2].offdiag_weight == 0);
    std::vector<OpData> ans3 = {OpData("Z", 1)};
    CHECK(op[3].operators() == ans3);
    CHECK(op[3].offdiag_weight == 0);
    std::vector<OpData> ans4 = {OpData("0", 0), OpData("Z", 2)};
    CHECK(op[4].operators() == ans4);
    CHECK(op[4].offdiag_weight == 0);
}

TEST_CASE("Test real phase 1")
{
    QubitOperator op = QubitOperator::from_label("IIII");
    CHECK(op[0].real_phase == 1);
}

TEST_CASE("Test real phase 2")
{
    QubitOperator op = QubitOperator::from_label("IYII");
    CHECK(op[0].real_phase == 0);
}

TEST_CASE("Test real phase 3")
{
    QubitOperator op = QubitOperator::from_label("XZ01");
    CHECK(op[0].real_phase == 1);
}

TEST_CASE("Test real phase 4")
{
    QubitOperator op = QubitOperator::from_label("IYIY");
    CHECK(op[0].real_phase == -1);
}

TEST_CASE("Test real phase 5")
{
    QubitOperator op = QubitOperator::from_label("IYYY");
    CHECK(op[0].real_phase == 0);
}

TEST_CASE("Test real phase 6")
{
    QubitOperator op = QubitOperator::from_label("YYYY");
    CHECK(op[0].real_phase == 1);
}

TEST_CASE("Test real phase 7")
{
    QubitOperator op = QubitOperator::from_label("YYYYYY");
    CHECK(op[0].real_phase == -1);
}

TEST_CASE("Test removal of diagonal terms")
{
    width_t N = 6;
    QubitOperator op = QubitOperator(N);
    std::vector<std::string> diag_ops = {"I", "X", "Z", "0", "Y", "I"};
    unsigned int kk;
    for(kk = 0; kk < N; kk++)
    {
        op += QubitOperator(N, {{diag_ops[kk], {kk}, 1.0 / (N + kk)}});
    }
    CHECK(std::abs(op.constant_energy() - 0.25757575757575757) < 1e-14);
    QubitOperator out = op.remove_constant_terms();
    CHECK(out.size() == 4);
    CHECK(out.constant_energy() == 0);
}
