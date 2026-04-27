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
#include <string>
#include <vector>

typedef std::complex<double> complex;

TEST_CASE("Test group term sort by ladder integers 1")
{
    QubitOperator op = QubitOperator::from_label("III+");
    op += QubitOperator::from_label("-II+");
    op += QubitOperator::from_label("+II-");
    op += QubitOperator::from_label("+II+");
    op += QubitOperator::from_label("++-I");
    op += QubitOperator::from_label("---I");
    op += QubitOperator::from_label("IZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int();
    CHECK(op.group_ptrs() == std::vector<std::size_t>{0, 1, 2, 5, 7});
    CHECK(op.ladder_integers() == std::vector<width_t>{MAX_WIDTH, 1, 1, 2, 3, 0, 6});
}

TEST_CASE("Test group term sort by ladder_width=3")
{
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(3);
    CHECK(op.ladder_integers() == std::vector<width_t>{MAX_WIDTH, 0, 7});
}

TEST_CASE("Test group term sort by ladder_width=2")
{
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(2);
    CHECK(op.ladder_integers() == std::vector<width_t>{MAX_WIDTH, 0, 3});
}

TEST_CASE("Test group term sort by ladder_width=1")
{
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(1);
    CHECK(op.ladder_integers() == std::vector<width_t>{MAX_WIDTH, 0, 1});
}

TEST_CASE("Test group term sort ladder_width=3 2")
{
    QubitOperator op = QubitOperator::from_label("_+II");
    op += QubitOperator::from_label("++II");
    op += QubitOperator::from_label("IIIZ");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(3);
    CHECK(op.ladder_integers() == std::vector<width_t>{MAX_WIDTH, 1, 3});
}

TEST_CASE("Verify that ladder integers are correct for terms in each group")
{
    // group 2
    QubitOperator op = QubitOperator::from_label("-II+"); // int = 1
    op += QubitOperator::from_label("+II-"); // int = 2
    op += QubitOperator::from_label("+II+"); // int = 3
    // group 1
    op += QubitOperator::from_label("II+-"); // int = 2
    op += QubitOperator::from_label("II-+"); // int = 1
    // group 3
    op += QubitOperator::from_label("I++-"); // int = 6
    op += QubitOperator::from_label("I---"); // int = 0
    op += QubitOperator::from_label("I+++"); // int = 7
    op.set_type(2);
    op.group_sort();
    op.group_term_sort_by_ladder_int();
    CHECK(op.terms_by_group(1).ladder_integers() == std::vector<width_t>{1, 2});
    CHECK(op.terms_by_group(2).ladder_integers() == std::vector<width_t>{1, 2, 3});
    CHECK(op.terms_by_group(3).ladder_integers() == std::vector<width_t>{0, 6, 7});
}

TEST_CASE("Verify that ladder integers are correct for terms in each group 2")
{
    // group 1
    QubitOperator op = QubitOperator::from_label("-II+"); // int = 1,
    op += QubitOperator::from_label("-ZZ+"); // int = 1,
    op += QubitOperator::from_label("+II-"); // int = 2,
    op += QubitOperator::from_label("-ZZ-"); // int = 0,
    op += QubitOperator::from_label("+0I-"); // int = 2,
    op += QubitOperator::from_label("+01+"); // int = 3,
    // group 2
    op += QubitOperator::from_label("II-+"); // int = 1
    op += QubitOperator::from_label("II++"); // int = 3
    op += QubitOperator::from_label("ZZ--"); // int = 0
    op += QubitOperator::from_label("ZZ+-"); // int = 2
    op += QubitOperator::from_label("Z0+-"); // int = 2
    op += QubitOperator::from_label("Z0-+"); // int = 1
    op += QubitOperator::from_label("ZI++"); // int = 3
    op.set_type(2);
    op.group_sort();
    op.group_term_sort_by_ladder_int();
    CHECK(op.terms_by_group(2).ladder_integers() == std::vector<width_t>{0, 1, 1, 2, 2, 3});
    CHECK(op.terms_by_group(1).ladder_integers() == std::vector<width_t>{0, 1, 1, 2, 2, 3, 3});
}

TEST_CASE("Verify that off-diag indices are correct for ladder operator terms")
{
    QubitOperator op = QubitOperator::from_label("I+I+");
    op += QubitOperator::from_label("+II+");
    op += QubitOperator::from_label("+III");
    op += QubitOperator::from_label("-III");
    op += QubitOperator::from_label("I++I");
    op += QubitOperator::from_label("I--I");
    op += QubitOperator::from_label("----");
    op += QubitOperator::from_label("--I-");
    op.set_type(2);
    op.group_sort();
    auto inds_list = op.group_offdiag_indices();
    CHECK(inds_list[0] == std::vector<width_t>{0, 2});
    CHECK(inds_list[1] == std::vector<width_t>{3});
    CHECK(inds_list[2] == std::vector<width_t>{0, 3});
    CHECK(inds_list[3] == std::vector<width_t>{1, 2});
    CHECK(inds_list[4] == std::vector<width_t>{0, 2, 3});
    CHECK(inds_list[5] == std::vector<width_t>{0, 1, 2, 3});
}

TEST_CASE("Verify that off-diag indices are correct for ladder operator terms 2")
{
    QubitOperator op = QubitOperator::from_label("IXIY");
    op += QubitOperator::from_label("XIIX");
    op += QubitOperator::from_label("YIII");
    op += QubitOperator::from_label("XIII");
    op += QubitOperator::from_label("IXXI");
    op += QubitOperator::from_label("IYYI");
    op += QubitOperator::from_label("YYYY");
    op += QubitOperator::from_label("YYIY");
    op.set_type(2);
    op.group_sort();
    auto inds_list = op.group_offdiag_indices();
    CHECK(inds_list[0] == std::vector<width_t>{0, 2});
    CHECK(inds_list[1] == std::vector<width_t>{3});
    CHECK(inds_list[2] == std::vector<width_t>{0, 3});
    CHECK(inds_list[3] == std::vector<width_t>{1, 2});
    CHECK(inds_list[4] == std::vector<width_t>{0, 2, 3});
    CHECK(inds_list[5] == std::vector<width_t>{0, 1, 2, 3});
}

TEST_CASE("Test ladder integers works for various ladder widths")
{
    QubitOperator op = QubitOperator::from_label("I+-Z0X+-+Y");

    op.set_type(2);
    op.group_term_sort_by_ladder_int(5);
    CHECK(op.ladder_integers()[0] == std::stoi("10101", 0, 2));
    op.group_term_sort_by_ladder_int(4);
    CHECK(op.ladder_integers()[0] == std::stoi("0101", 0, 2));
    op.group_term_sort_by_ladder_int(3);
    CHECK(op.ladder_integers()[0] == std::stoi("101", 0, 2));
    op.group_term_sort_by_ladder_int(2);
    CHECK(op.ladder_integers()[0] == std::stoi("01", 0, 2));
    op.group_term_sort_by_ladder_int(1);
    CHECK(op.ladder_integers()[0] == std::stoi("1", 0, 2));
}
