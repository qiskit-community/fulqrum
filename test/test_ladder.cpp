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


TEST_CASE("Test group term sort by ladder integers 1") {
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
    CHECK(op.ladder_integers() == std::vector<unsigned int>{MAX_UINT, 1, 1, 2, 3, 0, 6});
}


TEST_CASE("Test group term sort by ladder_width=3") {
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(3);
    CHECK(op.ladder_integers() == std::vector<unsigned int>{MAX_UINT, 0, 7});
}


TEST_CASE("Test group term sort by ladder_width=2") {
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(2);
    CHECK(op.ladder_integers() == std::vector<unsigned int>{MAX_UINT, 0, 3});
}


TEST_CASE("Test group term sort by ladder_width=1") {
    QubitOperator op = QubitOperator::from_label("++++++");
    op += QubitOperator::from_label("------");
    op += QubitOperator::from_label("IIIZZI");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(1);
    CHECK(op.ladder_integers() == std::vector<unsigned int>{MAX_UINT, 0, 1});
}


TEST_CASE("Test group term sort ladder_width=3 2") {
    QubitOperator op = QubitOperator::from_label("_+II");
    op += QubitOperator::from_label("++II");
    op += QubitOperator::from_label("IIIZ");
    op.set_type(2);
    op.group_term_sort_by_ladder_int(3);
    CHECK(op.ladder_integers() == std::vector<unsigned int>{MAX_UINT, 1, 3});
}



TEST_CASE("Verify that ladder integers are correct for terms in each group") {
    // group 1
    QubitOperator op = QubitOperator::from_label("-II+");  // int = 1
    op += QubitOperator::from_label("+II-");  // int = 2
    op += QubitOperator::from_label("+II+");  // int = 3
    // group 2
    op += QubitOperator::from_label("II+-");  // int = 2
    op += QubitOperator::from_label("II-+");  // int = 1
    // group 3
    op += QubitOperator::from_label("I++-");  // int = 6
    op += QubitOperator::from_label("I---");  // int = 0
    op += QubitOperator::from_label("I+++");  // int = 7
    op.set_type(2);
    op.group_sort();
    op.group_term_sort_by_ladder_int();
    CHECK(op.terms_by_group(1).ladder_integers() == std::vector<unsigned int>{1, 2, 3});
    CHECK(op.terms_by_group(2).ladder_integers() == std::vector<unsigned int>{1, 2});
    CHECK(op.terms_by_group(3).ladder_integers() == std::vector<unsigned int>{0, 6, 7});
}


TEST_CASE("Verify that ladder integers are correct for terms in each group 2") {
    // group 1
    QubitOperator op = QubitOperator::from_label("-II+");  // int = 1,
    op += QubitOperator::from_label("-ZZ+");  // int = 1,
    op += QubitOperator::from_label("+II-");  // int = 2,
    op += QubitOperator::from_label("-ZZ-");  // int = 0,
    op += QubitOperator::from_label("+0I-");  // int = 2,
    op += QubitOperator::from_label("+01+");  // int = 3,
    // group 2
    op += QubitOperator::from_label("II-+");  // int = 1
    op += QubitOperator::from_label("II++");  // int = 3
    op += QubitOperator::from_label("ZZ--");  // int = 0
    op += QubitOperator::from_label("ZZ+-");  // int = 2
    op += QubitOperator::from_label("Z0+-");  // int = 2
    op += QubitOperator::from_label("Z0-+");  // int = 1
    op += QubitOperator::from_label("ZI++");  // int = 3
    op.set_type(2);
    op.group_sort();
    op.group_term_sort_by_ladder_int();
    CHECK(op.terms_by_group(1).ladder_integers() == std::vector<unsigned int>{0, 1, 1, 2, 2, 3});
    CHECK(op.terms_by_group(2).ladder_integers() == std::vector<unsigned int>{0, 1, 1, 2, 2, 3, 3});
}