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


TEST_CASE("Test simple operator with one term") {
    FermionicOperator_t op = FermionicOperator(5, {{"XXXXX", {0,1,2,3,4}, 1}});
    std::vector<OpData> ans = {OpData("X", 0), OpData("X", 1), OpData("X", 2), OpData("X", 3), OpData("X", 4)};
    CHECK(op[0].operators() == ans);
}
