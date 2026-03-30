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
#include "fulqrum.hpp"
#include <fulqrum.hpp>

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