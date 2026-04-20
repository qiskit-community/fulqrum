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

/**
 * Test Extended Jordan-Wigner transformation
 *
 */
TEST_CASE("Test JW does not crash on empty operator")
{
    FermionicOperator_t fop = FermionicOperator(5);
    QubitOperator_t op = fop.extended_jw_transformation();
    CHECK(op.size() == 0);
}

TEST_CASE("Test JW simple 1")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"+", {0}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"+", {0}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 2")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"-", {0}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"-", {0}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 3")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"0", {0}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"0", {0}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 4")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"1", {0}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"1", {0}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 5")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"+", {1}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"Z+", {0, 1}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 6")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"-", {1}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"Z-", {0, 1}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 7")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"0", {1}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"0", {1}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW simple 8")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"1", {1}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"1", {1}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW medium 1")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"Z-+", {0, 1, 4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"Z-ZZ+", {0, 1, 2, 3, 4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(-1, 0)); // sign change in coeff
}

TEST_CASE("Test JW medium 2")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"--", {1, 4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"-ZZ-", {1, 2, 3, 4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(-1, 0)); // sign change in coeff
}

TEST_CASE("Test JW medium 3")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"0+", {2, 4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"0Z+", {2, 3, 4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW medium 4")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"+", {4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"ZZZZ+", {0, 1, 2, 3, 4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW medium 5")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"-", {4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"ZZZZ-", {0, 1, 2, 3, 4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}

TEST_CASE("Test JW medium 6")
{
    FermionicOperator_t fop = FermionicOperator(5, {{"1", {4}, 1}});
    QubitOperator_t op = fop.extended_jw_transformation();
    QubitOperator_t ans = QubitOperator(5, {{"1", {4}, 1}});
    CHECK(op[0].operators() == ans[0].operators());
    CHECK(op[0].coeff == complex(1, 0));
}
