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


TEST_CASE("Test term grouping 1") {
    QubitOperator op = QubitOperator(2);
    std::vector<std::string> labels = {"XY", "XI", "IY", "YY", "IZ", "II", "Z0"};
    for(auto label: labels)
    {
        op += QubitOperator::from_label(label);
    }
    op.group_sort();
    std::vector<int> ans = {0, 0, 0, 1, 2, 3, 3};
    for(unsigned int kk=0; kk < op.size(); kk++)
    {
        CHECK(op[kk].group == ans[kk]);
    }
}


TEST_CASE("Test term grouping 2") {
    QubitOperator op = QubitOperator(3);
    std::vector<std::string> labels = {"III", "ZZ1", "Z0Z", "IZI", "ZI0"};
    for(auto label: labels)
    {
        op += QubitOperator::from_label(label);
    }
    op.group_sort();
    std::vector<int> ans(5, 0);
    for(unsigned int kk=0; kk < op.size(); kk++)
    {
        CHECK(op[kk].group == ans[kk]);
    }
}


TEST_CASE("Test term grouping 3") {
    QubitOperator op = QubitOperator(4);
    std::vector<std::string> labels = {"XIIY", "+ZZ-", "Y01X", "-00+"};
    for(auto label: labels)
    {
        op += QubitOperator::from_label(label);
    }
    op.group_sort();
    std::vector<int> ans(4, 1);
    for(unsigned int kk=0; kk < op.size(); kk++)
    {
        CHECK(op[kk].group == ans[kk]);
    }
}


TEST_CASE("Test split operator grouping") {
    QubitOperator op = QubitOperator(4);
    std::vector<std::string> labels = {"XIII", "ZYII", "ZIZI", "01+Z", "IIII", "+Z00"};
    for(auto label: labels)
    {
        op += QubitOperator::from_label(label);
    }
    op.group_sort();
    auto [diag, off] = op.split_diagonal();

    std::vector<int> diag_ans = {0, 0};
    std::vector<int> off_ans = {1, 1, 2, 3};
    CHECK(diag.groups() == diag_ans);
    CHECK(off.groups() == off_ans);
}
