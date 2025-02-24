/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <algorithm>


/**
 * Sorting of indices and values for Operator term data
 *
 * @param inds The term indices (qubits) array
 * @param vals The term values (operators) array
 */
void sort_term_data(std::vector<std::size_t>& inds, std::vector<unsigned char>& vals) {
    std::size_t n = inds.size();
    for (std::size_t i = 1; i < n; i++) {
        std::size_t key = inds[i];
        char val = vals[i];
        std::size_t j = std::lower_bound(inds.begin(), inds.begin() + i, key) - inds.begin();
        
        for (std::size_t k = i; k > j; k--) {
            inds[k] = inds[k-1];
            vals[k] = vals[k-1];
        }
        inds[j] = key;
        vals[j] = val;
    }
}

