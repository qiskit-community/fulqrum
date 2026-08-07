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
#pragma once
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <complex>
#include <cstdlib>
#include <utility>
#include <vector>

template <typename T, typename U>
T partition(T* __restrict indices, U* __restrict data, T start, T stop)
{
    // rightmost element as pivot
    T pivot = indices[stop];

    T temp_inds, jj, ii = start - 1;
    U temp_data;

    for(jj = start; jj < stop; jj++)
    {
        if(indices[jj] <= pivot)
        {
            ii = ii + 1;
            temp_inds = indices[ii];
            temp_data = data[ii];

            indices[ii] = indices[jj];
            indices[jj] = temp_inds;
            data[ii] = data[jj];
            data[jj] = temp_data;
        }
    }

    temp_inds = indices[ii + 1];
    temp_data = data[ii + 1];

    indices[ii + 1] = indices[stop];
    indices[stop] = temp_inds;
    data[ii + 1] = data[stop];
    data[stop] = temp_data;
    return ii + 1;
}

template <typename T, typename U>
void quicksort_indices_data(T* __restrict indices, U* __restrict data, T start, T stop)
{
    T pi;
    if(start < stop)
    {
        pi = partition(indices, data, start, stop);
        quicksort_indices_data(indices, data, start, pi - 1);
        quicksort_indices_data(indices, data, pi + 1, stop);
    }
}

template <typename T, typename U>
void sort_paired(std::vector<std::vector<T>>& cols, std::vector<std::vector<U>>& data)
{
#pragma omp parallel
    {
        std::vector<std::pair<T, U>> tmp;

#pragma omp for schedule(dynamic)
        for(std::size_t kk = 0; kk < cols.size(); kk++)
        {
            auto& row1 = cols[kk];
            auto& row2 = data[kk];
            std::size_t jj;
            const std::size_t num_elems = row1.size();

            tmp.resize(num_elems);
            for(jj = 0; jj < num_elems; jj++)
                tmp[jj] = {row1[jj], row2[jj]};

            boost::sort::pdqsort(
                tmp.begin(), tmp.end(), [](const std::pair<T, U>& a, const std::pair<T, U>& b) {
                    return a.first < b.first;
                });

            for(jj = 0; jj < num_elems; jj++)
            {
                row1[jj] = std::move(tmp[jj].first);
                row2[jj] = std::move(tmp[jj].second);
            }
        }
    }
}
