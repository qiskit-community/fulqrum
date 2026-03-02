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
#include <complex>
#include <cstdlib>

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
