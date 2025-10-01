/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstring>
#include <vector>
#include <complex>


// Data types for CSR-like matrix structure

typedef struct RowData_Real64{
    std::vector<std::vector<long long> > cols;
    std::vector<std::vector<double> > data;
} RowData_Real64_t;


typedef struct RowData_Real32{
    std::vector<std::vector<int> > cols;
    std::vector<std::vector<double> > data;
} RowData_Real32_t;


typedef struct RowData_Complex64{
    std::vector<std::vector<long long> > cols;
    std::vector<std::vector<std::complex<double> > > data;
} RowData_Complex64_t;


typedef struct RowData_Complex32{
    std::vector<std::vector<int> > cols;
    std::vector<std::vector<std::complex<double> > > data;
} RowData_Complex32_t;


/**
 * Set the row-pointers array data from the data for a CSRLike struct
 *
 * @param row_data The data for each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 * 
 */
template <typename T>
void set_csr_ptr(const std::vector<std::vector<T>>& cols, T * __restrict ptrs)
{
    std::size_t num_rows = cols.size();
    std::size_t kk;
    std::size_t temp, _sum = 0;
    for(kk=0; kk < num_rows; kk++)
    {
        ptrs[kk] = cols[kk].size();
        temp = _sum + ptrs[kk];
        ptrs[kk] = _sum;
        _sum = temp;
    }
    ptrs[num_rows] = _sum;
}


/**
 * Set the indices and data arrays for a CSR matrix from a CSRLike struct
 *
 * @param in_data The data for each row in the matrix
 * @param in_data The columns for the data in each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 * @param inds A pointer to the indices array for a CSR matrix of length nnz
 * @param out_data A pointer to the output data array for a CSR matrix of length nnz
 * 
 */
template <typename T, typename U>
void set_csr_data(const std::vector<std::vector<T> >& in_data, const std::vector<std::vector<U> >& cols, 
                  U * __restrict ptrs, U * __restrict inds, T * __restrict out_data)
{
    std::size_t num_rows = in_data.size();
    std::size_t kk;
    #pragma parallel omp for schedule(dynamic)
    for(kk=0; kk < num_rows; kk++)
    {
        U start, stop;
        start = ptrs[kk];
        stop = ptrs[kk+1];
        std::memcpy(&inds[start], cols[kk].data(), (stop-start)*sizeof(U));
        std::memcpy(&out_data[start], in_data[kk].data(), (stop-start)*sizeof(T));
    }
}



template <typename T, typename U>
void csrlike_spmv(const std::vector<std::vector<T>>& data, const std::vector<std::vector<U>>& cols, 
                   const T *__restrict vec, T *__restrict out, U dim)
{
    U row;
    #pragma omp parallel for if(dim > 128) schedule(dynamic)
    for (row = 0; row < dim; row++)
    {
        T dot = 0.0;
        std::size_t jj, row_end;
        const T * row_data = data[row].data();
        const U * row_cols = cols[row].data();
        row_end = data[row].size();
        for (jj = 0; jj < row_end; jj++)
        {
            dot += row_data[jj] * vec[row_cols[jj]];
        }
        out[row] += dot;
    }
}
