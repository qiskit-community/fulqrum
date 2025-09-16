/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>


// Data types for CSR-like matrix structure

typedef struct RowData_Real64{
    std::vector<std::size_t> cols;
    std::vector<double> data;
} RowData_Real64_t;


typedef struct RowData_Real32{
    std::vector<unsigned int> cols;
    std::vector<double> data;
} RowData_Real32_t;


typedef struct RowData_Complex64{
    std::vector<std::size_t> cols;
    std::vector<std::complex<double> > data;
} RowData_Complex64_t;


typedef struct RowData_Complex32{
    std::vector<unsigned int> cols;
    std::vector<std::complex<double> > data;
} RowData_Complex32_t;
