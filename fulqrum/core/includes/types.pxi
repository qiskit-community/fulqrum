# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

ctypedef long long int64

ctypedef fused int32_or_int64:
    int
    long long

ctypedef fused double_or_complex:
    double
    double complex