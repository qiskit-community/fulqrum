# Fulqrum
# Copyright (C) 2024, IBM


cdef extern from "../src/csr_utils.hpp":

    void quicksort_indices_data[T,U](T * indices, U * data, T start, T stop) nogil

