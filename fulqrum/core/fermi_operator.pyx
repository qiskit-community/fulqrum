# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

from collections.abc import Iterable
import numbers
from fulqrum.exceptions import FulqrumError

import numpy as np
cimport numpy as np

include "includes/base_header.pxi"
include "includes/converters.pxi"

cdef const FermionicTerm_t EmptyFermionicTerm


cdef class FermionicOperator():
    """Operator class for Fermionic operators
    """
    def __cinit__(self, size_t num_qubits,
                  object operators=None):
        self.oper.width = num_qubits
        cdef object item
        cdef FermionicTerm_t term
        cdef string op_str
        cdef double complex coeff
        cdef object inds
        cdef size_t kk
        cdef char op, ind
        if operators is not None:
            for item in operators:
                term = EmptyFermionicTerm
                if any(item):
                    if len(item) == 1:
                        term.coeff = item[0]
                    else:
                        op_str = (<string>item[0]).c_str()
                        inds = item[1] if isinstance(item[1], Iterable) else [item[1]] 
                        coeff = item[2] if len(item) == 3 else 1.0
                        for kk in range(<size_t>len(item[0])):
                            if inds[kk] > <size_t>(self.oper.width - 1):
                                raise FulqrumError(f'Index {item[1]} is out of range for width={self.oper.width}')
                            if op_str[kk] != 73:
                                term.indices.push_back(inds[kk])
                                ind = STR_TO_IND[op_str[kk]]
                                term.values.push_back(ind)
                        term.coeff = coeff
                else:
                    term.coeff = 1
                self.oper.terms.push_back(term)

    
    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        self.oper.terms = vector[FermionicTerm_t]()
    
    def __len__(self):
        return self.oper.terms.size()

    @cython.boundscheck(False)
    def __add__(self, FermionicOperator other):
        """Addition of QubitOperators withcopy
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            out.oper.terms.push_back(other.oper.terms[kk])
        return out

    def __iadd__(self, FermionicOperator other):
        """Inplace addition of FermionicOperators
        """
        self.append(other)
        return self

    @cython.boundscheck(False)
    def __sub__(self, FermionicOperator other):
        """Subtraction of QubitOperators with copy
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.oper.width)
        cdef FermionicTerm_t term
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            term = other.oper.terms[kk]
            term.coeff *= -1
            out.oper.terms.push_back(term)
        return out

    @cython.boundscheck(False)
    def __mul__(self, double complex other):
        """Multiplication of FermionicOperators on left with copy
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __rmul__(self, double complex other):
        """Multiplication of FermionicOperators on right with copy
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, key):
        """Impliments indexing and slicing of terms

        Parameters:
            key (integral or slice or list or tuple): Indices

        Returns:
            FermionicOperator: Indexed terms
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.oper.width)
        if isinstance(key, numbers.Integral):
            if key < 0:
                 key = self.oper.terms.size() + key 
            kk = <size_t>key
            if kk > self.oper.terms.size() - 1:
                raise FulqrumError(f"Index {kk} is out of range for operator"
                                   f" of length {self.oper.terms.size()}")
            out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, slice):
            for kk in range(*key.indices(self.oper.terms.size())):
                out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, (list, tuple)):
            for idx in key:
                kk = <size_t>idx
                out.oper.terms.push_back(self.oper.terms[kk])
        else:
            raise FulqrumError(f"Cannot get operator terms using {type(key)}")
        return out

    @property
    def num_terms(self):
        """Return the number of terms in the operator

        Returns:
            int: Number of terms in operator
        """
        return self.oper.terms.size()
    
    @property
    def width(self):
        return self.oper.width

    @property
    def coeff(self):
        """Return the coeff for a single term or empty operator
        """
        cdef size_t kk, jj
        cdef FermionicTerm_t * term
        cdef list out = []
        if self.num_terms > 2:
            raise FulqrumError('Can only grab coeff from operators with < 2 terms')
        elif self.num_terms == 0:
            return 0+0j
        return self.oper.terms[0].coeff


    @cython.boundscheck(False)
    def __repr__(self):
        cdef size_t idx 
        cdef list out = []
        cdef str temp_str
        cdef FermionicTerm_t term
        cdef size_t kk
        cdef size_t num_terms = self.oper.terms.size()
        for idx in range(num_terms):
            temp_str = ''
            term = self.oper.terms[idx]
            for kk in range(term.indices.size()):
                if kk:
                    temp_str += ' '
                temp_str += IND_TO_STR[term.values[kk]] + ':'
                temp_str += str(term.indices[kk])
            out.append((temp_str, term.coeff))

        out_strs = ', '.join(str(kk) for kk in out)
        return f"<FermionicOperator[{out_strs}], width={self.oper.width}>"


    @property
    def operators(self):
        """Return the operators for a single term or empty operator

        Returns:
            list or None : List of operator index tuples, if any, else None

        Notes:
            This returns a list of tuples to allow for multiple indices in
            Fermionic operators
        """
        cdef size_t kk, jj
        cdef FermionicTerm_t * term
        cdef list out = []
        if self.num_terms > 1:
            raise FulqrumError('Can only grab operators from operators with < 2 terms')
        elif self.num_terms == 0:
            return None
        else:
            for kk in range(self.oper.terms.size()):
                term = &self.oper.terms[kk]
                for jj in range(term.indices.size()):
                    out.append((IND_TO_STR[term.values[jj]], term.indices[jj]))
            return out

    @cython.boundscheck(False)
    def weights(self):
        """Weight of each term in the operator

        Returns:
            ndarray: Array of operator weights
        """
        cdef size_t[::1] out = np.zeros(self.oper.terms.size(), dtype=np.uintp)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].values.size()
        return np.asarray(out)


    @cython.boundscheck(False)
    cpdef void append(self, FermionicOperator other):
        cdef size_t kk
        if self.oper.width != other.oper.width:
            raise FulqrumError('Appending number of qubits does not match current number')
        for kk in range(other.oper.terms.size()):
            self.oper.terms.push_back(other.oper.terms[kk])


    @cython.boundscheck(False)
    def index_ordering(self):
        """Inplace ordering of term elements by index.
        """
        cdef size_t kk, ll
        for kk in range(self.oper.terms.size()):
            insertion_sort_term(&self.oper.terms[kk])
         


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void insertion_sort_term(FermionicTerm_t * term):
    cdef size_t kk
    cdef int ll
    cdef size_t num_elems = term.indices.size()
    cdef size_t temp_index
    cdef unsigned char temp_value
    cdef int prefactor = 1
    for kk in range(1, num_elems):
        temp_index = term.indices[kk]
        temp_value = term.values[kk]
        ll = kk - 1
        # Only switch elements if they are of different indices
        # In this case we always pick up a minus sign that
        # we need to keep track of with the 'prefactor'
        while ll >= 0 and temp_index < term.indices[ll]:
            term.indices[ll + 1] = term.indices[ll]
            term.values[ll + 1] = term.values[ll]
            prefactor *= -1
            ll -= 1
        term.indices[ll + 1] = temp_index
        term.values[ll + 1] = temp_value
    term.coeff *= prefactor
