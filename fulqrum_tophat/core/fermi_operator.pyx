# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from fulqrum_tophat.exceptions import FulqrumError
from fulqrum_tophat.core.base cimport OperatorTerm, size_uchar_pair, diagonal_term

import numbers
import numpy as np
cimport numpy as np

ctypedef pair[uchar, bool] uchar_bool_pair

cdef unordered_map[string, size_t] STR_TO_IND = {'-': 0, '+': 1, '0': 2, '1': 3}

cdef unordered_map[size_t, string] IND_TO_STR = {0: '-', 1: '+', 2: '0', 3: '1'}

cdef uchar_bool_pair[16] COLLECTOR_VEC = [uchar_bool_pair(0, False), # - - null
                                          uchar_bool_pair(2, True),  # - +
                                          uchar_bool_pair(0, False), # - 0 null
                                          uchar_bool_pair(0, True), # - 1
                                          uchar_bool_pair(3, True), # + -
                                          uchar_bool_pair(1, False), # + + null
                                          uchar_bool_pair(1, True), # + 0
                                          uchar_bool_pair(1, False), # + 1 null
                                          uchar_bool_pair(0, True), # 0 -
                                          uchar_bool_pair(2, False), # 0 + null
                                          uchar_bool_pair(2, True), # 0 0
                                          uchar_bool_pair(2, False), # 0 1 null
                                          uchar_bool_pair(3, False), # 1 - null
                                          uchar_bool_pair(1, True), # 1 +
                                          uchar_bool_pair(3, False), # 1 0 null
                                          uchar_bool_pair(3, True), # 1 1
                                         ]


cdef const OperatorTerm EmptyOperatorTerm


cdef class FermionicOperator():
    """Operator class for Fermionic operators
    """
    def __cinit__(self, size_t width,
                  object operators=None,
                  double complex coeff=1.0):
        self.width = width
        self.num_orbitals = width
        self.sorted = False
        cdef object item
        cdef OperatorTerm term
        if operators is not None:
            term = EmptyOperatorTerm
            term.coeff = coeff
            for item in operators:
                if item[1] > self.width-1:
                    raise FulqrumError(f'Index {item[1]} is out of range for width={self.width}')
                term.operators.push_back(size_uchar_pair(item[1], STR_TO_IND[item[0]]))
            self.terms.push_back(term)

    def __len__(self):
        return self.terms.size()

    def __getitem__(self, key):
        """Impliments indexing and slicing of terms

        Parameters:
            key (integral or slice or list or tuple): Indices

        Returns:
            QubitOperator: Indexed terms
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        if isinstance(key, numbers.Integral):
            kk = <size_t>key
            if kk > self.terms.size() - 1:
                raise FulqrumError(f"Index {kk} is out of range for operator"
                                   f" of length {self.terms.size()}")
            out.terms.push_back(self.terms[kk])
        elif isinstance(key, slice):
            for kk in range(*key.indices(self.terms.size())):
                out.terms.push_back(self.terms[kk])
        elif isinstance(key, (list, tuple)):
            for idx in key:
                kk = <size_t>idx
                out.terms.push_back(self.terms[kk])
        else:
            raise FulqrumError(f"Cannot get operator terms using {type(key)}")
        return out
    
    def __iadd__(self, FermionicOperator other):
        """Inplace addition of FermionicOperators
        """
        self.append(other)
        return self

    def __add__(self, FermionicOperator other):
        """Addition of FermionicOperators with copy
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        out.terms = self.terms
        for kk in range(other.terms.size()):
            out.terms.push_back(other.terms[kk])
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void append(self, FermionicOperator other):
        cdef size_t kk
        if self.width != other.width:
            raise FulqrumError('Appending operator width does not match current width')
        for kk in range(other.terms.size()):
            self.terms.push_back(other.terms[kk])

    @property
    def coeff(self):
        """Return the coeff for a single term or empty operator
        """
        cdef size_t kk, jj
        cdef OperatorTerm * term
        cdef list out = []
        if self.num_terms > 2:
            raise FulqrumError('Can only grab coeff from operators with < 2 terms')
        elif self.num_terms == 0:
            return 0+0j
        return self.terms[0].coeff

    @cython.boundscheck(False)
    def weight(self):
        """Weight of each term in the operator

        Returns:
            ndarray: Array of operator weights
        """
        cdef size_t[::1] out = np.zeros(self.terms.size(), dtype=np.uint64)
        cdef size_t kk
        for kk in range(self.terms.size()):
            out[kk] = self.terms[kk].operators.size()
        return np.asarray(out)

    @property
    def operators(self):
        """Return the operators for a single term or empty operator

        Returns:
            list or None : List of operator index tuples, if any, else None
        """
        cdef size_t kk, jj
        cdef OperatorTerm * term
        cdef list out = []
        if self.num_terms > 1:
            raise FulqrumError('Can only grab operators from operators with < 2 terms')
        elif self.num_terms == 0:
            return None
        else:
            for kk in range(self.terms.size()):
                term = &self.terms[kk]
                for jj in range(term.operators.size()):
                    out.append((IND_TO_STR[term.operators[jj].second], 
                                term.operators[jj].first))
            return out

    @property
    def num_terms(self):
        """Return the number of terms in the operator

        Returns:
            int: Number of terms in operator
        """
        return self.terms.size()

    @cython.boundscheck(False)
    def __repr__(self):
        cdef unsigned int idx 
        cdef list out = []
        cdef str temp_str
        cdef unsigned char temp_char
        cdef OperatorTerm term
        cdef size_t kk
        for idx in range(self.terms.size()):
            temp_str = ''
            term = self.terms[idx]
            for kk in range(term.operators.size()):
                if kk:
                    temp_str += ' '
                temp_str += IND_TO_STR[term.operators[kk].second]
                temp_str += str(term.operators[kk].first)
            out.append([temp_str, term.coeff])

        out_strs = ', '.join(str(kk) for kk in out)
        return f"FermionicOperator<[{out_strs}], width={self.width}>"

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normal_ordering(self):
        """Put into normal ordering

        Returns:
            FermionicOperator: Operator in normal ordering
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        for kk in range(self.terms.size()):
            normal_order_term(self.terms[kk], &out.terms)
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def index_ordering(self):
        """Put into index ordering

        Returns:
            FermionicOperator: Operator in index ordering
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        for kk in range(self.terms.size()):
            index_order_term(self.terms[kk], &out.terms)
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def combine_repeated_indices(self):
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        for kk in range(self.terms.size()):
            single_term_index_combine(self.terms[kk], &out.terms)
        return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void index_order_term(OperatorTerm term, vector[OperatorTerm] * out_terms):
    """Index order a single OperatorTerm

    Parameters:
        term (OperatorTerm): Term to be ordered
        out_terms (vector[OperatorTerm] *): Pointer to output vector of terms
    """
    cdef size_t kk, term_idx
    cdef int mm, temp_idx
    cdef bool attach_term = True
    cdef size_uchar_pair temp_pair
    cdef OperatorTerm new_term
    # Do insertion sorting of each term 
    for term_idx in range(1, term.operators.size()):
        temp_pair = term.operators[term_idx]
        mm = term_idx - 1
        while mm >= 0 and attach_term:
            # If index on left is bigger then flip them
            if (term.operators[mm].first > temp_pair.first):
                term.operators[mm+1] = term.operators[mm]
                term.coeff *= -1
                # Reduce the iter index
                mm -= 1
            # Both operators are on same index
            elif (term.operators[mm].first == temp_pair.first):
                # If operators are the same then the term is zero
                if (term.operators[mm].second == temp_pair.second):
                    attach_term = False
                break
            else:
                break
        term.operators[mm+1] = temp_pair
    if attach_term:
        out_terms.push_back(term)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void normal_order_term(OperatorTerm term, vector[OperatorTerm] * out_terms):
    """Normal order a single OperatorTerm

    Parameters:
        term (OperatorTerm): Term to be ordered
        out_terms (vector[OperatorTerm] *): Pointer to output vector of terms
    """
    cdef size_t kk, term_idx, temp_idx
    cdef int mm
    cdef bool attach_term = True
    cdef size_uchar_pair temp_pair
    cdef OperatorTerm new_term
    # Do insertion sorting of each term 
    for term_idx in range(1, term.operators.size()):
        temp_pair = term.operators[term_idx]
        mm = term_idx - 1
        while mm >= 0 and attach_term:
            # If rasing op on right move it
            if (term.operators[mm].second < temp_pair.second):
                term.operators[mm+1] = term.operators[mm]
                term.coeff *= -1
                # Operator indices are the same then need to add a term
                if (term.operators[mm].first == temp_pair.first):
                    new_term = EmptyOperatorTerm
                    # This term has no sign flip from initial operator
                    # so need to flip the flip that we made (-1 * -1 = 1)
                    new_term.coeff = -1*term.coeff
                    for temp_idx in range(0, <size_t>mm):
                        new_term.operators.push_back(term.operators[temp_idx])
                    for temp_idx in range(<size_t>(mm+2), term.operators.size()):
                        new_term.operators.push_back(term.operators[temp_idx])
                    normal_order_term(new_term, out_terms)
                # Reduce the iter index
                mm -= 1
            # Both operators are the same kind
            elif (term.operators[mm].second == temp_pair.second):
                # If indices are the same then the operator is a zero operator
                if (term.operators[mm].first == temp_pair.first):
                    attach_term = False
                # If indices are the different, then bring smaller index to the left
                elif (term.operators[mm].first > temp_pair.first):
                    term.operators[mm+1] = term.operators[mm]
                    term.coeff *= -1
                    # Reduce the iter index
                    mm -= 1
                    continue
                break
            else:
                break
        term.operators[mm+1] = temp_pair
    if attach_term:
        out_terms.push_back(term)


@cython.boundscheck(False)
cdef void single_term_index_combine(OperatorTerm term, vector[OperatorTerm] * out_terms) noexcept:
    cdef size_t kk, jj
    cdef bool append_term
    cdef uchar_bool_pair new_op_pair
    append_term = True
    # Look only if more than one operator in the term
    if term.operators.size() > 1:
        for jj in range(term.operators.size()-1):
            # If terms operate on same index then combine
            if term.operators[jj].first == term.operators[jj+1].first:
                new_op_pair = COLLECTOR_VEC[4*term.operators[jj].second + term.operators[jj+1].second]
                # The result is not a NULL term we need to process
                if new_op_pair.second:
                    new_term = EmptyOperatorTerm
                    new_term.coeff = term.coeff
                    for mm in range(jj):
                        new_term.operators.push_back(term.operators[mm])
                    # Add new combined term
                    new_term.operators.push_back(size_uchar_pair(term.operators[jj].first,
                                                 new_op_pair.first))
                    for mm in range(jj+2, term.operators.size()):
                        new_term.operators.push_back(term.operators[mm])
                    # Run it through routine again looking for more repeat indices
                    single_term_index_combine(new_term, out_terms)
                    append_term = False
                    break
                # We have a null op, so terminate
                else:
                    append_term = False
                    break
    if append_term:
        out_terms.push_back(term)
