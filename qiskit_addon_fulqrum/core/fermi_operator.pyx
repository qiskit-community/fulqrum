# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair

from collections.abc import Iterable
import numbers
from qiskit_addon_fulqrum import __version__ as VERSION
from .qubit_operator cimport QubitOperator
from ..exceptions import FulqrumError

import orjson
import lzma
from pathlib import Path
import numpy as np
cimport numpy as np

include "includes/base_header.pxi"
include "includes/fermi_header.pxi"
include "includes/operators_header.pxi"
include "includes/converters.pxi"
include "includes/io.pxi"
include "includes/constants.pxi"


cdef const FermionicTerm_t EmptyFermionicTerm


cdef class FermionicOperator():
    """Operator class for Fermionic operators
    """
    def __cinit__(self, unsigned int num_qubits,
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
                            if inds[kk] > (self.oper.width - 1):
                                raise FulqrumError(f'Index {item[1]} is out of range for width={self.oper.width}')
                            if op_str[kk] != 73:
                                term.indices.push_back(inds[kk])
                                ind = STR_TO_IND[op_str[kk]]
                                term.values.push_back(ind)
                        term.coeff = coeff
                else:
                    term.coeff = 1
                insertion_sort_term(&term)
                self.oper.terms.push_back(term)

    
    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        (vector[FermionicTerm_t]()).swap(self.oper.terms)
    
    def __len__(self):
        """Number of terms in operator

        Returns:
            int
        """
        return self.oper.terms.size()

    def __iter__(self):
        self._iter_index = 0
        return self
    
    def __next__(self):
        cdef FermionicOperator out
        if self._iter_index < self.oper.terms.size():
            self._iter_index += 1
            out = FermionicOperator(self.oper.width)
            out.oper.terms.push_back(self.oper.terms[self._iter_index - 1])
            return out
        else:
            raise StopIteration

    @cython.boundscheck(False)
    def __add__(self, FermionicOperator other):
        """Addition of QubitOperators with copy

        Parameters:
            other (FermionicOperator): Other operator

        Returns:
            FermionicOperator
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

        Parameters:
            other (FermionicOperator): Other operator

        Returns:
            FermionicOperator
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

        Parameters:
            other (complex): Complex number

        Returns:
            FermionicOperator
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

        Parameters:
            other (complex): Complex number

        Returns:
            FermionicOperator
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

    @classmethod
    def from_label(self, size_t width, str label="", double complex coeff = 1.0):
        """Create FermionicOperator from a string label

        Parameters:
            width (int): Width of operator
            label (str): Label of operator
            coeff (complex): Complex coefficient, default=1.0

        Returns:
            FermionicOperator
        """
        cdef FermionicTerm_t term = EmptyFermionicTerm
        cdef FermionicOperator out = FermionicOperator(width)
        cdef list items = label.split(' ')
        cdef list temp
        if any(items):
            for item in items:
                temp = item.split(':')
                term.indices.push_back(<unsigned int>int(temp[1]))
                ind = STR_TO_IND[(<string>temp[0]).c_str()[0]]
                term.values.push_back(ind)
        term.coeff = coeff
        insertion_sort_term(&term)
        out.oper.terms.push_back(term)
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
        """Width of operator

        Returns:
            int
        """
        return self.oper.width

    @property
    def coeff(self):
        """Return the coeff for a single term or empty operator

        Returns:
            complex
        """
        cdef size_t kk, jj
        cdef list out = []
        if self.num_terms > 2:
            raise FulqrumError('Can only grab coeff from operators with < 2 terms')
        elif self.num_terms == 0:
            return 0+0j
        return self.oper.terms[0].coeff

    @cython.boundscheck(False)
    def coefficients(self):
        """Return the coefficients for each term in the operator

        Returns:
            ndarray: complex-valued array of coefficients
        """
        cdef size_t kk
        if self.oper.terms.size() == 0:
            raise FulqrumError('FermionicOperator has zero terms')
        cdef double complex[::1] out = np.empty(self.oper.terms.size(), dtype=complex)
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].coeff
        return np.asarray(out)


    @cython.boundscheck(False)
    def __repr__(self):
        cdef size_t idx 
        cdef list out = []
        cdef str temp_str
        cdef FermionicTerm_t term
        cdef size_t kk
        cdef size_t num_terms = self.oper.terms.size()
        cdef size_t total_terms = num_terms
        cdef int too_many_terms = 0
        if num_terms > 100:
            too_many_terms = 1
            num_terms = 100
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
        if too_many_terms:
            out_strs += f' + {total_terms-100} more terms'
        return f"<FermionicOperator[{out_strs}], width={self.oper.width}>"


    @property
    @cython.boundscheck(False)
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
        cdef unsigned int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.uint32)
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
    def deflate_repeated_indices(self):
        """Collapse repeated indices into singles and remove zero terms

        Returns:
            FermionicOperator: Deflated operator
        """
        cdef size_t kk
        cdef FermionicOperator out = FermionicOperator(self.width)
        for kk in range(self.oper.terms.size()):
            deflate_term_indicies(&self.oper.terms[kk], &out.oper.terms)
        return out
    
    def extended_jw_transformation(self):
        """Jordan-Wigner transformation over extended alphabet 
        from Fermionic -> Qubit operator
        """
        cdef FermionicOperator fermi = self.deflate_repeated_indices()
        cdef size_t num_terms = fermi.oper.terms.size()
        cdef QubitOperator out = QubitOperator(fermi.width)
        out.oper.terms.resize(num_terms)
        extended_jw_transform(fermi.oper, out.oper, num_terms)
        out.oper.type = 2
        return out.combine_repeated_terms()

    @cython.boundscheck(False)
    def to_dict(self):
        """Dictionary represenation of FermionicOperator
        
        Returns:
            dict: Dictionary representation of FermionicOperator
        """
        cdef dict out = {'operator-type': 'fermi',
                        'format-version': FORMAT_VERSION,
                        'fulqrum-version': VERSION,
                        'width': self.width
                        }
        cdef FermionicTerm_t * term
        cdef size_t kk, jj
        cdef list terms = []
        cdef list temp_inds
        cdef str temp_vals
        for kk in range(self.oper.terms.size()):
            term = &self.oper.terms[kk]
            temp_inds = []
            temp_vals = ''
            for jj in range(term.indices.size()):
                temp_inds.append(term.indices[jj])
                temp_vals += IND_TO_STR[term.values[jj]]
            terms.append([temp_vals, temp_inds, (term.coeff.real, term.coeff.imag)])
        out['terms'] = terms
        return out
    

    @classmethod
    def from_dict(self, dict dic):
        """FermionicOperator from dictionary

        Parameters:
            dic(dict): Dictionary representation of operator
        
        Returns:
            FermionicOperator
        """
        if dic['operator-type'] != 'fermi':
            raise FulqrumError("Dictionary operator-type is not 'fermi'")
        cdef size_t width = dic['width']
        cdef FermionicOperator out = FermionicOperator(width)
        for term in dic['terms']:
            out += FermionicOperator(width, [(term[0], term[1], complex(*term[2]))])
        return out
    

    def to_json(self, filename, overwrite=False):
        """Save operator to a JSON or XZ file. File extension can be 'json'
        or 'xz', the latter or which does LZMA compression which is
        recommended for large operators.

        Parameters:
            filename (str): File to store to
            overwrite (bool): Overwrite file if it exits, default=False
        """
        file = Path(filename)
        if file.is_file() and not overwrite:
            raise Exception("File already exists, set overwrite=True")
        file_type = filename.split('.')[-1].lower()
        if file_type == 'json':
            with open(filename, "wb") as fd:
                fd.write(orjson.dumps(self.to_dict()))
        elif file_type == 'xz':
            compressor = lzma.LZMACompressor()
            json_data = orjson.dumps(self.to_dict())
            lzma_data = compressor.compress(json_data) + compressor.flush()
            with open(filename, "wb") as fd:
                fd.write(lzma_data)
        else:
            raise FulqrumError("File type must be 'json' or 'xz'")


    @classmethod
    def from_json(self, filename):
        """Load operator from a JSON or XZ file.

        Parameters:
            filename (str): File to load from

        Returns:
            FermionicOperator
        """
        filename = str(filename)  # convert PosixPath
        file_type = filename.split('.')[-1].lower()
        if file_type == 'json':
            with open(filename, "r", encoding="utf-8") as fd:
                dic = orjson.loads(fd.read())
        elif file_type == 'xz':
            lzma_file = lzma.open(filename)
            data = lzma_file.readlines()
            dic = orjson.loads(data[0])
        else:
            raise FulqrumError("File type must be 'json' or 'xz'")
        out = FermionicOperator.from_dict(dic)
        return out
  


@cython.cdivision(True)
@cython.boundscheck(False)
cdef void insertion_sort_term(FermionicTerm_t * term):
    cdef size_t kk
    cdef int ll
    cdef size_t num_elems = term.indices.size()
    cdef unsigned int temp_index
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
            # Only add a minus sign if both operators (values)
            # are not projectors (ie. > 4 since '-'=5 and '+'=6)
            if (temp_value > 4) and (term.values[ll] > 4):
                prefactor *= -1
            ll -= 1
        term.indices[ll + 1] = temp_index
        term.values[ll + 1] = temp_value
    term.coeff *= prefactor


# DEFLATION ROUTINES

# These are the values returned when compressing two values over the same index
cdef int[16] COLLAPSED_VALUES = [1, -1, 5, -1, -1, 2, -1, 6, -1, 5, -1, 1, 6 , -1, 2, -1]


cdef int collapse_value(unsigned char x):
    """Converts a regular value index into a deflated one
    """
    if x == 1:
        return 0
    elif x == 2:
        return 1
    elif x == 5:
        return 2
    else: # x=6
        return 3


cdef void deflate_term_indicies(FermionicTerm_t * term, vector[FermionicTerm_t] * out_terms):
    cdef size_t num_elems = term.indices.size()
    cdef size_t kk 
    cdef size_t start, num_touched
    cdef FermionicTerm_t new_term = EmptyFermionicTerm
    cdef unsigned int current_index
    cdef int temp_int
    cdef unsigned char current_value

    num_touched = 0
    while(num_touched < num_elems):
        current_index = term.indices[num_touched]
        current_value = term.values[num_touched]
        num_touched += 1
        for kk in range(num_touched, num_elems):
            #next term has a matching index with the current one
            if term.indices[kk] == current_index:
                temp_int = COLLAPSED_VALUES[4*collapse_value(current_value) + collapse_value(term.values[kk])]
                # This operator becomes a null operator return
                if temp_int < 0:
                    return
                else:
                    current_value = <unsigned char>temp_int
                num_touched += 1
            else:
                # Move on to next index since not matching and we assume we index sorted already
                break
        new_term.indices.push_back(current_index)
        new_term.values.push_back(current_value)
    new_term.coeff = term.coeff
    out_terms.push_back(new_term)
