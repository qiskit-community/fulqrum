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
#include "bitset_hashmap.hpp"
#include "constants.hpp"
#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <ostream>



// Map converting standard char values into continuous values
inline std::unordered_map<unsigned char, unsigned char> oper_map =
    {
        {90, 0}, {48, 1}, {49, 2}, {88, 3}, {89, 4}, {45, 5}, {43, 6}
    };

// Reverse map back to standard char values
inline std::unordered_map<unsigned char, unsigned char> rev_oper_map =
    {
        {0, 90}, {1, 48}, {2, 49}, {3, 88}, {4, 89}, {5, 45}, {6, 43}
    };



typedef std::tuple<std::string, std::vector<unsigned int>, std::complex<double>> TermData;
typedef std::tuple<std::string, std::vector<unsigned int>> OpData;


/** @brief Data structure for each operator term, i.e. 'word' in the operator
 *
 * @var indices the qubits (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 * @var offdiag_weight is the number of non-diagonal operators in the term
 */
typedef struct OperatorTerm
{
    std::vector<unsigned char> values;
    std::vector<unsigned int> indices;
    std::complex<double> coeff;
    std::vector<unsigned int> proj_indices;
    std::vector<unsigned int> proj_bits;
    unsigned int offdiag_weight{0};
    int extended{0};
    int real_phase{1}; // 'phase' of real part (+/- 1), 0 means operator is complex-valued
    int group{-1}; // -1 means unset here

    OperatorTerm() {}
    OperatorTerm(std::complex<double> c): coeff(c) {} // Init empty term with given coefficient
    OperatorTerm(std::string vals, std::vector<unsigned int> inds, std::complex<double> c): coeff(c)
    {
        //check that length of values == length of indices
        if(vals.size() != inds.size())
        {
            throw std::runtime_error("Size of input string does not equal that of indices");
        }
        unsigned char val;
        unsigned int counter = 0;
        // Iterate over string of values, mapping to new values and adding to term
        for(std::string::iterator it = vals.begin(); it != vals.end(); ++it)
        {
            counter += 1;
            if(*it == 73) // identity operator
            {
                continue;
            }
            else{
                val = oper_map[*it];
                values.push_back(val);
                indices.push_back(inds[counter-1]);
                this->offdiag_weight += (val > 2);
            }
        }
        //check that length of values == length of indices
        if(values.size() != indices.size())
        {
            throw std::runtime_error("Size of values vector does not equal that of indices.");
        }
        sort_term_data(); // sort term data from low -> high indices
        set_proj_indices(); // set projection operator indices, if any
    }
    // destructor
    ~OperatorTerm()
    {
        std::vector<unsigned char>().swap(values);
        std::vector<unsigned int>().swap(indices);
        std::vector<unsigned int>().swap(proj_indices);
        std::vector<unsigned int>().swap(proj_bits);
    }
    /**
     * Inplace multiplication by a complex value
     */
    OperatorTerm& operator*=(std::complex<double> c)
    {
        coeff *= c;
        return *this;
    }
    OperatorTerm copy() const
    {
        OperatorTerm out = OperatorTerm(this->coeff);
        out.values = this->values;
        out.indices = this->indices;
        out.proj_indices = this->proj_indices;
        out.proj_bits = this->proj_bits;
        out.offdiag_weight = this->offdiag_weight;
        return out;
    }
    /**
     * Term multiplication by a complex number
     */
    friend OperatorTerm operator*(OperatorTerm& op, std::complex<double> c)
    {
        OperatorTerm out = op.copy();
        out.coeff *= c;
        return out;
    }
    /**
     * Term multiplication by a complex number
     */
    friend OperatorTerm operator*(std::complex<double> c, OperatorTerm& op)
    {
        OperatorTerm out = op.copy();
        out.coeff *= c;
        return out;
    }
    /**
     * Return the size of the term
     */
    std::size_t size() const {return indices.size();}
    /**
     * Return the weight (num. non-identity) operators
     * 
     * @param[out] weight The weight of the term
     */
    unsigned int weight() const
    {
        return static_cast<unsigned int>(indices.size());
    }
    /**
     * Sorting of indices and values for Operator term data
     */
    OperatorTerm& sort_term_data()
    {
        std::size_t n = indices.size();
        for(std::size_t i = 1; i < n; i++)
        {
            unsigned int key = indices[i];
            char val = values[i];
            std::size_t j = std::lower_bound(indices.begin(), indices.begin() + i, key) - indices.begin();

            for(std::size_t k = i; k > j; k--)
            {
                indices[k] = indices[k - 1];
                values[k] = values[k - 1];
            }
            indices[j] = key;
            values[j] = val;
        }
        return *this;
    }
    /**
     * Set the projector indices and bits for term
     */
    OperatorTerm& set_proj_indices()
    {
        std::size_t kk;
        unsigned int val;
        proj_indices.resize(0);
        proj_bits.resize(0);
        for(kk = 0; kk < values.size(); kk++)
        {
            val = values[kk];
            if(val == 1 || val == 2)
            {
                proj_indices.push_back(indices[kk]);
                proj_bits.push_back(val - 1);
            }
        }
        return *this;
    }
    std::vector<OpData> operators() const
    {
        std::vector<OpData> out;
        for(std::size_t kk=0; kk < indices.size(); kk++)
        {
            OpData item{std::string(1, static_cast<char>(rev_oper_map[values[kk]])), indices[kk]};
            out.push_back(item);
        }
        return out;
    }
    /**
     * Is the term diagonal
     */
    bool is_diagonal() const
    {
        std::size_t kk;
        bool diag = 1;
        for(kk=0; kk < values.size(); kk++)
        {
            if(values[kk] > 2)
            {
                diag = 0;
                break;
            }
        }
        return diag;
    }
} OperatorTerm_t;


/**
 * Validate that term indices are less than operator width
 *
 * @param[in] indices Indices for the given term
 * @param[in] width The operator width
 */
inline void _validate_indices(std::vector<unsigned int>& inds, unsigned int width){
    std::size_t size = inds.size();
    for(std::size_t kk=0; kk < size; kk++)
    {
        if(inds[kk] >= width)
        {
            throw std::runtime_error("Index is larger than the operator width.");
        }
    }
}


/**
 * Comparator for weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int weight_comp(OperatorTerm& term1, OperatorTerm& term2)
{
    return term1.indices.size() < term2.indices.size();
}

/**
 * Comparator for off-diagonal weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offweight_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.offdiag_weight < term2.offdiag_weight;
}


inline void set_weight_ptrs(std::vector<OperatorTerm>& __restrict terms, std::vector<std::size_t>& vec)
{
    vec.resize(0);
    vec.push_back(0);
    std::size_t kk;
    unsigned int val = terms[0].indices.size();
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].indices.size() > val)
        {
            vec.push_back(kk);
            val = terms[kk].indices.size();
        }
    }
    vec.push_back(terms.size());
}


inline void set_group_ptrs(const std::vector<OperatorTerm>& __restrict terms, std::vector<std::size_t>& vec)
{
    vec.resize(0);
    vec.push_back(0);
    std::size_t kk;
    int val = terms[0].group;
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].group > val)
        {
            vec.push_back(kk);
            val = terms[kk].group;
        }
    }
    vec.push_back(terms.size());
}



/**
 * Combine repeated terms that represent same
 * operators, dropping terms smaller than requested tolerance.
 *
 * Input terms must be sorted by weight before calling this routine
 *
 * @param[in] terms Terms for input operator
 * @param[in] out_terms Terms for output operator (to push_back to)
 * @param[in] touched pointer array indicating if term has been touched
 * @param[in] num_terms Number of terms in input operator
 * @param[in] atol Absolute tolerance for term truncation
 *
 */
inline void combine_qubit_terms(std::vector<OperatorTerm>& __restrict terms,
                         std::vector<OperatorTerm>& __restrict out_terms,
                         unsigned int* touched,
                         double atol)
{
    std::size_t kk, qq, num_terms = terms.size();
    std::vector<std::size_t> weight_ptrs;
    set_weight_ptrs(terms, weight_ptrs);
    std::vector<std::vector<OperatorTerm_t>> temp_terms;
    temp_terms.resize(weight_ptrs.size() - 1);
    // do sort over each collection of terms with same weight
    #pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
    for(kk = 0; kk < weight_ptrs.size() - 1; kk++)
    {
        std::size_t jj, mm, pp;
        std::size_t start, stop;
        OperatorTerm_t target_term;
        OperatorTerm_t* current_term;
        int do_combine;
        // set start and stop for terms of the same weight
        start = weight_ptrs[kk];
        stop = weight_ptrs[kk + 1];
        for(jj = start; jj < stop; jj++)
        {
            if(touched[jj]) // If touched, move onto next term
            {
                continue;
            }
            touched[jj] = 1;
            target_term = terms[jj];
            for(mm = jj + 1; mm < stop; mm++)
            {
                if(touched[mm])
                {
                    continue;
                }
                current_term = &terms[mm];
                // filter if offdiag weights differ
                if(target_term.offdiag_weight != current_term->offdiag_weight)
                {
                    continue;
                }

                do_combine = 1;
                // look to see if indices and values match
                for(pp = 0; pp < target_term.indices.size(); pp++)
                {
                    if((target_term.indices[pp] != current_term->indices[pp]) ||
                       (target_term.values[pp] != current_term->values[pp]))
                    {
                        do_combine = 0;
                        break;
                    }
                }
                if(do_combine)
                {
                    touched[mm] = 1;
                    target_term.coeff += current_term->coeff;
                }
            } // end mm for-loop
            // Add term to output if either real or imag parts are greater than atol
            if(std::abs(target_term.coeff) > atol)
            {
                temp_terms[kk].push_back(target_term);
            }
        } // end main jj loop
    } //end kk-loop

    // at end of all, add to output terms
    for(kk = 0; kk < weight_ptrs.size() - 1; kk++)
    {
        for(qq = 0; qq < temp_terms[kk].size(); qq++)
        {
            out_terms.push_back(temp_terms[kk][qq]);
        }
    }

} // end combine_qubit_terms

/**
 * Compute an integer value from the off-diagonal structure of a term
 *
 * @param term The term
 *
 * @return Structure value
 */
inline std::size_t term_offdiag_structure(const OperatorTerm_t& term)
{
    std::size_t kk;
    std::size_t out = 0;
    //#pragma omp simd reduction(+:out)
    for(kk = 0; kk < term.values.size(); ++kk)
    {
        out +=
            (term.indices[kk] + 1) *
            (term.values[kk] > 2); // need plus one here so that an offdiag on 0 index does not look like a diagonal term
    }
    return out;
}

/**
 * Comparator for off-diagonal grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offdiag_comp(const OperatorTerm& term1, const OperatorTerm& term2)
{
    return term_offdiag_structure(term1) < term_offdiag_structure(term2);
}

/**
 * Sort terms in operator by their off-diagonal structure value
 *
 * @param terms Vector of operator terms
 *
 */
inline void term_offdiag_sort(std::vector<OperatorTerm>& terms)
{
    std::sort(terms.begin(), terms.end(), offdiag_comp);
}

/**
 * Set the pointers for the off-diagonal weights
 *
 * @param terms Operator terms
 * @param vec Vector to add pointers to
 *
 */
inline void set_offdiag_weight_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                             std::vector<std::size_t>& vec)
{
    vec.resize(0);
    std::size_t kk;
    unsigned int val = terms[0].offdiag_weight;
    vec.push_back(0);
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].offdiag_weight > val)
        {
            vec.push_back(kk);
            val = terms[kk].offdiag_weight;
        }
    }
    vec.push_back(terms.size());
}

/**
 * Find max. number of elements with same off-diag weight
 *
 * Used for offsetting the group counter for parallel execution
 *
 * @param[in] vec Vector of off-diagonal pointers
 * @param[in] size Number of elements in vec
 * 
 * @returns size_t for max. number of terms
 *
 */
inline std::size_t max_offdiag_ptr_size(std::vector<std::size_t>& vec)
{
    std::size_t kk;
    std::size_t temp, max = 0;
    if(!vec.size()) // This is the case for all diagonals operator
    {
        max = 0;
    }
    else
    {
    for(kk = 0; kk < vec.size() - 1; kk++)
        {
            temp = vec[kk + 1] - vec[kk];
            if(temp > max)
            {
                max = temp;
            }
        }
    }   
    return max;
}

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};


/**
 * In-place marks a term as extended or not
 *
 * @param term Hamiltonian term
 *
 */
inline void set_extended_flag(OperatorTerm_t& term)
{
    std::size_t kk;
    int out = 1;
    for(kk = 0; kk < term.values.size(); kk++)
    {
        out *= REV_EXT_MASK[term.values[kk]];
    }
    term.extended = (!out);
}

/**
 * In-place set off-diagonal weight and real_phase
 *
 * @param term Hamiltonian term
 *
 */
inline void set_offdiag_weight_and_phase(OperatorTerm_t& term)
{
    std::size_t kk;
    unsigned int weight = 0;
    unsigned int temp, num_y = 0;
    unsigned char* values = &term.values[0];
    for(kk = 0; kk < term.values.size(); kk++)
    {
        weight += (values[kk] > 2);
        num_y += (values[kk] == 4);
    }
    term.offdiag_weight = weight;
    // Do the real_phase for checking if operator itself can be cast as symmetric (real)
    temp = num_y % 4;
    if(temp)
    {
        term.real_phase = (temp % 2) - 1;
    }
}


/**
 * Comparator for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offdiag_group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.group < term2.group;
}


/**
 * Sort terms with same off-diagonal weight into groups that share the
 * same off-diagonal structure
 *
 */
inline void term_group_sort(std::vector<OperatorTerm_t>& terms,
                     std::size_t* __restrict offdiag_weight_ptrs,
                     std::size_t len_ptrs,
                     std::size_t max_group_size)
{
    std::size_t ii;
    // Reset all groupings
    for(ii = 0; ii < terms.size(); ii++)
    {
        terms[ii].group = 0; // diagonals are group 0 by convention
        if(terms[ii].offdiag_weight > 0)
        {
            terms[ii].group = -1;
        }
    } // end reset

 

    std::ptrdiff_t dist;
    #pragma omp parallel for schedule(dynamic) if(terms.size() > 1024)
    for(ii = 0; ii < len_ptrs - 1; ii++)
    {
        std::size_t start = offdiag_weight_ptrs[ii];
        std::size_t stop = offdiag_weight_ptrs[ii + 1];
        int group_idx = ii * (max_group_size);
        std::size_t kk, ll, idx;
        OperatorTerm_t* term;
        OperatorTerm_t* term2;
        std::vector<unsigned int>::iterator inds_it;
        int match;
        std::size_t ind_size;

        if(terms[start].group == 0) // group is the diagonal group
        {
            continue;
        }

        for(kk = start; kk < stop; kk++)
        {
            term = &terms[kk];
            ind_size = term->indices.size();
            if(term->group < 0) // term is not touched yet
            {
                group_idx += 1; // diags are group zero, so go to 1 first
                term->group = group_idx;
            }
            // Loop over all terms from kk+1 on up t ostop
            for(ll = kk + 1; ll < stop; ll++)
            {
                term2 = &terms[ll];
                // term2 is not matched and number of off-diag ops is equal
                if((term2->group < 0) && (term2->offdiag_weight == term->offdiag_weight))
                {
                    match = 1;
                    for(idx = 0; idx < ind_size; idx++)
                    {
                        // found off-diag term at idx
                        if(term->values[idx] > 2)
                        {
                            // Tell me if the index is also found in term2
                            inds_it = std::find(
                                term2->indices.begin(), term2->indices.end(), term->indices[idx]);
                            if(inds_it == term2->indices.end())
                            {
                                match = 0;
                                break;
                            }
                            // if the index is in term2, find out its location and check for off-diag there
                            else
                            {
                                dist = std::distance(term2->indices.begin(), inds_it);
                                if(!(term2->values[dist] > 2))
                                {
                                    match = 0;
                                    break;
                                }
                            }
                        } // end found off-diag term
                    } // end idx for-loop

                    if(match)
                    { // If match
                        term2->group = group_idx;
                    }
                } // end non-id match
            } // end ll for-loop
        } // end kk for-loop
        // sort by group index within the start and stop indices
        std::sort(&terms[start], &terms[stop], offdiag_group_comp);
    } // end ii loop

    // relabel groups into continuous integers
    int current_group = 0;
    int current_idx = 0, next_idx = 1;
    for(ii = 0; ii < terms.size(); ii++)
    {
        if(terms[ii].group != current_group)
        {
            current_group = terms[ii].group;
            current_idx = next_idx;
            terms[ii].group = current_idx;
            next_idx += 1;
        }
        else
        {
            terms[ii].group = current_idx;
        }
    }
}

/**
 * Compute the ladder integer value for a given qubit term
 *
 */
inline unsigned int term_ladder_int(const OperatorTerm& term, unsigned int ladder_width)
{
    unsigned int subset = 0;
    unsigned int kk, counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 4)
        {
            subset = subset | ((unsigned int)term.values[kk] - 5U) << counter;
            counter += 1;
        }
    }
    if(counter < ladder_width)
    {
        ladder_width = counter;
    }
    if(!counter)
    {
        subset = MAX_UINT;
    }
    else
    {
        subset = subset & ((1U << ladder_width) - 1U);
    }
    return subset;
}



/**
 * Sort terms within each group by their ladder integer values
 *
 */
inline void sort_groups_by_ladder_int(std::vector<OperatorTerm>& terms,
                                      const std::size_t* group_ptrs,
                                      unsigned int num_groups,
                                      unsigned int ladder_width)
{

    unsigned int kk;
    #pragma omp parallel for if(num_groups > 128)
    for(kk = 0; kk < num_groups; kk++)
    {
        std::size_t start, stop;
        start = group_ptrs[kk];
        stop = group_ptrs[kk + 1];
        if(!terms[start].group) // This is true if the group=0 and thus are diagonal terms
        {
            continue;
        }
        std::sort(&terms[start],
                  &terms[stop],
                  [=](const OperatorTerm& a, const OperatorTerm& b) {
                      unsigned int res_a, res_b;
                      res_a = term_ladder_int(a, ladder_width);
                      res_b = term_ladder_int(b, ladder_width);
                      return res_a < res_b;
                  });
    }
}


/**
 * Compute the offdiag indices for the first term in a group and add it to the group
 * offdiag indices vector
 *
 * @param term Operator term
 * @param ladder_inds Pre-sized array (size=num_inds) to store indices in
 * @param num_inds Number of elements to consider for appending
 *
 */
inline void compute_term_offdiag_inds(const OperatorTerm_t& term,
                                      unsigned int* offdiag_inds)
{
    unsigned int kk;
    unsigned int counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 2)
        {
            offdiag_inds[counter] = term.indices[kk];
            counter += 1;
        }
    }
}

/**
 * Set the offdiag indices for each group in a off-diagonal type=2 Hamiltonian
 *
 * @param terms Operator terms
 * @param group_indices Vector of vectors of group_indices
 * @param group_ptrs Pointer of array of group pointers
 * @param num_groups Number of groups = len(group_ptrs) - 1
 * @param ladder_width Target ladder indices width for type=2 operators
 * @param oper_type Type of operator, 1 or 2
 *
 */
inline void set_group_offdiag_indices(const std::vector<OperatorTerm_t>& terms,
                               std::vector<std::vector<unsigned int>>& group_indices,
                               const std::size_t* group_ptrs,
                               unsigned int num_groups)
{
    unsigned int kk;
    unsigned int inds_len;
    group_indices.resize(num_groups);
    for(kk = 0; kk < num_groups; kk++)
    {
        inds_len = terms[group_ptrs[kk]].offdiag_weight;
        group_indices[kk].resize(inds_len);
        compute_term_offdiag_inds(terms[group_ptrs[kk]], &(group_indices[kk])[0]);
    }
}








/** @struct QubitOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator
{
    unsigned int width;
    std::vector<OperatorTerm_t> terms;
    int type{1};
    unsigned int ladder_width{DEFAULT_LADDER_WIDTH};
    int sorted{0};
    int weight_sorted{0};
    int off_weight_sorted{0};
    int ladder_sorted{0};

    QubitOperator() {}
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    QubitOperator(unsigned int x){width = x;}

    QubitOperator(unsigned int x, std::vector<TermData> data): width(x)
    {
       unsigned int num_terms = data.size();
       std::size_t kk;
       TermData tdata;
       OperatorTerm term;
       std::complex<double> coeff = 1.0;
       for(kk =0; kk < num_terms; kk++)
       {
        tdata = data[kk];
        _validate_indices(std::get<1>(tdata), width); // validate that all indices are less than operator width
        // If there are no indices and the coeff==0 then the term should be an identity term with coeff=1
        if(std::get<1>(tdata).size() == 0 && std::get<2>(tdata) == std::complex<double>(0,0)){
            coeff = 1.0;
        }
        else{
            coeff = std::get<2>(tdata);
        }
        term = OperatorTerm(std::get<0>(tdata), std::get<1>(tdata), coeff);
        term.set_proj_indices();
        set_offdiag_weight_and_phase(term);
        set_extended_flag(term);
        terms.push_back(term);
       }
    }
    // destructor
    ~QubitOperator()
    {
        std::vector<OperatorTerm_t>().swap(terms);
    }
    /**
     * QubitOperator from string label
     */
    static QubitOperator from_label(std::string label)
    {
        unsigned int width = label.size();
        unsigned char val;
        std::size_t counter=0;
        QubitOperator out = QubitOperator(width);
        OperatorTerm term = OperatorTerm(1.0); // start with term set with coeff = 1.0
        for (auto it = label.rbegin(); it != label.rend(); it++)
        {
            if(*it != 73)
            {
                val = oper_map[*it];
                term.values.push_back(val);
                term.indices.push_back(counter);
            }
            counter += 1;
        }
        set_offdiag_weight_and_phase(term);
        term.set_proj_indices();
        set_extended_flag(term);
        out.terms.push_back(term);
        return out;
    }
    /**
     * Grab a single term by index
     */
    OperatorTerm_t operator[] (std::size_t kk) const
    {
        return terms[kk];
    }
    /**
     * Inplace multiplication by a complex value
     */
    QubitOperator& operator*=(std::complex<double> c)
    {
        for(std::size_t kk=0; kk<this->size(); kk++)
        {
            terms[kk] *= c;
        }
        return *this;
    }
    /**
     * multiplication by a complex value (need one for mult on each side)
     */
    friend QubitOperator operator*(QubitOperator& op, std::complex<double> c)
    {
        QubitOperator out = op.copy();
        for(std::size_t kk=0; kk<out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    friend QubitOperator operator*(std::complex<double> c, QubitOperator& op)
    {
        QubitOperator out = op.copy();
        for(std::size_t kk=0; kk<out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    /**
     * Inplace addition by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator+=(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        for(std::size_t kk=0; kk<other.size(); kk++)
        {
            this->terms.push_back(other.terms[kk]);
        }
        this->sorted = 0;
        return *this;
    }
    /**
     * Inplace subtraction by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator-=(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        
        OperatorTerm term;
        for(std::size_t kk=0; kk<other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            this->terms.push_back(term);
        }
        this->sorted = 0;
        return *this;
    }
    /**
     * Subtraction by another QubitOperator
     * 
     * @param[in] other Operator to subject to this one
     * @return New operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator-(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        
        OperatorTerm term;
        QubitOperator out = this->copy();
        for(std::size_t kk=0; kk<other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            out.terms.push_back(term);
        }
        return out;
    }
    /**
     * Addition by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @return The new operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator+(QubitOperator other) const
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        QubitOperator out = this->copy();
        for(std::size_t kk=0; kk<other.size(); kk++)
        {
            out.terms.push_back(other.terms[kk]);
        }
        return out;
    }
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const QubitOperator& self) -> std::ostream&
    { 
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        OperatorTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<QubitOperator["; // start output here
        for(kk=0; kk < num_terms; kk++)
        {
            term = self.terms[kk];
            os << "{";
            for(jj=0; jj < term.indices.size(); jj++)
            {
                os << rev_oper_map[term.values[jj]] << ":" << term.indices[jj];
                if(jj!=term.indices.size()-1)
                {
                    os << " ";
                }
                
            }
            os << ", " << term.coeff;
            os << "}";
            if(kk!=num_terms-1)
            {
                os << ", ";
            }
        }
        if(too_many_terms)
        {
            os << " + " << (total_terms-100) << "terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    auto begin() { return terms.begin(); }
    auto end()   { return terms.end(); } 
    /**
     * The number of terms in the operator
     *
     * @return The number of terms in the operator
     */
    std::size_t size() const {return terms.size();}
    std::size_t num_terms() const {return terms.size();}
    /**
     * The number of terms in the operator
     *
     * @return A copy of the current operator
     */
    QubitOperator copy() const
    {
        QubitOperator out = QubitOperator(this->width);
        out.terms = this->terms;
        out.type = this->type;
        return out;
    }
    /**
     * Is the operator diagonal
     */
    bool is_diagonal() const
    {
        std::size_t kk;
        bool diag = 1;
        for(kk=0; kk<terms.size(); kk++)
        {
            if(!terms[kk].is_diagonal()){
                diag =0;
                break;
            }
        }
        return diag;
    }
    /**
     * Can operator be described via a symmetric matrix
     */
    bool is_real() const
    {
        bool out = true;
        for(std::size_t kk=0; kk < this->size(); kk++)
        {
            if(std::abs(terms[kk].coeff.imag()) > ATOL || !terms[kk].real_phase)
            {
                out = false;
                break;
            }
        }
        return out;
    }
    /**
     * Set operator type inplace
     * 
     * @param[in] x Integer type of operator
     * @throw Error if type is not 1 or 2
     */
    QubitOperator& set_type(int x)
    {
        if (x > 2 || x < 1)
        {
            throw std::runtime_error("Type must be 1 or 2");
        }
        this->type = x;
        return *this;
    }
    /**
    * Return vector of weights for each term
    * 
    * @param[out] out Vector of weights for terms
    * 
    */
    std::vector<unsigned int> weights() const
    {
        std::vector<unsigned int> out;
        for(std::size_t kk=0; kk<this->size(); kk++)
        {
            out.push_back(this->terms[kk].weight());
        }
        return out; 
    }
    /**
     * Split operator into diagonal and off-diagonal components
     * 
     * @return Diagonal and off-diagonal operators
     */
    std::pair<QubitOperator, QubitOperator> split_diagonal() const
    {
        QubitOperator diag = QubitOperator(this->width);
        QubitOperator off = QubitOperator(this->width);
        for(auto term: this->terms)
        {
            if(term.is_diagonal())
            {
                diag.terms.push_back(term);
            }
            else
            {
                off.terms.push_back(term);
            }
        }
        off.type = this->type;
        diag.type = this->type;
        return {diag, off};
    }
    /**Constant energy of operator
    * 
    */
    double constant_energy() const
    {
        double out = 0;
        for(std::size_t kk=0; kk < terms.size(); kk++)
        {
            if(!terms[kk].indices.size())
            {
                out += terms[kk].coeff.real();
            } 
        }
        return out;
    }
    /**
    * Remove constant terms from operator
    * 
    */
    QubitOperator remove_constant_terms()
    {
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk=0; kk < this->size(); kk++)
        {
            if(terms[kk].indices.size())
            {
                out.terms.push_back(terms[kk]);
            } 
        }
        out.type = this->type;
        return out;
    }
    /**
    * In-place sorting of terms by weight
    * 
    */
    QubitOperator& weight_sort()
    {
        // sort by weight
        std::sort(terms.begin(), terms.end(), weight_comp);
        this->off_weight_sorted = 0;
        this->weight_sorted = 1;
        this->sorted = 0;
        return *this;
    }
    /**
    * In-place sorting of terms by off-diagonal weight
    * 
    */
    QubitOperator& offdiag_weight_sort()
    {
        // sort by off-diagonal weight
        std::sort(terms.begin(), terms.end(), offweight_comp);
        this->off_weight_sorted = 1;
        this->weight_sorted = 0;
        this->sorted = 0;
        return *this;
    }
    /**
    * Pointers to starting indices for off-diagonally sorted operator
    * 
    */
    std::vector<std::size_t> offdiag_weight_ptrs()
    {
        std::vector<std::size_t> ptrs;
        if(!this->off_weight_sorted)
        {
            this->offdiag_weight_sort();
        }
        set_offdiag_weight_ptrs(terms, ptrs);
        return ptrs;
    }
    /**
    * In-place sorting of terms into groups (shared off-diagonal structure)
    * 
    */
    QubitOperator& group_sort()
    {
        if(this->size()) // do stuff only if there are terms in the operator
        {
            if(!this->off_weight_sorted)
            {
                this->offdiag_weight_sort();
            }
            std::vector<std::size_t> ptrs = this->offdiag_weight_ptrs();
            std::size_t max_group_size = max_offdiag_ptr_size(ptrs);
            term_group_sort(this->terms, &ptrs[0], ptrs.size(), max_group_size);
        }
        this->sorted = 1;
        return *this;
    }
    /**
    * Return a vector of all the term group labels
    * 
    */
    std::vector<int> groups() const
    {
        std::vector<int> out;
        out.resize(terms.size());
        for(std::size_t kk=0 ; kk < terms.size(); kk++)
        {
            out[kk] = terms[kk].group;
        }
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    std::vector<std::size_t> group_ptrs()
    {
        std::vector<std::size_t> out;
        if(!this->size()) // return empty vector if no terms
        {
            return out;
        }
        if(!this->sorted)
        {
            this->group_sort();
        }
        set_group_ptrs(terms, out);
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    QubitOperator terms_by_group(int idx)
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk=0; kk < this->size(); kk++)
        {
            if(terms[kk].group == idx)
            {
                out.terms.push_back(terms[kk]);
            }
            else if (terms[kk].group > idx)
            {
                break;
            }
        }
        if(!out.size())
        {
            throw std::runtime_error("No terms with given group index found");
        }
        out.sorted = 1;
        out.type = this->type;
        return out;
    }
    /**
    * Off-diagonal indices for each group of terms
    */
    std::vector<std::vector<unsigned int>> group_offdiag_indices()
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }

        std::vector<std::vector<unsigned int>> out;
        std::vector<std::size_t> ptrs = this->group_ptrs();
        set_group_offdiag_indices(this->terms, out, &ptrs[0], ptrs.size()-1);
        return out;
    }
    /**
    * Combine repeated terms in operator
    * 
    * @param[in] atol Tolerance for determining if a combined coefficient is zero
    * 
    * @return Output QubitOperator with terms combined
    * 
    */
    QubitOperator combine_repeated_terms(double atol = 1e-12)
    {
        QubitOperator out = QubitOperator(this->width);
        if(!this->weight_sorted){
            this->weight_sort();
        }
        std::vector<unsigned int> touched;
        touched.resize(this->size());
        combine_qubit_terms(this->terms, out.terms, &touched[0], atol);
        out.type = this->type;
        return out;
    }
    /**
    * Return vector of off-diagonal weights for each term
    * 
    * @return Vector of off-diagonal weights for terms
    * 
    */
    std::vector<unsigned int> offdiag_weights() const
    {
        std::vector<unsigned int> out;
        out.resize(terms.size());
        std::size_t kk;
        for(kk=0; kk < terms.size(); kk++)
        {
            out[kk] = terms[kk].offdiag_weight;
        }
        return out;
    }
    /**
    * In-place sort terms in groups by their ladder integer values
    * 
    */
    QubitOperator& group_term_sort_by_ladder_int(unsigned int ladder_width=4)
    {
        if(!this->sorted)
        {
            this->group_sort();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        sort_groups_by_ladder_int(this->terms, &ptrs[0], ptrs.size()-1, ladder_width);
        this->ladder_width = ladder_width;
        this->ladder_sorted = 1;
        return *this;
    }
    /**
    * Vector of ladder integer values for terms in operators
    * 
    * If no ladder ops present then default int is max(uint32)
    * 
    */
    std::vector<unsigned int> ladder_integers()
    {
       std::vector<unsigned int> out;
       if(!this->ladder_sorted)
       {
        this->group_term_sort_by_ladder_int();
       }
       for(std::size_t kk=0; kk < this->size(); kk++)
       {
            out.push_back(term_ladder_int(terms[kk], this->ladder_width));
       }
       return out;
    }

} QubitOperator_t;
























// Fermionic components ---------------------------------------------------------------------------


/** @brief Data structure for each Fermionic operator term
 *
 * @var indices the modes (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 */
typedef struct FermionicTerm
{
    std::vector<unsigned char> values;
    std::vector<unsigned int> indices;
    std::complex<double> coeff;

    FermionicTerm() {}
    FermionicTerm(std::complex<double> c): coeff(c) {} // Init empty term with given coefficient
    FermionicTerm(std::string vals, std::vector<unsigned int> inds, std::complex<double> c): indices(inds), coeff(c)
    {
        // Iterate over string of values, mapping to new values and adding to term
        for(std::string::iterator it = vals.begin(); it != vals.end(); ++it)
        {
            if(*it == 73)
            {
                throw std::runtime_error("Cannot use identity operators in sparse format.");
            }
            else{
                values.push_back(oper_map[*it]);
            }
        }
        //check that length of values == length of indices
        if(values.size() != indices.size())
        {
            throw std::runtime_error("Size of values vector does not equal that of indices.");
        }
        insertion_sort();
    }
    // destructor
    ~FermionicTerm()
    {
        std::vector<unsigned char>().swap(values);
        std::vector<unsigned int>().swap(indices);
    }
    /**
     * Return the size of the term
     */
    std::size_t size() const {return indices.size();}
    /**
     * Insertion sort indices (and values) in the term
     */
    void insertion_sort()
    {
        std::size_t kk;
        int ll;
        std::size_t num_elems = indices.size();
        unsigned int temp_index;
        unsigned char temp_value;
        int prefactor = 1;
        for(kk=1; kk < num_elems; kk++)
        {
            temp_index = indices[kk];
            temp_value = values[kk];
            ll = kk - 1;
            // Only switch elements if they are of different indices
            // In this case we always pick up a minus sign that
            // we need to keep track of with the 'prefactor'
            while(ll >= 0 && temp_index < indices[ll])
            {
                indices[ll + 1] = indices[ll];
                values[ll + 1] = values[ll];
                // Only add a minus sign if both operators (values)
                // are not projectors (ie. > 4 since '-'=5 and '+'=6)
                if((temp_value > 4) and (values[ll] > 4))
                {
                    prefactor *= -1;
                }
                ll -= 1;
            }
            indices[ll + 1] = temp_index;
            values[ll + 1] = temp_value;
        }
        coeff *= prefactor;
    }
} FermionicTerm_t;



/** @struct FermionicOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct FermionicOperator
{
    unsigned int width;
    std::vector<FermionicTerm_t> terms;
    FermionicOperator() {}
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    FermionicOperator(unsigned int x){width = x;}
    FermionicOperator(unsigned int x, std::vector<TermData> data): width(x)
    {
       unsigned int num_terms = data.size();
       std::size_t kk;
       TermData tdata;
       for(kk =0; kk < num_terms; kk++)
       {
        tdata = data[kk];
        _validate_indices(std::get<1>(tdata), width); // validate that all indices are less than operator width
        terms.push_back(FermionicTerm(std::get<0>(tdata), std::get<1>(tdata), std::get<2>(tdata)));
       }
    }
    // deallocation
    ~FermionicOperator()
    {
        std::vector<FermionicTerm_t>().swap(terms);
    }
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const FermionicOperator& self) -> std::ostream&
    { 
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        FermionicTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<FermionicOperator["; // start output here
        for(kk=0; kk < num_terms; kk++)
        {
            term = self.terms[kk];
            os << "{";
            for(jj=0; jj < term.indices.size(); jj++)
            {
                os << rev_oper_map[term.values[jj]] << ":" << term.indices[jj];
                if(jj!=term.indices.size()-1)
                {
                    os << " ";
                }
                
            }
            os << ", " << term.coeff;
            os << "}";
            if(kk!=num_terms-1)
            {
                os << ", ";
            }
        }
        if(too_many_terms)
        {
            os << " + " << (total_terms-100) << "terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    /**
     * Return the size of the operator
     */
    std::size_t size() const {return terms.size();}
} FermionicOperator_t;


// Subspace components ---------------------------------------------------------------------------


/** @struct subspace
 * @brief Data structure for subspace defined by counts
 *
 * @var bitstrings The subspace bit-strings stored in a hash table
 * @var num_qubits The number of qubits, i.e length of bitstrings
 * @var size Dimension / number of bit-strings in the subspace
 */
typedef struct Subspace
{
    bitset_map_namespace::BitsetHashMapWrapper bitstrings;
    unsigned int num_qubits;
    std::size_t size;
} Subspace_t;
