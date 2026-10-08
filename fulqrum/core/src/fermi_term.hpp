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
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "oper_utils.hpp"
#include "packing.hpp"
#include "qubit_term.hpp"

// Fermionic components ---------------------------------------------------------------------------

/** @brief Data structure for each Fermionic operator term
 *
 * @var indices the modes (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 */
typedef struct FermionicTerm
{
    std::complex<double> coeff{0};
    std::vector<width_t> data;
    std::vector<width_t> proj_indices;
    std::vector<width_t> proj_bits;
    unsigned int offdiag_structure{0};
    unsigned int proj_structure{0};
    width_t offdiag_weight{0};

    FermionicTerm() = default;
    FermionicTerm(const FermionicTerm&) = default;
    FermionicTerm(FermionicTerm&&) = default;
    FermionicTerm& operator=(const FermionicTerm&) = default;
    FermionicTerm& operator=(FermionicTerm&&) = default;
    ~FermionicTerm() = default;

    FermionicTerm(std::complex<double> c)
        : coeff(c)
    {} // Init empty term with given coefficient

    FermionicTerm(std::string vals, std::vector<width_t> inds, std::complex<double> c)
        : coeff(c)
    {
        const std::size_t num_vals = vals.size();
        if(num_vals != inds.size())
        {
            throw std::runtime_error("Size of values vector does not equal that of indices.");
        }
        data.reserve(num_vals);
        if(!num_vals &&
           c ==
               0.0) //if empty data is passed in then we assume this is an identity and set coeff properly
        {
            this->coeff = 1.0;
        }
        // Iterate over string of values, mapping to internal codes
        for(std::size_t i = 0; i < num_vals; ++i)
        {
            const unsigned char ch = static_cast<unsigned char>(vals[i]);
            if(ch == 73) // 'I' identity
            {
                throw std::runtime_error("Cannot use identity operators in sparse format.");
            }
            const unsigned char val = oper_map[ch];
            data.push_back(pack_indval(inds[i], val));
            const bool is_offdiag = (val > 2);
            offdiag_weight += static_cast<width_t>(is_offdiag);
            offdiag_structure += (inds[i] + 1) * static_cast<unsigned int>(is_offdiag);
        }
        insertion_sort();
        set_term_proj_indices_fermi(*this);
    }

    FermionicTerm copy() const
    {
        return *this;
    }

    /**
     * Inplace multiplication by a complex value
     */
    FermionicTerm& operator*=(std::complex<double> c)
    {
        coeff *= c;
        return *this;
    }
    /**
     * Term multiplication by a complex number
     */
    friend FermionicTerm operator*(const FermionicTerm& op, std::complex<double> c)
    {
        FermionicTerm out = op;
        out.coeff *= c;
        return out;
    }
    /**
     * Term multiplication by a complex number
     */
    friend FermionicTerm operator*(std::complex<double> c, const FermionicTerm& op)
    {
        FermionicTerm out = op;
        out.coeff *= c;
        return out;
    }
    /**
     * Return the size of the term
     */
    std::size_t size() const
    {
        return data.size();
    }
    /**
     * Return vector of operator and index pairs
     */
    std::vector<OpData> operators() const
    {
        std::vector<OpData> out;
        for(std::size_t kk = 0; kk < data.size(); kk++)
        {
            auto [ind, val] = unpack_indval(data[kk]);
            OpData item{std::string(1, static_cast<char>(rev_oper_map[val])), ind};
            out.push_back(item);
        }
        return out;
    }
    /**
     * Insertion sort indices (and values) in the term.
     * Tracks anticommutation sign flips for ladder operators (val > 4).
     */
    void insertion_sort()
    {
        const int num_elems = static_cast<int>(data.size());
        int prefactor = 1;
        for(int kk = 1; kk < num_elems; kk++)
        {
            const auto [temp_index, temp_value] = unpack_indval(data[kk]);
            int ll = kk - 1;
            // Swapping two ladder operators (val > 4) over different indices costs a sign.
            while(ll >= 0 && temp_index < unpack_ind(data[ll]))
            {
                data[ll + 1] = data[ll];
                if(temp_value > 4 && unpack_val(data[ll]) > 4)
                {
                    prefactor = -prefactor;
                }
                --ll;
            }
            data[ll + 1] = pack_indval(temp_index, temp_value);
        }
        coeff *= prefactor;
    }
} FermionicTerm_t;

/**
 * Compute the JW phase for a given operator.
 * Returns -1 for op=5 ('-') or op=2 ('1'), +1 otherwise.
 *
 * @param[in] op The operator in question
 * @return Integer phase value (+1 or -1)
 */
inline int jw_phase(const unsigned char op)
{
    return (op == 2 || op == 5) ? -1 : 1;
}

/**
 * Compute the extended JW transformation for a single Fermionic term
 *
 * @param[in] fermi_term Input Fermionic term
 * @param[in,out] qubit_term Output qubit term
 */
inline void jw_term(const FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)
{
    const int num_elems = static_cast<int>(fermi_term.data.size());
    int phase = 1;
    qubit_term.coeff = fermi_term.coeff;

    // Reserve for case where all elements plus Z-fill between them
    if(num_elems > 0)
    {
        qubit_term.indices.reserve(unpack_ind(fermi_term.data[0]) + 1);
        qubit_term.values.reserve(unpack_ind(fermi_term.data[0]) + 1);
    }

    // Start with do_z = 0 since nothing has been done yet
    int do_z = 0;
    for(int kk = num_elems - 1; kk > -1; kk--)
    {
        const auto [current_ind, current_val] = unpack_indval(fermi_term.data[kk]);
        // Add start element to qubit operator
        qubit_term.indices.push_back(current_ind);
        qubit_term.values.push_back(current_val);
        // If a Z term acts on the current value, need to take into account phase factor
        if(do_z)
        {
            phase *= jw_phase(current_val);
        }
        // update do_z with this operator
        do_z ^= (current_val > 4);
        // if not at last element in num_elems and do_z
        // make every identity site between this and the next element a Z operator
        if(kk && do_z)
        {
            for(width_t jj = current_ind - 1; jj > unpack_ind(fermi_term.data[kk - 1]); jj--)
            {
                qubit_term.indices.push_back(jj);
                qubit_term.values.push_back(0);
            }
        }
        // If only one element exists (kk==0) but do_z, add Z operators down to site 0
        else if(num_elems == 1 && do_z)
        {
            for(int mm = static_cast<int>(current_ind) - 1; mm > -1; mm--)
            {
                qubit_term.indices.push_back(static_cast<width_t>(mm));
                qubit_term.values.push_back(0);
            }
        }
    } // end kk loop
    qubit_term.coeff *= phase;
}

// Converts internal operator code into a deflated index.
// Codes: 1->0, 2->1, 5->2, 6->3
inline int collapse_value(unsigned char x)
{
    // Small look up table for collapse values
    static constexpr int collapse_lookup[7] = {0, 0, 1, 0, 0, 2, 3};
    return collapse_lookup[x];
}

inline void deflate_term_indices(const FermionicTerm& term,
                                 FermionicTerm& out_term,
                                 const std::vector<int>& collapsed_values)
{
    const std::size_t num_elems = term.data.size();
    out_term.data.reserve(num_elems);

    std::size_t num_touched = 0;
    while(num_touched < num_elems)
    {
        auto [current_index, current_value] = unpack_indval(term.data[num_touched]);
        ++num_touched;
        for(std::size_t kk = num_touched; kk < num_elems; kk++)
        {
            // next term has a matching index with the current one
            if(unpack_ind(term.data[kk]) == current_index)
            {
                const int temp_int = collapsed_values[4 * collapse_value(current_value) +
                                                      collapse_value(unpack_val(term.data[kk]))];
                // This operator becomes a null operator so get rid of whole term
                if(temp_int < 0)
                {
                    return;
                }
                current_value = static_cast<unsigned char>(temp_int);
                ++num_touched;
            }
            else
            { // Move on to next index, note we already index sorted here
                break;
            }
        }
        out_term.data.push_back(pack_indval(current_index, current_value));
        out_term.offdiag_weight += static_cast<width_t>(current_value > 2);
        out_term.offdiag_structure +=
            (current_index + 1) * static_cast<unsigned int>(current_value > 2);
    }
    out_term.coeff = term.coeff;
    set_term_proj_indices_fermi(out_term);
}