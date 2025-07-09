#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <vector>
#include <algorithm>
#include "base.hpp"
#include <boost/dynamic_bitset.hpp>


// Mask for checking if extended operator term
// has a nonzero value at the given row bit-string
const int EXT_NZ_MASK[8] = {1, 1, 0, 1, 1, 1, 0, 1};


/**
 * Compute the integer corresponding to the bin-width of a bitset
 *
 * @param bitset The target bitset
 * @param bin_width The bin-width
 * @param res The resulting integer
 */
inline void bin_int(const boost::dynamic_bitset<std::size_t>& bitset, 
                    const unsigned int bin_width, std::size_t& res)
    {
        res = bitset.m_bits[0] & (( 1ULL << bin_width) - 1);
    }


/**
 * Flip a series of bits in a bitset
 *
 * @param bitset The target bitset
 * @param arr Pointer to array of indices which to flip
 * @param size The size of the array
 */
inline void flip_bits(boost::dynamic_bitset<std::size_t>& bitset,
                      unsigned int * arr, unsigned int size)
    {
        unsigned int  kk;
        unsigned int block_num, block_idx;
        unsigned int pos;
        for(kk=0; kk < size; kk++)
        {
            pos = arr[kk];
            block_num = pos / BITS_PER_BLOCK;
            block_idx = pos % BITS_PER_BLOCK;
            bitset.m_bits[block_num] = bitset.m_bits[block_num] ^ (1ULL << block_idx);
        }
    }


/**
 * Gets the column bitset from an input row bitset and operator term
 *
 * @param col The input row bitset that will be converted to column
 * @param pos Array of indices on which operators act
 * @param val The value representing each operator
 * @param N Number of non-ID operators in the term
 */
inline void get_column_bitset(boost::dynamic_bitset<std::size_t>& col,
                              const unsigned int *__restrict pos,
                              const unsigned char *__restrict val,
                              const unsigned int N)
{
    unsigned int block_num, block_idx;
    unsigned int ind;
    unsigned int kk;
    for (kk = 0; kk < N; kk++)
        {
            ind = pos[kk];
            block_num = ind / BITS_PER_BLOCK;
            block_idx = ind % BITS_PER_BLOCK;
            col.m_bits[block_num] = col.m_bits[block_num] ^ (std::size_t)((val[kk] > 2) << block_idx);
        }
}


inline void bitset_column_index(const std::size_t start, const std::size_t stop,
                                const boost::dynamic_bitset<std::size_t>& col,
                                const std::vector<boost::dynamic_bitset<std::size_t> >& subspace,
                                std::size_t& col_idx)
{
    std::size_t kk;
    col_idx = MAX_SIZE_T;
    for(kk=start; kk < stop; kk++)
    {
        if(col == subspace[kk])
        {
            col_idx = kk;
            break;
        }
    }
}

inline int nonzero_extended_bitset(const OperatorTerm_t * term,
                                   const boost::dynamic_bitset<std::size_t>& row)
{
    std::size_t kk;
    unsigned int ind, block_num, block_idx, row_int;
    int out = 1;
    for(kk=0; kk < term->indices.size(); kk++){
        ind = term->indices[kk];
        block_num = ind / BITS_PER_BLOCK;
        block_idx = ind % BITS_PER_BLOCK;
        row_int = (row.m_bits[block_num] >> block_idx) & 1;
        if(!EXT_NZ_MASK[term->values[kk] + row_int])
        {
            out = 0;
            break;
        }
    }
    return out;
}


inline void sort_bitset_vector(std::vector<boost::dynamic_bitset<std::size_t> >& vec,
                               unsigned int bin_width)
    {
        
        std::sort(vec.begin(), vec.end(), [=](const boost::dynamic_bitset<std::size_t> a,
                                              const boost::dynamic_bitset<std::size_t> b)
                                  {
                                    std::size_t res_a, res_b;  
                                    bin_int(a, bin_width, res_a);
                                    bin_int(b, bin_width, res_b);
                                    return res_a < res_b;
                                  });
    }

/**
 * Convert bits at given indices into an unsigned integer
 *
 * @param row The input row bitset
 * @param inds Pointer to array of indices as unsigned ints 
 * @param ladder_width The number of bits to consider
 */
inline unsigned int bitset_ladder_int(const boost::dynamic_bitset<std::size_t>& row, 
                                      const unsigned int * inds,
                                      unsigned int ladder_width)
{
    unsigned int subset = 0;
    unsigned int kk, pos, block_num, block_idx, row_int, counter=0;
    for(kk=0; kk < ladder_width; kk++)
    {
        pos = inds[kk];
        block_num = pos / BITS_PER_BLOCK;
        block_idx = pos % BITS_PER_BLOCK;
        row_int = (row.m_bits[block_num] >> block_idx) & 1;
        subset = subset | row_int << counter;
        counter += 1U;
    }
    return subset;
}


/**
 * verifies that bitstring passes constraints of the term projection operators
 *
 * @param bitset The target row bitset
 * @param values The values representing the type of operator
 * @param proj_indices Pointer to array of indices on which projectors act
 * @param size The size of the proj array
 */
inline bool passes_proj_validation(boost::dynamic_bitset<std::size_t>& bitset,
                                  const unsigned int * proj_bits,
                                  const unsigned int * proj_indices,
                                  unsigned int size)
{
    unsigned int  kk;
    unsigned int block_num, block_idx;
    unsigned int pos;
    bool out = 1;
    unsigned int bit;
    for(kk=0; kk < size; kk++)
    {
        pos = proj_indices[kk];
        block_num = pos / BITS_PER_BLOCK;
        block_idx = pos % BITS_PER_BLOCK;
        bit = ((bitset.m_bits[block_num] >> block_idx) & static_cast<std::size_t>(1));
        out = (bit == proj_bits[kk]);
    }
    return out;
}