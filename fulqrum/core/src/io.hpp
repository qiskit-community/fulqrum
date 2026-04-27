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
#include <iostream>
#include <complex>
#include <vector>
#include <fstream>
#include <format>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "./external/json.hpp"
#include "base.hpp"

using json = nlohmann::json;
typedef std::complex<double> complex;
typedef std::pair<double, double> double_pair;
typedef std::tuple<std::string, std::vector<width_t>, double_pair> JsonTerm;

struct QubitOperator;
struct FermionicOperator;


/**
 * Check if a file exists
 *
 * @param[in] name File name to check
 *
 * @return bool indicating if file exists
 */
inline bool file_exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}


/**
 * Execute a command
 *
 * @param[in] cmd Command string
 *
 * @return Result from running the command
 */
inline std::string exec(const char* cmd) {
    std::vector<char> buffer(128);
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


/**
 * Split a string by a given delimiter
 *
 * @param[in] s Input string
 * @param[in] delimiter The delimiter to split the string by
 *
 * @return Array of strings
 */
std::vector<std::string> split_string(std::string s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}



/**
 * Convert an operator to JSON format, optionally with XZ or ZST compression
 *
 * @param[in] oper Operator to convert to JSON
 * @param[in] filename The name of the output file, e.g. *.json, *.json.xz, or *.json.zst
 * @param[in] overwrite Allow for overwriting files if they already exist
 *
 * @note One should always use compression as it saves ~10x in file size 
 */
template <typename T>
inline void operator_to_json(const T& oper, const std::string& filename, bool overwrite=false)
{
    std::string op_type;
    std::string ending;
    if constexpr(std::is_same_v<T, QubitOperator>)
    {
        op_type = "qubit";
    }
    else if constexpr(std::is_same_v<T, FermionicOperator>)
    {
        op_type = "fermi";
    }
    else
    {
        throw std::runtime_error("Invalid input operator type");
    }
    // check if file already exists
    bool exists = file_exists(filename);
    if(exists && (overwrite == false)){
        throw std::runtime_error(std::format("File '{}' exists and 'overwrite=false'", filename));
    }
    
    // split filename by '.' to get the extension 
    std::vector<std::string> split = split_string(filename, ".");
    ending = split.at(split.size()-1);

    JsonTerm json_term;
    std::vector<JsonTerm> terms;
    for(auto term: oper.terms)
    {
        std::string s = "";
        for(width_t kk=0; kk < term.values.size(); kk++){
            s += rev_oper_map[term.values[kk]];
        }
        json_term = {s, term.indices, {term.coeff.real(), term.coeff.imag()}};
        terms.push_back(json_term);
    }

    json Doc{{"format-version", "1.0"},
            {"fulqrum-version", "0.1.0"},
            {"operator-type", "fermi"},
            {"width", oper.width},
            {"terms", terms}
            };

    
    std::string short_filename;
    if(ending != "json")
    {
        for(std::size_t kk=0; kk < (split.size()-1); kk++)
        {
            short_filename.append(split[kk]);
            if(kk != (split.size()-2)){
                short_filename.append(".");
            }
        }
    }
    else
    {
        short_filename = filename;
    }

    // output json file
    std::fstream File;
    File.open(short_filename, std::ios::out);
    File << Doc;
    File.close();
    
    // compress file, if needed
    std::string compress_str;
    if(ending == "xz")
    {
        compress_str = std::format("xz -9 -f {} ", short_filename);
        exec(compress_str.c_str());  // compress and delete original json 
    }
    else if(ending == "zst")
    {
        compress_str = std::format("zstd --ultra -20 --rm -q {}", short_filename);
        exec(compress_str.c_str());  // compress and delete original json  
    }
    else if(ending != "json")
    {
        throw std::runtime_error("Unknown filename ending");
    }
}



/**
 * Convert a JSON file, optionally with XZ or ZST compression, to an operator
 *
 * @param[in] filename The name of the output file, e.g. *.json, *.json.xz, or *.json.zst
 * @param[in, out] oper The operator to populate with data
 *
 * @note The oper is assumed to be empty  
 */
template<typename U>
inline void json_to_operator(const std::string& filename, U& oper)
{
    // check if file already exists
    bool exists = file_exists(filename);
    if(!exists){
        throw std::runtime_error(std::format("File '{}' not found", filename));
    }
    // split filename by '.' to get the extension 
    std::string ending;
    std::vector<std::string> split = split_string(filename, ".");
    ending = split.at(split.size()-1);

    std::string short_filename;
    if(ending != "json")
    {
        for(std::size_t kk=0; kk < (split.size()-1); kk++)
        {
            short_filename.append(split[kk]);
            if(kk != (split.size()-2)){
                short_filename.append(".");
            }
        }
    }
    else
    {
        short_filename = filename;
    }

    std::string uncompress_str;
    if(ending == "xz")
    {
        uncompress_str = std::format("xz -d {} ", filename);
        exec(uncompress_str.c_str());  // compress and delete original json 
    }
    else if(ending == "zst")
    {
        uncompress_str = std::format("zstd -d -q -f {}", filename);
        exec(uncompress_str.c_str());  // compress and delete original json  
    }
    else if(ending != "json")
    {
        throw std::runtime_error("Unknown filename ending");
    }
    
    std::fstream File;
    File.open(short_filename, std::ios::in);
    json Doc(json::parse(File));
    File.close();

    // remove temp json file if original is a compressed version
    if(ending == "xz" || ending == "zst")
    {
       std::remove(short_filename.c_str());
    }

    oper.width = Doc["width"];
    oper.terms.resize(0);
    
    if constexpr(std::is_same_v<U, QubitOperator>)
    {
        if(Doc["operator-type"] != "qubit"){
            throw std::runtime_error("JSON operator type does not match input");
        }
        for(JsonTerm item: Doc["terms"])
        {
            auto [a,b] = get<2>(item);
            OperatorTerm term = OperatorTerm(std::get<0>(item), std::get<1>(item), complex(a, b));
            oper.terms.push_back(term);
        }
    }
    else if constexpr(std::is_same_v<U, FermionicOperator>)
    {
       if(Doc["operator-type"] != "fermi"){
            throw std::runtime_error("JSON operator type does not match input");
        }
        for(JsonTerm item: Doc["terms"])
        {
            auto [a,b] = get<2>(item);
            FermionicTerm term = FermionicTerm(std::get<0>(item), std::get<1>(item), complex(a, b));
            oper.terms.push_back(term);
        }
    }
    else
    {
        throw std::runtime_error("Invalid input operator type");
    }
}
