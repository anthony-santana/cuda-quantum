/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "matrix.h"
#include "definition.h"
#include <map>
#include <string>
#include <functional>
#include <vector>
#include <utility>
#include <variant>

namespace cudaq {
using NumericType = std::variant<int, double, std::complex<double>>;

using ReturnType = std::variant<complex_matrix, std::complex<double>>;

using VariantArg =
    std::variant<NumericType, std::vector<NumericType>, std::string>;

inline std::map<std::string, std::string> aggregate_parameters(
    const std::map<std::string, Definition> &param1,
    const std::map<std::string, Definition> &param2) {
    std::map<std::string, std::string> merged_map = param1;

    for (const auto &[key, value] : param2) {
        if (merged_map.find(key) != merged_map.end()) {
            // Combine the descriptions if key exists in both parameter sets
            merged_map[key] += "\n------\n" + value.m_id;
        } else {
            merged_map[key] = value;
        }
    }

    return merged_map;
}
}