/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/definition.h"
#include "cudaq/qis/state.h"

#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace cudaq {

Definition::Definition() = default;

// Convenience setter
void Definition::create_definition(const std::string &operator_id,
                                   std::vector<int> expected_dimensions,
                                   callback_function &&create) {
  m_id = operator_id;
  m_expected_dimensions = std::move(expected_dimensions);
  m_generator = std::move(create);
}

complex_matrix
Definition::generate_matrix(const std::vector<int> &degrees,
                            const std::vector<NumericType> &parameters) const {
  return m_generator(degrees, parameters);
}

Definition::~Definition() = default;
} // namespace cudaq
