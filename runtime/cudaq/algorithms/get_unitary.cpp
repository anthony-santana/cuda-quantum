/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/get_unitary.h"

#include <iostream>

std::vector<int> convertToIDs(const std::vector<cudaq::QuditInfo> &qudits) {
  std::vector<int> ids;
  ids.reserve(qudits.size());
  std::transform(qudits.cbegin(), qudits.cend(), std::back_inserter(ids),
                 [](auto &info) { return info.id; });
  return ids;
}

namespace cudaq {

matrix_2 cudaq::__internal__::get_unitary(const Trace &trace) {
  auto qubit_count = trace.getNumQudits();
  std::vector<std::size_t> qubit_indices(qubit_count);
  std::iota(qubit_indices.begin(), qubit_indices.end(), 0);

  /// Assuming a hilbert space containing only qubits.
  auto hilbert_space_dim = std::pow(2, qubit_count);
  auto unitary = matrix_2::identity(hilbert_space_dim);
  for (const auto &inst : trace) {
    auto instruction_matrix =
        matrix_2(nvqir::getGateByName<double>(
                     nvqir::stringToGateName(inst.name), inst.params),
                 {2, 2});
    auto identity_matrix = matrix_2::identity(2);

    if (!inst.controls.empty()) {
      matrix_2 zero_projection(2,2);
      zero_projection[{0,0}] = 1.0+0.0j;
      matrix_2 one_projection(2,2);
      one_projection[{1,1}] = 1.0+0.0j;
      
      // (projection_0 cross ID) + (projection_1 cross X)
      throw std::runtime_error("implementation not yet finished.");
    }

    matrix_2 matrix(1, 1);
    matrix[{0, 0}] = 1.0;
    auto controls = convertToIDs(inst.controls);
    auto targets = convertToIDs(inst.targets);
    for (auto qubit : qubit_indices) {
      if (std::find(targets.begin(), targets.end(), qubit) != targets.end()) {
        matrix.kronecker_inplace(instruction_matrix);
      // } else if (std::find(controls.begin(), controls.end(), qubit) != controls.end()) {
      //   matrix.kronecker_inplace(identity_matrix);
      } else {
        matrix.kronecker_inplace(identity_matrix);
      }
    }
    unitary *= matrix;
  }
  return unitary;
}

} // namespace cudaq