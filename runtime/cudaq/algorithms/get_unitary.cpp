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
  std::cout << "\n qubit_count = " << qubit_count << "\n";
  std::vector<std::size_t> qubit_indices(qubit_count);
  std::iota(qubit_indices.begin(), qubit_indices.end(), 0);

  /// Defining some helper matrices.
  auto identity_matrix = matrix_2::identity(2);
  matrix_2 zero_projection(2,2);
  zero_projection[{0,0}] = 1.0+0.0j;
  matrix_2 one_projection(2,2);
  one_projection[{1,1}] = 1.0+0.0j;

  /// Assuming a hilbert space containing only qubits.
  auto hilbert_space_dim = std::pow(2, qubit_count);

  /// Create a unitary matrix for the full Hilbert space
  /// that we will accumulate the gate operations into.
  auto unitary = matrix_2::identity(hilbert_space_dim);

  for (const auto &inst : trace) {
    auto controls = convertToIDs(inst.controls);
    auto targets = convertToIDs(inst.targets);
    auto instruction_matrix =
        matrix_2(nvqir::getGateByName<double>(
                     nvqir::stringToGateName(inst.name), inst.params),
                 {2, 2});

    if (!controls.empty()) {
      matrix_2 control_matrix(1, 1);
      control_matrix[{0, 0}] = 1.0;
      matrix_2 target_matrix(1, 1);
      target_matrix[{0, 0}] = 1.0;
      std::string control_list = "";
      std::string target_list = "";
      std::vector<matrix_2> control_matrices;
      std::vector<matrix_2> target_matrices;
      for (auto qubit : qubit_indices) {
        if (std::find(controls.begin(), controls.end(), qubit) != controls.end()) {
          control_matrix.kronecker_inplace(zero_projection);
          target_matrix.kronecker_inplace(one_projection);

          control_list.append("0");
          target_list.append("1");

          control_matrices.push_back(zero_projection);
          target_matrices.push_back(one_projection);
        }
        else if (std::find(targets.begin(), targets.end(), qubit) != targets.end()) {
          control_matrix.kronecker_inplace(identity_matrix);
          target_matrix.kronecker_inplace(instruction_matrix);

          control_list.append("I");
          target_list.append("U");

          control_matrices.push_back(identity_matrix);
          target_matrices.push_back(instruction_matrix);
        } else {
          control_matrix.kronecker_inplace(identity_matrix);
          target_matrix.kronecker_inplace(identity_matrix);

          control_list.append("I");
          target_list.append("I");

          control_matrices.push_back(identity_matrix);
          target_matrices.push_back(identity_matrix);
        }
      }
      // std::cout << "\ncontrol_matrix = \n" << control_matrix.dump() << "\n";
      // std::cout << "\ntarget_matrix = \n" << target_matrix.dump() << "\n";
      std::cout << "\ncontrol_list = \n" << control_list << "\n";
      std::cout << "\ntarget_list = \n" << target_list << "\n";
      // std::reverse(control_matrices.begin(), control_matrices.end());
      // std::reverse(target_matrices.begin(), target_matrices.end());
      // auto ctrl = cudaq::kronecker(control_matrices.begin(), control_matrices.end());
      // auto targ = cudaq::kronecker(target_matrices.begin(), target_matrices.end());
      unitary *= (control_matrix + target_matrix);
      // auto sum = ctrl + targ;
      // unitary *= sum;
    } else {
      matrix_2 matrix(1, 1);
      matrix[{0, 0}] = 1.0;
      for (auto qubit : qubit_indices) {
        if (std::find(targets.begin(), targets.end(), qubit) != targets.end()) {
          matrix.kronecker_inplace(instruction_matrix);
        } else {
          matrix.kronecker_inplace(identity_matrix);
        }
      }
      unitary *= matrix;
    }
  }
  return unitary;
}

} // namespace cudaq