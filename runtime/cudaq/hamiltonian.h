/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "operators.h"
#include "utils/cudaq_utils.h"
#include <cudaq/spin_op.h>

#include <complex>
#include <functional>
#include <random>
#include <vector>

namespace cudaq {

class hamiltonian {
protected:
  /// @brief We will store the independent terms of the
  /// Hamiltonian in each entry of this vector. The underlying
  /// operator data-type is a `cudaq::Operator`.
  std::vector<Operator> hamiltonian_terms;
  int hamiltonian_qubit_count;

public:
  /// @brief The constructor
  hamiltonian(int qubit_count) { set_qubit_count(qubit_count); };
  /// @brief The destructor
  virtual ~hamiltonian() = default;

  void append_hamiltonian_term(Operator term);
  void set_qubit_count(int qubit_count) {
    hamiltonian_qubit_count = qubit_count;
  };
};

} // namespace cudaq