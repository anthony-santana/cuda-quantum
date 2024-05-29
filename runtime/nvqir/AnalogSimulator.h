/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"

namespace nvqir {

class AnalogSimulator {
public:
  /// @brief The constructor
  AnalogSimulator() = default;
  /// @brief The destructor
  virtual ~AnalogSimulator() = default;

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  virtual cudaq::observe_result observe(/*TODO*/) = 0;

};

/// @brief The AnalogSimulatorBase is the type that is meant to
/// be subclassed for new simulation strategies. The separation of
/// AnalogSimulator from AnalogSimulatorBase allows simulation sub-types
/// to specify the floating point precision for the simulation
class AnalogSimulatorBase : public AnalogSimulator {
protected:
  /// @brief Delegates to the differential equation solver to
  /// evolve the unitary time propagator for the system.
  void evolve_state(/*TODO*/) override {
    // pass
  }

public:
  cudaq::observe_result observe(/*TODO*/) override {
    // pass
  }
};
}; // namespace nvqir