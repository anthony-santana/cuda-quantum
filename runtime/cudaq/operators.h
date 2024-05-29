/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cudaq/spin_op.h>
#include "utils/cudaq_utils.h"


namespace cudaq {

/// @brief A data class used to keep track of the
/// coefficients to a `cudaq::Operator`. 
class OperatorCoefficient {

public:
  /// @brief The constructor
  OperatorCoefficient() = default;
  /// @brief The destructor
  virtual ~OperatorCoefficient() = default;

};

class Operator {

protected:
  /// @brief We will store away the `cudaq::spin_op` that is
  /// being multiplied by our coefficient/s.
  spin_op basis_term;
  /// @brief We will store away the single coefficient or
  /// the vector of coefficients away in the coefficient
  /// data-type.
  OperatorCoefficient coefficient;

public:
  /// @brief The constructor
  Operator() = default;
  /// @brief The destructor
  virtual ~Operator() = default;
  
};

} // namespace cudaq

