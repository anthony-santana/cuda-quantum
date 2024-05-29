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
  /// @brief The constructors
  OperatorCoefficient();
  /// @brief The destructor
  virtual ~OperatorCoefficient() = default;

  bool isVector = false;
  bool isConstant = false;
  std::vector<double> coefficientValues;

  void set_value(std::vector<double> &values) {
    coefficientValues = values;
    isVector = true;
  }

};

class Operator {
public:
  /// @brief The constructor
  Operator(int qubit_index);
  Operator(std::vector<int> indices);
  /// @brief The destructor
  virtual ~Operator() = default;

  /// @brief We will store away the `cudaq::spin_op` that is
  /// being multiplied by our coefficient/s.
  spin_op basis_term;
  /// @brief We will store away the single coefficient or
  /// the vector of coefficients away in the coefficient
  /// data-type.
  OperatorCoefficient coefficient;

  virtual void set_basis_operator(int qubit) = 0;
  /// Will only be overrided by the `Composite` operator.
  virtual void set_basis_operator(Operator op, ...) = 0;

  void set_coefficient_value(double coefficient_value) {
    // pass
  }

  void set_coefficient_value(std::vector<double> &coefficient_values) {
    // pass
  }
  
};

class X : public Operator {
public:
  X(int qubit_index) : Operator(qubit_index) { 
    set_basis_operator(qubit_index);
  }

  void set_basis_operator(int qubit_index) override {
    basis_term = cudaq::spin::x(qubit_index);
  };

};

class Y : public Operator {
public:
  Y(int qubit_index) : Operator(qubit_index) { 
    set_basis_operator(qubit_index);
  }

  void set_basis_operator(int qubit_index) override {
    basis_term = cudaq::spin::y(qubit_index);
  };

};

class Z : public Operator {
public:
  Z(int qubit_index) : Operator(qubit_index) { 
    set_basis_operator(qubit_index);
  }

  void set_basis_operator(int qubit_index) override {
    basis_term = cudaq::spin::z(qubit_index);
  };

};


/// @brief Stores an operator term that is the tensor
/// product between operators on different qubits.
class Composite : public Operator {
public:

  /// Since this will represent terms across multiple qubits,
  /// we will have to 
  Composite(int qubit_index) : Operator(qubit_index) {}
  Composite(std::vector<int> indices) : Operator(indices) {}

  /// Can accept a variadic list of other `cudaq::Operator`'s
  /// that we'd like to join together.

  void set_basis_operator(Operator op, ...) override {
    // pass
    // basis_term = ;
  };

};

} // namespace cudaq

