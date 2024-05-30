/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "utils/cudaq_utils.h"
#include <cudaq/spin_op.h>

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

  /// @brief Keeps track if the child type's basis term
  /// is a  composite of other `cudaq::spin_op`'s.
  bool isComposite;

  /// @brief We will store away the `cudaq::spin_op` that is
  /// being multiplied by our coefficient/s.
  spin_op basis_operator;
  /// @brief We will store away the single coefficient or
  /// the vector of coefficients away in the coefficient
  /// data-type.
  OperatorCoefficient coefficient;
  /// @brief Define the qubit or qubit/s that this operator
  /// acts upon.
  int qubit;
  std::vector<int> qubits;

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
    qubit = qubit_index;
  }

  bool isComposite = false;

  void set_basis_operator(int qubit_index) override {
    basis_operator = cudaq::spin::x(qubit_index);
  };

  Operator *operator*(const Operator &other) {
    // if (coefficient.isConstant && other.coefficient.isConstant) {
    // If these coefficients are both constant, then they
    // should be baked into the actual `spin_op`, so we just
    // need to multiply them directly.
    spin_op new_op = basis_operator * other.basis_operator;
    std::vector<int> qubit_indices = {qubit, other.qubit};
    auto composite_op = Operator(qubit_indices);
    composite_op.from_operators(basis_operator, other.basis_operator);
    // }
    return composite_op;
  }
};

class Y : public Operator {
public:
  Y(int qubit_index) : Operator(qubit_index) {
    set_basis_operator(qubit_index);
    qubit = qubit_index;
  }

  bool isComposite = false;

  void set_basis_operator(int qubit_index) override {
    basis_operator = cudaq::spin::y(qubit_index);
  };

  Operator *operator*(const Operator &other) {

    // if (coefficient.isConstant && other.coefficient.isConstant) {
    // If these coefficients are both constant, then they
    // should be baked into the actual `spin_op`, so we just
    // need to multiply them directly.
    spin_op new_op = basis_operator * other.basis_operator;
    //
    std::vector<int> qubit_indices = {}
    // }
    // then create and return a composite operator
    return 0;
  }
};

class Z : public Operator {
public:
  Z(int qubit_index) : Operator(qubit_index) {
    set_basis_operator(qubit_index);
    qubit = qubit_index;
  }

  bool isComposite = false;

  void set_basis_operator(int qubit_index) override {
    basis_operator = cudaq::spin::z(qubit_index);
  };

  Operator *operator*(const Operator &other) {

    // if (coefficient.isConstant && other.coefficient.isConstant) {
    // If these coefficients are both constant, then they
    // should be baked into the actual `spin_op`, so we just
    // need to multiply them directly.
    spin_op new_op = basis_operator * other.basis_operator;
    //
    std::vector<int> qubit_indices = {}

    // }
    // then create and return a composite operator
    return 0;
  }
};

class I : public Operator {
public:
  I(int qubit_index) : Operator(qubit_index) {
    set_basis_operator(qubit_index);
    qubit = qubit_index;
  }

  bool isComposite = false;

  void set_basis_operator(int qubit_index) override {
    basis_operator = cudaq::spin::i(qubit_index);
  };

  Operator *operator*(const Operator &other) {

    // if (coefficient.isConstant && other.coefficient.isConstant) {
    // If these coefficients are both constant, then they
    // should be baked into the actual `spin_op`, so we just
    // need to multiply them directly.
    spin_op new_op = basis_operator * other.basis_operator;
    //
    std::vector<int> qubit_indices = {}

    // }
    // then create and return a composite operator
    return 0;
  }
};

/// @brief Stores an operator term that is the tensor
/// product between operators on different qubits.
/// This type will only ever be created dynamically,
/// where someone adds a term like:
///      ``` 1.0 * (X(0) * X(1) * X(2) * ...); ```
/// In this case, individual `cudaq::operator`'s are created
/// when the `X(i)` constructors are called. The multiplication
class Composite : public Operator {
public:
  /// Since this will represent terms across multiple qubits,
  /// we will have to
  Composite(std::vector<int> indices) : Operator(indices) { qubits = indices; }

  bool isComposite = true;

  /// Can accept a pair of other `cudaq::Operator`'s
  /// that we'd like to join together.
  template <typename OpTypeA, typename OpTypeB>
  void from_operators(OpTypeA opA, OpTypeB opB) {
    basis_operator = opA.basis_operator * opB.basis_operator;
  };
};

} // namespace cudaq
