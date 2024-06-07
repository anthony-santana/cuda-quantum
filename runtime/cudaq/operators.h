/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include "common/EigenSparse.h"

#include "utils/cudaq_utils.h"
#include <cudaq/spin_op.h>

namespace cudaq {

/// @brief A data class used to keep track of the
/// coefficients to a `cudaq::Operator`.
class OperatorCoefficient {
// protected:
//   // To make algebraic manipulations a bit simpler between
//   // coefficient vectors, we will work in Eigen under the
//   // hood.
//   Eigen::VectorXd m_coefficientValues;

public:
  /// @brief The constructors
  OperatorCoefficient();
  /// @brief The destructor
  virtual ~OperatorCoefficient() = default;

  // Making public for now while I prototype, but the eigen
  // vector should ultimately remain protected. Then we can
  // cast back to std vector to return values to user.
  Eigen::VectorXd m_coefficientValues;

  bool isVector = false;
  bool isConstant = true;

  void set_coefficient_values(std::vector<double> &values) {
    m_coefficientValues = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        values.data(), values.size());
    isVector = true;
    isConstant = false;
  }

  void operator*=(double value) {
    // If this coefficient is a constant value, all of this
    // multiplication will have already occured at the `spin_op`
    // level. Hence, we do nothing.
    // Otherwise, if it's a `float * vector`, we will scale the
    // vector accordingly. 
    if (!isConstant) {
      m_coefficientValues *= value;
    }
  }

  void operator*=(std::vector<double> &values) {
    auto cast_values = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        values.data(), values.size());
    // Should check these are the same length so no
    // Eigen errors bleed through to the user.
    m_coefficientValues *= cast_values;
  }

  void operator*=(OperatorCoefficient other) {
    m_coefficientValues *= other.m_coefficientValues;
  }

  // std::vector<double> get_coefficient_values() { /* TODO */ }

  Eigen::VectorXd get_raw_coefficient_values() { return m_coefficientValues; }
};

class Operator {
public:
  /// @brief The constructor
  Operator(int qubit_index);
  Operator(std::vector<int> &indices);
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
  /// REMOVEME: Will only be overrided by the `Composite` operator.
  // virtual void set_basis_operator(Operator op, ...) = 0;

  // Bakes any constant coefficient values directly into
  // the underlying `spin_op` so that we don't need to keep
  // track of it in the coefficient operator class.
  // Ex: `1.0 * X(0)`
  void operator*(const double coefficient_value) { basis_operator *= coefficient_value; }

  // If someone passes a vector of coefficients to multiply by,
  // we will store the vector in the `OperatorCoefficient`.
  // Ex: `{1.0, 2.0, ...} * X(0)`
  void operator*(std::vector<double> &coefficient_values) {
    coefficient.set_coefficient_values(coefficient_values);
  }

  // If someone multiplies one operator by another, e.g,
  // `X(0) * Y(1)`, we will create an operator of type
  // `Composite`. This will store the indices of any qubits
  // the composite term applies to, will multiply the underlying
  // basis operators, and will handle multiplying any constant
  // or time-dependent coefficient values.
  Operator *operator*(Operator &other) {

    // FIXME: Handle this if they are both on the same qubit (we
    // don't really need this vector in that case)
    std::vector<int> new_qubit_indices = {qubit, other.qubit};
    auto* composite_op = Operator(new_qubit_indices);

    // Multiply the underlying basis ops.
    spin_op new_basis_op = basis_operator * other.basis_operator;
    composite_op->from_composite_spin_op(new_basis_op);

    // Multiply the underlying coefficients.
    coefficient *= other.coefficient;
    
    return composite_op;
  }

  // Will only be overridden by the `Composite` child class.
  virtual void from_composite_spin_op(spin_op composite_op) = 0;
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
  Composite(std::vector<int> &indices) : Operator(indices) { qubits = indices; }

  bool isComposite = true;

  /// Can accept a pair of other `cudaq::Operator`'s
  /// that we'd like to join together.
  void from_composite_spin_op(spin_op composite_op) override {
    basis_operator = composite_op;
  }
};

} // namespace cudaq
