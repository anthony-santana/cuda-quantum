/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/state.h"
#include "definition.h"
#include "matrix.h"

#include <functional>
#include <iostream>
#include <map>
#include <set>

namespace cudaq {

class operator_sum;
class product_operator;
class scalar_operator;
class elementary_operator;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
class operator_sum {
private:
  std::vector<product_operator> m_terms;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  canonicalize_product(const product_operator &prod) const;

  std::vector<std::tuple<scalar_operator, elementary_operator>>
  _canonical_terms() const;

public:
  /// @brief Empty constructor that a user can aggregate terms into.
  operator_sum() = default;

  /// @brief Construct a `cudaq::operator_sum` given a sequence of
  /// `cudaq::product_operator`'s.
  /// This operator expression represents a sum of terms, where each term
  /// is a product of elementary and scalar operators.
  operator_sum(const std::vector<product_operator> &terms);

  operator_sum canonicalize() const;

  /// @brief The degrees of freedom that the oeprator acts on in canonical
  /// order.
  std::vector<int> degrees() const;

  std::map<std::string, std::complex<double>> parameters() const;

  bool _is_spinop() const;

  /// TODO: implement
  // template<typename TEval>
  // TEval _evaluate(OperatorArithmetics<TEval> &arithmetics) const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the paramter names to their concrete, complex
  /// values.
  complex_matrix
  to_matrix(const std::map<int, int> &dimensions,
            const std::map<std::string, double> &params = {}) const;

  // Arithmetic operators
  operator_sum operator+(const operator_sum &other) const;
  operator_sum operator-(const operator_sum &other) const;
  operator_sum operator*(const operator_sum &other) const;
  operator_sum operator/(const operator_sum &other) const;
  void operator+=(const operator_sum &other);
  void operator-=(const operator_sum &other);

  operator_sum operator+(const scalar_operator &other) const;
  operator_sum operator-(const scalar_operator &other) const;
  void operator+=(const scalar_operator &other);
  void operator-=(const scalar_operator &other);

  operator_sum operator+(const product_operator &other) const;
  operator_sum operator-(const product_operator &other) const;
  void operator+=(const product_operator &other);
  void operator-=(const product_operator &other);

  operator_sum operator+(const elementary_operator &other) const;
  operator_sum operator-(const elementary_operator &other) const;
  void operator+=(const elementary_operator &other);
  void operator-=(const elementary_operator &other);

  /// @brief Return the operator_sum as a string.
  std::string to_string() const;

  /// @brief  True, if the other value is an operator_sum with equivalent terms,
  /// and False otherwise. The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms blockwise; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const operator_sum &other) const;
};

// class product_operator : public operator_sum {
// public:
// product_operator() {};
// ~product_operator() {};
// };

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
class product_operator : public operator_sum {
public:
  product_operator() = default;
  ~product_operator() = default;

  /// @brief Constructor for an operator expression that represents a product
  /// of elementary operators.
  /// @arg atomic_operators : The operators of which to compute the product when
  ///                         evaluating the operator expression.
  product_operator(std::vector<elementary_operator> atomic_operators);

  /// @brief Constructor for an operator expression that represents a product
  /// of scalar and elementary operators.
  /// @arg atomic_operators : The operators of which to compute the product when
  ///                         evaluating the operator expression.
  /// @arg scalar_operator : The scalar operators of which to compute the
  /// product when evaluating the operator expression.
  product_operator(std::vector<elementary_operator> atomic_operators,
                   std::vector<scalar_operator> scalar_operators);

  // Arithmetic overloads against all other operator types.
  operator_sum operator+(std::complex<double> other);
  operator_sum operator-(std::complex<double> other);
  operator_sum operator+=(std::complex<double> other);
  operator_sum operator-=(std::complex<double> other);
  product_operator operator*(std::complex<double> other);
  product_operator operator*=(std::complex<double> other);
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator+=(operator_sum other);
  operator_sum operator-=(operator_sum other);
  product_operator operator*(operator_sum other);
  product_operator operator*=(operator_sum other);
  operator_sum operator+(scalar_operator other);
  operator_sum operator-(scalar_operator other);
  operator_sum operator+=(scalar_operator other);
  operator_sum operator-=(scalar_operator other);
  product_operator operator*(scalar_operator other);
  product_operator operator*=(scalar_operator other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  operator_sum operator+=(product_operator other);
  operator_sum operator-=(product_operator other);
  product_operator operator*(product_operator other);
  product_operator operator*=(product_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  operator_sum operator+=(elementary_operator other);
  operator_sum operator-=(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator*=(elementary_operator other);
  /// @brief True, if the other value is an operator_sum with equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms blockwise; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(product_operator other);

  /// @brief Return the `product_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `operator_sum` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the paramter names to their concrete, complex
  /// values.
  complex_matrix
  to_matrix(std::map<int, int> dimensions,
            std::map<std::string, std::complex<double>> parameters);

  /// @brief Creates a representation of the operator as a `cudaq::pauli_word`
  /// that can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word();

  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;

  /// @brief A map of the paramter names to their concrete, complex values.
  std::map<std::string, std::complex<double>> parameters;
};

class elementary_operator : public product_operator {
private:
  std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documentd for now, however.
  /// @brief Constructor.
  /// @arg operator_id : The ID of the operator as specified when it was
  /// defined.
  /// @arg degrees : the degrees of freedom that the operator acts upon.
  elementary_operator(std::string operator_id, std::vector<int> degrees);

  // Arithmetic overloads against all other operator types.
  operator_sum operator+(operator_sum other);
  operator_sum operator-(operator_sum other);
  operator_sum operator+=(operator_sum other);
  operator_sum operator-=(operator_sum other);
  product_operator operator*(operator_sum other);
  product_operator operator*=(operator_sum other);
  operator_sum operator+(scalar_operator other);
  operator_sum operator-(scalar_operator other);
  operator_sum operator+=(scalar_operator other);
  operator_sum operator-=(scalar_operator other);
  product_operator operator*(scalar_operator other);
  product_operator operator*=(scalar_operator other);
  operator_sum operator+(product_operator other);
  operator_sum operator-(product_operator other);
  operator_sum operator+=(product_operator other);
  operator_sum operator-=(product_operator other);
  product_operator operator*(product_operator other);
  product_operator operator*=(product_operator other);
  operator_sum operator+(elementary_operator other);
  operator_sum operator-(elementary_operator other);
  operator_sum operator+=(elementary_operator other);
  operator_sum operator-=(elementary_operator other);
  product_operator operator*(elementary_operator other);
  product_operator operator*=(elementary_operator other);
  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(elementary_operator other);

  /// @brief Return the `elementary_operator` as a string.
  std::string to_string() const;

  /// @brief Return the `elementary_operator` as a matrix.
  /// @arg  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  complex_matrix
  to_matrix(std::map<int, int> dimensions,
            std::map<std::string, std::complex<double>> parameters);

  // Predefined operators.
  static elementary_operator identity(int degree);
  static elementary_operator zero(int degree);
  static elementary_operator annihilate(int degree);
  static elementary_operator create(int degree);
  static elementary_operator momentum(int degree);
  static elementary_operator number(int degree);
  static elementary_operator parity(int degree);
  static elementary_operator position(int degree);
  /// FIXME:
  static elementary_operator squeeze(int degree,
                                     std::complex<double> amplitude);
  static elementary_operator displace(int degree,
                                      std::complex<double> amplitude);

  /// @brief Adds the definition of an elementary operator with the given id to
  /// the class. After definition, an the defined elementary operator can be
  /// instantiated by providing the operator id as well as the degree(s) of
  /// freedom that it acts on. An elementary operator is a parameterized object
  /// acting on certain degrees of freedom. To evaluate an operator, for example
  /// to compute its matrix, the level, that is the dimension, for each degree
  /// of freedom it acts on must be provided, as well as all additional
  /// parameters. Additional parameters must be provided in the form of keyword
  /// arguments. Note: The dimensions passed during operator evaluation are
  /// automatically validated against the expected dimensions specified during
  /// definition - the `create` function does not need to do this.
  /// @arg operator_id : A string that uniquely identifies the defined operator.
  /// @arg expected_dimensions : Defines the number of levels, that is the
  /// dimension,
  ///      for each degree of freedom in canonical (that is sorted) order. A
  ///      negative or zero value for one (or more) of the expected dimensions
  ///      indicates that the operator is defined for any dimension of the
  ///      corresponding degree of freedom.
  /// @arg create : Takes any number of complex-valued arguments and returns the
  ///      matrix representing the operator in canonical order. If the matrix
  ///      can be defined for any number of levels for one or more degree of
  ///      freedom, the `create` function must take an argument called
  ///      `dimension` (or `dim` for short), if the operator acts on a single
  ///      degree of freedom, and an argument called `dimensions` (or `dims` for
  ///      short), if the operator acts
  ///     on multiple degrees of freedom.
  template <typename Func>
  void define(std::string operator_id, std::map<int, int> expected_dimensions,
              Func create) {
    if (m_ops.find(operator_id) != m_ops.end()) {
      // todo: make a nice error message to say op already exists
      throw;
    }
    auto defn = Definition();
    defn.create_definition(operator_id, expected_dimensions, create);
    m_ops[operator_id] = defn;
  }

  // Attributes.

  /// @brief The number of levels, that is the dimension, for each degree of
  /// freedom in canonical order that the operator acts on. A value of zero or
  /// less indicates that the operator is defined for any dimension of that
  /// degree.
  std::map<int, int> expected_dimensions;
  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  /// @brief A map of the paramter names to their concrete, complex values.
  /// This will be enabled once we can handle generalized callback function
  /// arguments.
  /// @FIXME: Not needed until generalizing the function arguments.
  // std::map<std::string, std::complex<double>> parameters;
  std::string id;

  // /// @brief Creates a representation of the operator as `pauli_word` that
  // can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word ovveride();
};

class scalar_operator : public product_operator {
private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::complex<double> m_constant_value;

public:
  /// @brief Constructor that just takes a callback function with no
  /// arguments.

  scalar_operator(ScalarCallbackFunction &&create) {
    generator = ScalarCallbackFunction(create);
  }

  /// @brief Constructor that just takes and returns a complex double value.
  /// @NOTE: This replicates the behavior of the python `scalar_operator::const`
  /// without the need for an extra member function.
  scalar_operator(std::complex<double> value);

  // Arithmetic overloads against other operator types.
  scalar_operator operator+(scalar_operator other);
  scalar_operator operator-(scalar_operator other);
  scalar_operator operator*(scalar_operator other);
  scalar_operator operator/(scalar_operator other);
  /// TODO:
  scalar_operator pow(scalar_operator other);

  // Implement after scalar operator arithmetic:
  scalar_operator operator+(elementary_operator other);
  void operator+=(elementary_operator other);
  scalar_operator operator-(elementary_operator other);
  void operator-=(elementary_operator other);
  scalar_operator operator*(elementary_operator other);
  void operator*=(elementary_operator other);
  scalar_operator operator/(elementary_operator other);
  void operator/=(elementary_operator other);

  // Implement last:
  scalar_operator operator+(operator_sum other);
  scalar_operator operator-(operator_sum other);
  scalar_operator operator+=(operator_sum other);
  scalar_operator operator-=(operator_sum other);
  scalar_operator operator*(operator_sum other);
  scalar_operator operator*=(operator_sum other);
  scalar_operator operator/(operator_sum other);
  scalar_operator operator/=(operator_sum other);
  scalar_operator operator+(product_operator other);
  scalar_operator operator-(product_operator other);
  scalar_operator operator+=(product_operator other);
  scalar_operator operator-=(product_operator other);
  scalar_operator operator*(product_operator other);
  scalar_operator operator*=(product_operator other);
  scalar_operator operator/(product_operator other);
  scalar_operator operator/=(product_operator other);

  /// @brief Return the scalar operator as a concrete complex value.
  std::complex<double>
  evaluate(std::map<std::string, std::complex<double>> parameters);

  // /// @brief Returns true if other is a scalar operator with the same
  // /// generator.
  // bool operator==(scalar_operator other);

  /// @brief The function that generates the value of the scalar operator.
  /// The function can take a vector of complex-valued arguments
  /// and returns a number.
  ScalarCallbackFunction generator;

  // Only populated when we've performed arithmetic between various
  // scalar operators.
  std::vector<scalar_operator> _operators_to_compose;

  /// NOTE: We should revisit these constructors and remove any that have
  /// become unecessary as the implementation improves.
  scalar_operator() = default;
  // Copy constructor.
  scalar_operator(const scalar_operator &other);
  scalar_operator(scalar_operator &other);

  scalar_operator &operator=(scalar_operator &other);

  ~scalar_operator() = default;

  // REMOVEME: just using this as a temporary patch:
  std::complex<double> get_val() { return m_constant_value; };
};

scalar_operator operator+(scalar_operator self, std::complex<double> other);
scalar_operator operator-(scalar_operator self, std::complex<double> other);
scalar_operator operator*(scalar_operator self, std::complex<double> other);
scalar_operator operator/(scalar_operator self, std::complex<double> other);
scalar_operator operator+(std::complex<double> other, scalar_operator self);
scalar_operator operator-(std::complex<double> other, scalar_operator self);
scalar_operator operator*(std::complex<double> other, scalar_operator self);
scalar_operator operator/(std::complex<double> other, scalar_operator self);
void operator+=(scalar_operator &self, std::complex<double> other);
void operator-=(scalar_operator &self, std::complex<double> other);
void operator*=(scalar_operator &self, std::complex<double> other);
void operator/=(scalar_operator &self, std::complex<double> other);
void operator+=(scalar_operator &self, scalar_operator other);
void operator-=(scalar_operator &self, scalar_operator other);
void operator*=(scalar_operator &self, scalar_operator other);
void operator/=(scalar_operator &self, scalar_operator other);

} // namespace cudaq