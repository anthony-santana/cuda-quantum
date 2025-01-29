/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "helpers.cpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <set>

namespace cudaq {

// Degrees property
template <typename HandlerTy>
std::vector<int> product_operator<HandlerTy>::degrees() const {
  std::set<int> unique_degrees;
  for (const HandlerTy &term : this->get_terms()) {
    unique_degrees.insert(term.degrees.begin(), term.degrees.end());
  }
  // FIXME: SORT THE DEGREES
  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

template <typename HandlerTy>
cudaq::matrix_2 product_operatorr<HandlerTy>::to_matrix(
    std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters) {
  return m_evaluate(MatrixArithmetics(dimensions, parameters), dimensions,
                    parameters);
}

cudaq::matrix_2
_padded_op(cudaq::MatrixArithmetics arithmetics, cudaq::elementary_operator op,
           std::vector<int> degrees, std::map<int, int> dimensions,
           std::map<std::string, std::complex<double>> parameters) {
  /// Creating the tensor product with op being last is most efficient.
  std::vector<cudaq::matrix_2> padded;
  for (const auto &degree : degrees) {
    if (std::find(op.degrees.begin(), op.degrees.end(), degree) ==
            op.degrees.end(),
        degree) {
      padded.push_back(
          arithmetics.evaluate(cudaq::elementary_operator::identity(degree))
              .matrix());
    }
    matrix_2 mat = op.to_matrix(dimensions, parameters);
    padded.push_back(mat);
  }
  /// FIXME: This directly uses cudaq::kronecker instead of the tensor method.
  /// I need to double check to make sure this gives the equivalent behavior
  /// to the method used in python.
  return cudaq::kronecker(padded.begin(), padded.end());
  ;
}

template <typename HandlerTy>
cudaq::matrix_2 product_operator<HandlerTy>::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) {
  std::set<int> noncanon_set;
  for (const auto &op : m_elementary_ops) {
    for (const auto &degree : op.degrees) {
      noncanon_set.insert(degree);
    }
  }
  std::vector<int> noncanon_degrees(noncanon_set.begin(), noncanon_set.end());

  // Calculate the total dimensions of the Hilbert space to create our
  // identity matrix.
  auto full_hilbert_size = 1;
  for (const auto [degree, dimension] : dimensions)
    full_hilbert_size *= dimension;
  cudaq::matrix_2 result(full_hilbert_size, full_hilbert_size);
  // If this product operator consists only of scalar operator terms,
  // we will avoid all of the below logic and just return the scalar value
  // stored in an identity matrix spanning the full Hilbert space of the
  // provided `dimensions`.
  if (m_elementary_ops.size() > 0) {
    if (pad_terms) {
      // Sorting the degrees to avoid unnecessary permutations during the
      // padding.
      std::set<int> noncanon_set;
      for (const auto &op : m_elementary_ops) {
        for (const auto &degree : op.degrees) {
          noncanon_set.insert(degree);
        }
      }
      auto degrees = _OperatorHelpers::canonicalize_degrees(noncanon_degrees);
      auto evaluated =
          EvaluatedMatrix(degrees, _padded_op(arithmetics, m_elementary_ops[0],
                                              degrees, dimensions, parameters));

      for (auto op_idx = 1; op_idx < m_elementary_ops.size(); ++op_idx) {
        auto op = m_elementary_ops[op_idx];
        if (op.degrees.size() != 1) {
          auto padded_op_to_print =
              _padded_op(arithmetics, op, degrees, dimensions, parameters);
          auto padded_mat =
              EvaluatedMatrix(degrees, _padded_op(arithmetics, op, degrees,
                                                  dimensions, parameters));
          evaluated = arithmetics.mul(evaluated, padded_mat);
        }
      }
      result = evaluated.matrix();
    } else {
      auto evaluated = arithmetics.evaluate(m_elementary_ops[0]);
      for (auto op_idx = 1; op_idx < m_elementary_ops.size(); ++op_idx) {
        auto op = m_elementary_ops[op_idx];
        auto mat = op.to_matrix(dimensions, parameters);
        evaluated =
            arithmetics.mul(evaluated, EvaluatedMatrix(op.degrees, mat));
      }
      result = evaluated.matrix();
    }
  } else {
    result = cudaq::matrix_2::identity(full_hilbert_size);
  }
  // We will merge all of the scalar values stored in `m_scalar_ops`
  // into a single scalar value.
  std::cout << "\n merging the scalars in `product_operator::m_evaluate` \n";
  std::complex<double> merged_scalar = 1.0+0.0j;
  std::cout << "\n number of scalar ops to merge = " << m_scalar_ops.size() << "\n";
  for (auto &scalar : m_scalar_ops) {
    std::cout << "\n merging in " << scalar.m_name << "\n";
    merged_scalar *= scalar.evaluate(parameters);
  }
  return merged_scalar * result;
}

} // namespace cudaq