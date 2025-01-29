/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"

#include <iostream>
#include <set>

namespace cudaq {

template <typename HandlerTy>
cudaq::matrix_2 operator_sum<HandlerTy>::m_evaluate(
    MatrixArithmetics arithmetics, std::map<int, int> dimensions,
    std::map<std::string, std::complex<double>> parameters, bool pad_terms) {

  std::cout << "\n evaluating operator sum \n";

  std::set<int> degrees_set;
  for (auto op : m_terms) {
    for (auto degree : op.degrees()) {
      std::cout << "degree = " << degree << "\n";
      degrees_set.insert(degree);
    }
  }
  std::vector<int> degrees(degrees_set.begin(), degrees_set.end());

  std::cout << "\n operator sum line 343 \n";

  // We need to make sure all matrices are of the same size to sum them up.
  auto paddedTerm = [&](product_operator term) {
    std::vector<int> op_degrees;
    for (auto op : term.m_elementary_ops) {
      for (auto degree : op.degrees)
        op_degrees.push_back(degree);
    }
    for (auto degree : degrees) {
      auto it = std::find(op_degrees.begin(), op_degrees.end(), degree);
      if (it == op_degrees.end()) {
        std::cout << "\n term count before \n" << term.term_count() << "\n";
        term *= elementary_operator::identity(degree);
        std::cout << "\n term count after \n" << term.term_count() << "\n";
      }
    }
    std::cout << "\n just checking \n" << term.to_matrix({{0,3}}).dump() << "\n";
    return term;
  };

  std::cout << "\n operator sum line 360 \n";

  auto sum = EvaluatedMatrix();
  if (pad_terms) {
 
    std::cout << "\n operator sum line 368 \n";
    sum = EvaluatedMatrix(degrees, paddedTerm(m_terms[0]).m_evaluate(arithmetics, dimensions,
                                              parameters, pad_terms));
    std::cout << "\n operator sum line 371 \n";
    for (auto term_idx = 1; term_idx < m_terms.size(); ++term_idx) {
      auto term = m_terms[term_idx];

      auto eval = paddedTerm(term).m_evaluate(arithmetics, dimensions,
                                              parameters, pad_terms);
      std::cout << "\n operator sum line 373 \n";
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
      std::cout << "\n operator sum line 375 \n";
    }
    std::cout << "\n operator sum line 377 \n";
  } else {
    std::cout << "\n operator sum line 379 \n";
    sum =
        EvaluatedMatrix(degrees, m_terms[0].m_evaluate(arithmetics, dimensions,
                                                       parameters, pad_terms));
    for (auto term_idx = 1; term_idx < m_terms.size(); ++term_idx) {
      auto term = m_terms[term_idx];
      auto eval =
          term.m_evaluate(arithmetics, dimensions, parameters, pad_terms);
      sum = arithmetics.add(sum, EvaluatedMatrix(degrees, eval));
    }
    std::cout << "\n operator sum line 389 \n";
  }
  return sum.matrix();
}

template <typename HandlerTy>
matrix_2 operator_sum<HandlerTy>::to_matrix(
    const std::map<int, int> &dimensions,
    const std::map<std::string, std::complex<double>> &parameters) const {
  /// FIXME: Not doing any conversion to spin op yet.
  return m_evaluate(MatrixArithmetics(dimensions, parameters), dimensions,
                    parameters);
}

// // std::string operator_sum::to_string() const {
// //   std::string result;
// //   // for (const auto &term : m_terms) {
// //   //   result += term.to_string() + " + ";
// //   // }
// //   // // Remove last " + "
// //   // if (!result.empty())
// //   //   result.pop_back();
// //   return result;
// // }

} // namespace cudaq