/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <gtest/gtest.h>

void checkEqual(cudaq::matrix_2 a,
                cudaq::matrix_2 b) {
  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      // EXPECT_NEAR(a[{i, j}].real(), b[{i, j}].real(), 1e-8);
      // FIXME
    }
  }
}

cudaq::matrix_2 zero_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++) {
    for (std::size_t j = 0; j < size; j++)
      mat[{i, j}] = 0.0 + 0.0j;
  }
  return mat;
}

cudaq::matrix_2 id_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = 1.0 + 0.0j;
  return mat;
}

cudaq::matrix_2 annihilate_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 create_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 position_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] =
        0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] =
        0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 momentum_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] =
        (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] =
        -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 number_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = static_cast<double>(i) + 0.0j;
  return mat;
}

cudaq::matrix_2 parity_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
  return mat;
}

// cudaq::matrix_2 displace_matrix(std::size_t size,
//                                       std::complex<double> amplitude) {
//   auto mat = cudaq::matrix_2(size, size);
//   for (std::size_t i = 0; i + 1 < size; i++) {
//     mat[{i + 1, i}] =
//         amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
//     mat[{i, i + 1}] = -1. * std::conj(amplitude) * (0.5 * 'j') *
//                         std::sqrt(static_cast<double>(i + 1)) +
//                     0.0 * 'j';
//   }
//   return mat.exp();
// }

TEST(ExpressionTester, checkPreBuiltElementaryOps) {
  std::vector<std::size_t> levels = {2, 3, 4, 5};

  // Keeping this fixed throughout.
  int degree_index = 0;

  // Identity operator.
  {
    for (auto level_count : levels) {
      // cudaq::operators::identity(int degree)
      auto id = cudaq::elementary_operator::identity(degree_index);
      auto got_id = id.to_matrix({{degree_index, level_count}}, {});
      auto want_id = id_matrix(level_count);
      checkEqual(want_id, got_id);
    }
  }

  // Zero operator.
  {
    for (auto level_count : levels) {
      auto zero = cudaq::elementary_operator::zero(degree_index);
      auto got_zero = zero.to_matrix({{degree_index, level_count}}, {});
      auto want_zero = zero_matrix(level_count);
      checkEqual(want_zero, got_zero);
    }
  }

  // Annihilation operator.
  {
    for (auto level_count : levels) {
      auto annihilate = cudaq::elementary_operator::annihilate(degree_index);
      auto got_annihilate =
          annihilate.to_matrix({{degree_index, level_count}}, {});
      auto want_annihilate = annihilate_matrix(level_count);
      checkEqual(want_annihilate, got_annihilate);
    }
  }

  // Creation operator.
  {
    for (auto level_count : levels) {
      auto create = cudaq::elementary_operator::create(degree_index);
      auto got_create = create.to_matrix({{degree_index, level_count}}, {});
      auto want_create = create_matrix(level_count);
      checkEqual(want_create, got_create);
    }
  }

  // Position operator.
  {
    for (auto level_count : levels) {
      auto position = cudaq::elementary_operator::position(degree_index);
      auto got_position = position.to_matrix({{degree_index, level_count}}, {});
      auto want_position = position_matrix(level_count);
      checkEqual(want_position, got_position);
    }
  }

  // Momentum operator.
  {
    for (auto level_count : levels) {
      auto momentum = cudaq::elementary_operator::momentum(degree_index);
      auto got_momentum = momentum.to_matrix({{degree_index, level_count}}, {});
      auto want_momentum = momentum_matrix(level_count);
      checkEqual(want_momentum, got_momentum);
    }
  }

  // Number operator.
  {
    for (auto level_count : levels) {
      auto number = cudaq::elementary_operator::number(degree_index);
      auto got_number = number.to_matrix({{degree_index, level_count}}, {});
      auto want_number = number_matrix(level_count);
      checkEqual(want_number, got_number);
    }
  }

  // Parity operator.
  {
    for (auto level_count : levels) {
      auto parity = cudaq::elementary_operator::parity(degree_index);
      auto got_parity = parity.to_matrix({{degree_index, level_count}}, {});
      auto want_parity = parity_matrix(level_count);
      checkEqual(want_parity, got_parity);
    }
  }

  // // // Displacement operator.
  // // {
  // //   for (auto level_count : levels) {
  // //     auto amplitude = 1.0 + 1.0j;
  // //     auto displace = cudaq::elementary_operator::displace(degree_index,
  // //     amplitude); auto got_displace = displace.to_matrix({{degree_index,
  // //     level_count}}, {}); auto want_displace =
  // displace_matrix(level_count,
  // //     amplitude); checkEqual(want_displace, got_displace);
  // //   }
  // // }

  // TODO: Squeeze operator.
}

// TEST(ExpressionTester, checkCustomElementaryOps) {
//   // pass
// }