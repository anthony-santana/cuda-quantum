/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/utils/tensor.h>

using namespace cudaq;

namespace utils {

template <typename Scalar = double>
static constexpr std::complex<Scalar> im = std::complex<Scalar>(0, 1.);

matrix_2 x() {
  return matrix_2({{0., 0.}, {1.0, 0.}, {1.0, 0.0}, {0., 0.}}, {2, 2});
}

matrix_2 y() {
  return matrix_2({{0., 0.}, {0.0, -1.0}, {0.0, 1.0}, {0., 0.}}, {2, 2});
}

matrix_2 z() {
  return matrix_2({{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {-1., 0.}}, {2, 2});
}

matrix_2 h() {
  matrix_2 result({1, 1, 1, -1}, {2, 2});
  return (1 / std::sqrt(2)) * result;
}

matrix_2 s() {
  return matrix_2({{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {0., 1.}}, {2, 2});
}

matrix_2 sdg() {
  return matrix_2({{1., 0.}, {0.0, 0.}, {0.0, 0.0}, {0., -1.}}, {2, 2});
}

matrix_2 t() {
  return matrix_2({{1., 0.},
                   {0.0, 0.},
                   {0.0, 0.0},
                   std::exp(im<double> * static_cast<double>(M_PI_4))},
                  {2, 2});
}

matrix_2 tdg() {
  return matrix_2({{1., 0.},
                   {0.0, 0.},
                   {0.0, 0.0},
                   std::exp(-im<double> * static_cast<double>(M_PI_4))},
                  {2, 2});
}

matrix_2 rx(std::complex<double> angle) {
  return matrix_2({std::cos(angle / 2.), -im<double> * std::sin(angle / 2.),
                   -im<double> * std::sin(angle / 2.), std::cos(angle / 2.)},
                  {2, 2});
}

matrix_2 ry(std::complex<double> angle) {
  return matrix_2({{std::cos(angle / 2.), -std::sin(angle / 2.),
                    std::sin(angle / 2.), std::cos(angle / 2.)}},
                  {2, 2});
}

matrix_2 rz(std::complex<double> angle) {
  return matrix_2({std::exp(-im<double> * angle / 2.), 0., 0.,
                   std::exp(im<double> * angle / 2.)},
                  {2, 2});
}

matrix_2 r1(std::complex<double> angle) {
  return matrix_2({1., 0., 0., std::exp(im<double> * angle)}, {2, 2});
}

matrix_2 u3(std::complex<double> theta, std::complex<double> phi,
            std::complex<double> lambda) {
  return matrix_2(
      {std::cos(theta / 2.),
       -std::exp(nvqir::im<double> * lambda) * std::sin(theta / 2.),
       std::exp(nvqir::im<double> * phi) * std::sin(theta / 2.),
       std::exp(nvqir::im<double> * (phi + lambda)) * std::cos(theta / 2.)},
      {2, 2});
}

void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b) {
  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      double a_val = a[{i, j}].real();
      double b_val = b[{i, j}].real();
      EXPECT_NEAR(a_val, b_val, 1e-8);
    }
  }
}
} // namespace utils

#ifndef CUDAQ_BACKEND_STIM

TEST(GetUnitaryTester, checkSimpleKernel) {

  /// Non-parameterized gates.
  {
    auto kernel = []() {
      cudaq::qvector qubit(1);
      h(qubit[0]);
      x(qubit[0]);
      y(qubit[0]);
      z(qubit[0]);
      s(qubit[0]);
      sdg(qubit[0]);
      t(qubit[0]);
      tdg(qubit[0]);
    };

    auto want_unitary = utils::h() * utils::x() * utils::y() * utils::z() *
                        utils::s() * utils::sdg() * utils::t() * utils::tdg();
    auto got_unitary = cudaq::get_unitary(kernel);

    utils::checkEqual(want_unitary, got_unitary);
  }

  /// Parameterized gates.
  {
    auto kernel = [](std::vector<double> angles) {
      cudaq::qubit qubit;
      rx(angles[0], qubit);
      ry(angles[1], qubit);
      rz(angles[2], qubit);
      r1(angles[3], qubit);
      u3(angles[4], angles[5], angles[6], qubit);
    };

    // Magic values.
    std::vector<double> angles = {M_PI,       M_PI_2,     M_PI_4,    M_PI / 8.,
                                  M_PI / 16., M_PI / 32., M_PI / 64.};
    auto want_unitary = utils::rx(angles[0]) * utils::ry(angles[1]) *
                        utils::rz(angles[2]) * utils::r1(angles[3]) *
                        utils::u3(angles[4], angles[5], angles[6]);
    auto got_unitary = cudaq::get_unitary(kernel, angles);

    utils::checkEqual(want_unitary, got_unitary);
  }
}

TEST(GetUnitaryTester, checkComplexKernel) {

  {
    auto kernel = [](int qubit_count) {
      cudaq::qvector qubits(qubit_count);
      h(qubits[0]);
      for (auto i = 1; i < qubit_count; ++i) {
        cx(qubits[0], qubits[i]);
      }
      h(qubits);
      h<cudaq::adj>(qubits[0]);
      mz(qubits);
    };

    int qubit_count = 4;

    auto want_unitary = utils::h();

    auto got_unitary = cudaq::get_unitary(kernel, qubit_count);

    utils::checkEqual(want_unitary, got_unitary);
  }

  /// TODO: Test the case of multiple control qubits
}


/// TODO: Test against a state vector simulation result.
TEST(GetUnitaryTester, checkAgainstSimulator) {
  {}
}

#endif