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

matrix_2 zero_projection() {
  auto result = matrix_2(2,2);
  result[{0,0}] = 1.0+0.0j;
  return result;
}

matrix_2 one_projection() {
  auto result = matrix_2(2,2);
  result[{1,1}] = 1.0+0.0j;
  return result;
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

  // /// Non-parameterized gates.
  // {
  //   auto kernel = []() {
  //     cudaq::qvector qubit(1);
  //     h(qubit[0]);
  //     x(qubit[0]);
  //     y(qubit[0]);
  //     z(qubit[0]);
  //     s(qubit[0]);
  //     sdg(qubit[0]);
  //     t(qubit[0]);
  //     tdg(qubit[0]);
  //   };

  //   auto want_unitary = utils::h() * utils::x() * utils::y() * utils::z() *
  //                       utils::s() * utils::sdg() * utils::t() * utils::tdg();
  //   auto got_unitary = cudaq::get_unitary(kernel);

  //   utils::checkEqual(want_unitary, got_unitary);
  // }
  
  // // /// FIXME:
  // // /// Adjoint gates.
  // // {
  // //   auto kernel = []() {
  // //     cudaq::qvector qubit(1);
  // //     h<cudaq::adj>(qubit[0]);
  // //     x<cudaq::adj>(qubit[0]);
  // //     y<cudaq::adj>(qubit[0]);
  // //     z<cudaq::adj>(qubit[0]);
  // //     s<cudaq::adj>(qubit[0]);
  // //     t<cudaq::adj>(qubit[0]);
  // //   };

  // //   auto want_unitary = utils::h() * utils::x() * utils::y() * utils::z() *
  // //                       utils::s() * utils::t();
  // //   auto got_unitary = cudaq::get_unitary(kernel);

  // //   std::cout << "\n got = \n" << got_unitary.dump() << "\n";
  // //   std::cout << "\n want = \n" << want_unitary.dump() << "\n";

  // //   utils::checkEqual(want_unitary, got_unitary);
  // // }

  // /// Parameterized gates.
  // {
  //   auto kernel = [](std::vector<double> angles) {
  //     cudaq::qubit qubit;
  //     rx(angles[0], qubit);
  //     ry(angles[1], qubit);
  //     rz(angles[2], qubit);
  //     r1(angles[3], qubit);
  //     u3(angles[4], angles[5], angles[6], qubit);
  //   };

  //   // Magic values.
  //   std::vector<double> angles = {M_PI,       M_PI_2,     M_PI_4,    M_PI / 8.,
  //                                 M_PI / 16., M_PI / 32., M_PI / 64.};
  //   auto want_unitary = utils::rx(angles[0]) * utils::ry(angles[1]) *
  //                       utils::rz(angles[2]) * utils::r1(angles[3]) *
  //                       utils::u3(angles[4], angles[5], angles[6]);
  //   auto got_unitary = cudaq::get_unitary(kernel, angles);

  //   utils::checkEqual(want_unitary, got_unitary);
  // }

  // // Single CX gate.
  // {
  //   auto kernel = []() {
  //     cudaq::qvector qubits(2);
  //     cx(qubits[0], qubits[1]);
  //   };

  //   auto want_unitary = cudaq::matrix_2({1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.}, {4,4});
  //   auto got_unitary = cudaq::get_unitary(kernel);

  //   utils::checkEqual(want_unitary, got_unitary);
  // }

  // // Single CY gate.
  // {
  //   auto kernel = []() {
  //     cudaq::qvector qubits(2);
  //     cy(qubits[0], qubits[1]);
  //   };

  //   auto want_unitary = cudaq::matrix_2({1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.0-1j,0.,0.,0.0+1j,0.}, {4,4});
  //   auto got_unitary = cudaq::get_unitary(kernel);

  //   utils::checkEqual(want_unitary, got_unitary);
  // }

  // // Single CZ gate.
  // {
  //   auto kernel = []() {
  //     cudaq::qvector qubits(2);
  //     cz(qubits[0], qubits[1]);
  //   };

  //   auto want_unitary = cudaq::matrix_2({1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,-1.}, {4,4});
  //   auto got_unitary = cudaq::get_unitary(kernel);

  //   utils::checkEqual(want_unitary, got_unitary);
  // }

  // Single CX gate in a sub-space.
  {
    auto kernel = [](int qubit_count) {
      cudaq::qvector qubits(qubit_count);
      cx(qubits[0], qubits[1]);
      x(qubits);
    };

    auto want_unitary = cudaq::matrix_2({1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,1.,0.}, {4,4});
    int qubit_count = 3;
    for (auto i=0; i<(qubit_count-2); i++)
      want_unitary.kronecker_inplace(matrix_2::identity(2));
    
    auto x_operator = utils::x();
    for (auto i=0; i<(qubit_count-1); i++)
      x_operator.kronecker_inplace(matrix_2::identity(2));

    want_unitary *= x_operator;

    auto got_unitary = cudaq::get_unitary(kernel, qubit_count);

    std::cout << "\nwant = \n" << want_unitary.dump();
    std::cout << "\ngot = \n" << got_unitary.dump();

    // utils::checkEqual(want_unitary, got_unitary);
  }

  // // Toffoli gate.
  // {
  //   auto kernel = []() {
  //     cudaq::qvector qubits(3);
  //     x<cudaq::ctrl>(qubits[0], qubits[1], qubits[2]);
  //   };

  //   auto want_unitary = matrix_2::identity(8);
  //   want_unitary[{6,6}] = 0.0;
  //   want_unitary[{6,7}] = 1.0;
  //   want_unitary[{7,7}] = 0.0;
  //   want_unitary[{7,6}] = 1.0;
  //   auto got_unitary = cudaq::get_unitary(kernel);

  //   std::cout << "\n want = \n" << want_unitary.dump() << "\n";
  //   std::cout << "\n got = \n" << got_unitary.dump() << "\n";

  //   // utils::checkEqual(want_unitary, got_unitary);
  // }

}

/// Checking accuracy against state vector simulators.
#ifndef CUDAQ_BACKEND_DM
TEST(GetUnitaryTester, checkAgainstSimulator) {
    auto kernel = [](int qubit_count) {
      cudaq::qvector qubits(qubit_count);
      h(qubits[0]);
      for (auto i = 1; i < qubit_count; ++i) {
        cx(qubits[0], qubits[i]);
      }
      mz(qubits);
    };

    auto qubit_counts = {2,3,4,5};
    for (auto qubit_count : qubit_counts) {
      {

      // auto want_unitary = utils::h();
      auto got_unitary = cudaq::get_unitary(kernel, qubit_count);

      /// Check against the simulator.
      std::vector<std::complex<double>> vector(std::pow(2, qubit_count), 0.0);
      matrix_2 initial_state(vector, {std::pow(2, qubit_count),1});
      initial_state[{0,0}] = 1.0;

      auto final_state = got_unitary * initial_state;

      // std::cout << "\n final_state = \n" << final_state.dump() << "\n";
      // std::cout << "\n final_state size = \n" << final_state.get_size() << "\n";

      auto want_state = cudaq::get_state(kernel, qubit_count);
      want_state.dump();

      // std::cout << "\n got = \n" << got_unitary.dump() << "\n";

      }
    }
}
#endif

#endif