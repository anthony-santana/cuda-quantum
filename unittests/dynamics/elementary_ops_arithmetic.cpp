/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/matrix.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>

/// NOTE: Not yet testing any of the matrix conversions. Just testing
/// the attributes of the output data type coming from the arithmetic.
/// These tests should be built upon to do actual numeric checks once
/// the implementations are complete.

TEST(ExpressionTester, checkPreBuiltElementaryOpsScalars) {

  auto function = [](std::map<std::string, std::complex<double>> parameters) {
    return parameters["value"];
  };

  // `elementary_operator + scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

   // `elementary_operator + scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

   // `elementary_operator - scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

   // `elementary_operator - scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

   // `elementary_operator * scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(1.0);

    auto product = self * other;
    auto reverse = other * self;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

   // `elementary_operator * scalar_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }
}

/// Prebuilt elementary ops against one another.
TEST(ExpressionTester, checkPreBuiltElementaryOpsSelf) {

  /// TODO: Check the output degrees attribute.

  // Addition, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.term_count() == 2);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.term_count() == 2);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.term_count() == 2);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.term_count() == 2);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(0);

    auto product = self * other;
    ASSERT_TRUE(product.term_count() == 2);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    auto other = cudaq::elementary_operator::create(1);

    auto product = self * other;
    ASSERT_TRUE(product.term_count() == 2);
  }
}

/// Testing arithmetic between elementary operators and operator
/// sums.
TEST(ExpressionTester, checkElementaryOpsAgainstOpSum) {

  /// `elementary_operator + operator_sum` and `operator_sum + elementary_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    /// Creating an arbitrary operator sum to work against.
    auto operator_sum = cudaq::elementary_operator::create(0) +
                        cudaq::elementary_operator::identity(1);

    auto got = self + operator_sum;
    auto reverse = operator_sum + self;

    ASSERT_TRUE(got.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  /// `elementary_operator - operator_sum` and `operator_sum - elementary_operator`
  {
    auto self = cudaq::elementary_operator::annihilate(0);
    /// Creating an arbitrary operator sum to work against.
    auto operator_sum = cudaq::elementary_operator::create(0) +
                        cudaq::elementary_operator::identity(1);

    auto got = self - operator_sum;
    auto reverse = operator_sum - self;

    ASSERT_TRUE(got.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  /// `operator_sum += elementary_operator`
  {
    auto operator_sum = cudaq::elementary_operator::create(0) +
                    cudaq::elementary_operator::identity(1);
    operator_sum += cudaq::elementary_operator::annihilate(0);

    ASSERT_TRUE(operator_sum.term_count() == 3);
  }

  /// `operator_sum -= elementary_operator`
  {
    auto operator_sum = cudaq::elementary_operator::create(0) +
                    cudaq::elementary_operator::identity(1);
    operator_sum -= cudaq::elementary_operator::annihilate(0);

    ASSERT_TRUE(operator_sum.term_count() == 3);
  }
}
