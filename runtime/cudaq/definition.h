/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qis/state.h"
#include "matrix.h"

#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace cudaq {

using NumericType = std::variant<int, double, std::complex<double>>;

using ReturnType = std::variant<complex_matrix, std::complex<double>>;

using Parameter =
    std::variant<NumericType, std::vector<NumericType>, std::string>;

// Limit the signature of the users callback function to accept a vector of ints
// for the degree of freedom dimensions, and a vector of complex doubles for the
// concrete parameter values.
using Func =
    std::function<ReturnType(std::vector<int>, std::vector<Parameter>)>;

class callback_function {
private:
  // The user provided callback function that takes the degrees of
  // freedom and a vector of complex parameters.
  Func _callback_func;

public:
  callback_function() = default;

  template <typename Callable>
  callback_function(Callable &&callable) {
    static_assert(std::is_invocable_r_v<ReturnType, Callable, std::vector<int>,
                                        std::vector<Parameter>>,
                  "Invalid callback function. Must have signature ReturnType("
                  "std::vector<int>, "
                  "std::vector<Parameter>)");
    _callback_func = std::forward<Callable>(callable);
  }

  ReturnType operator()(std::vector<int> degrees,
                        std::vector<Parameter> parameters) const {
    return _callback_func(std::move(degrees), std::move(parameters));
  }
};

/// FIXME: Too many issue with using this variant type for right now with
/// the scalar function. Should revisit in the near future to take a
/// `std::vector<Parameter>`.
using ScalarFunc =
    std::function<std::complex<double>(std::vector<std::complex<double>>)>;

// A scalar callback function does not need to accept the dimensions,
// therefore we will use a different function type for this specific class.
class scalar_callback_function : callback_function {
private:
  // The user provided callback function that takes a vector of parameters.
  ScalarFunc _callback_func;

public:
  scalar_callback_function() = default;

  template <typename Callable>
  scalar_callback_function(Callable &&callable) {
    static_assert(
        std::is_invocable_r_v<std::complex<double>, Callable,
                              std::vector<std::complex<double>>>,
        "Invalid callback function. Must have signature std::complex<double>("
        "std::vector<std::complex<double>>)");
    _callback_func = std::forward<Callable>(callable);
  }

  std::complex<double>
  operator()(std::vector<std::complex<double>> parameters) const {
    return _callback_func(std::move(parameters));
  }
};

/// @brief Object used to give an error if a Definition of an elementary
/// or scalar operator is instantiated by other means than the `define`
/// class method.
class Definition {
public:
  std::string m_id;
  // The user-provided generator function should take a variable number of
  // complex doubles for the parameters. It should return a
  // `cudaq::complex_matrix` type representing the operator matrix.
  callback_function m_generator;

  // Constructor.
  Definition();

  // Destructor.
  ~Definition();

  // Convenience setter. May be able to just move this to the constructor
  // now that we've restricted the function signature and no longer need
  // a template on this function.
  void create_definition(const std::string &operator_id,
                         std::vector<int> expected_dimensions,
                         callback_function &&create);

  // To call the generator function
  ReturnType generate_matrix(const std::vector<int> &degrees,
                             const std::vector<Parameter> &parameters) const;

private:
  // Member variables
  std::vector<int> m_expected_dimensions;
};
} // namespace cudaq
