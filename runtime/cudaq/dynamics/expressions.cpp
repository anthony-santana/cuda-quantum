#include "cudaq/expressions.h"
#include "cudaq/operator_utils.h"
#include "common/EigenDense.h"

#include <iostream>

namespace cudaq {
// OperatorSum class implementation

OperatorSum::OperatorSum(const std::vector<ProductOperator> &terms) {
  if (terms.empty()) {
    _terms.push_back(ProductOperator({ScalarOperator::const(0)}));
  } else {
    _terms = terms;
  }
}

std::vector<std::tuple<ScalarOperator, ElementaryOperator>> OperatorSum::canonicalize_product(const ProductOperator &prod) const {
  std::vector<std::tuple<ScalarOperator, ElementaryOperator>> canonicalized_terms;

  std::vector<int> all_degrees;
  std::vector<ScalarOperator> scalars;
  std::vector<ElementaryOperator> non_scalars;

  for (const auto &op : prod.get_operators()) {
    if (auto scalar_op = dynamic_cast<const ScalarOperator*>(&op)) {
      scalars.push_back(*scalar_op);
    } else {
      non_scalars.push_back(static_cast<ElementaryOperator>(op));
      all_degrees.insert(all_degrees.end(), op.get_degrees().begin(), op.get_degrees().end());
    }
  }

  if (all_degrees.size() == std::set<int>(all_degrees.begin(), all_degrees.end()).size()) {
    std::sort(non_scalars.begin(), non_scalars.end(), [](const ElementaryOperator &a, const ElementaryOperator &b)) {
      return a.get_degrees() < b.get_degrees();
    }
  }

  for (size_t i = 0; std::min(scalars.size(), non_scalars.size()); i++) {
    canonicalized_terms.push_back(std::make_tuple(scalars[i], non_scalars[i]));
  }

  return canonicalized_terms;
}

std::vector<std::tuple<ScalarOperator, ElementaryOperator>> OperatorSum::_canonical_terms() const {
  std::vector<std::tuple<ScalarOperator, ElementaryOperator>> terms;
  for (const auto &term : _terms) {
    auto canonicalized = canonicalize_product(term);
    terms.insert(terms.end(), canonicalized.begin(), canonicalized.end());
  }

  std::sort(terms.begin(), terms.end(), [](const auto &a, const auto &b) {
    return std::to_string(ProductOperator(a)) < std::to_string(ProductOperator(b));
  });

  return terms;
}

OperatorSum OperatorSum::canonicalize() const {
  std::vector<ProductOperator> canonical_terms;
  for (const auto &term : _canonical_terms()) {
    canonical_terms.push_back(ProductOperator(term));
  }

  return OperatorSum(canonical_terms);
}

bool OperatorSum::operator==(const OperatorSum &other) const {
  return _canonical_terms() == other._canonical_terms();
}

// Degrees property
std::vector<int> OperatorSum::degrees() const {
  std::set<int> unique_degrees;
  for (const auto &term : _terms) {
    for (const auto &op : term.get_operators()) {
      unique_degrees.insert(op.get_degrees().begin(), op.get_degrees().end());
    }
  }

  return std::vector<int>(unique_degrees.begin(), unique_degrees.end());
}

// Parameters property
std::map<std::string, std::string> OperatorSum::parameters() const {
  std::map<std::string, std::string> param_map;
  for (const auto &term : _terms) {
    for (const auto &op : term.get_operators()) {
      auto op_params = op.parameters();
      param_map.insert(op_params.begin(), op.params.end());
    }
  }

  return param_map;
}

// Check if all terms are spin operators
bool OperatorSum::_is_spinop() const {
  return std::all_of(_terms.begin(), _terms.end(), [](const ProductOperator &term) {
    return std::all_of(term.get_operators().begin(), term.get_operators().end(), [](const Operator &op) {
      return op.is_spinop();
    });
  });
}

// Arithmetic operators
OperatorSum OperatorSum::operator+(const OperatorSum &other) const {
  std::vector<ProductOperator> combined_terms = _terms;
  combined_terms.insert(combined_terms.end(), other._terms.begin(), other._terms.end());
  return OperatorSum(combined_terms);
}

OperatorSum OperatorSum::operator-(const OperatorSum &other) const {
  return *this + (-1 * other);
}

OperatorSum OperatorSum::operator+=(const OperatorSum &other) {
  *this = *this + other;
  return *this;
}

OperatorSum OperatorSum::operator-=(const OperatorSum &other) {
  *this = *this - other;
  return *this;
}

OperatorSum OperatorSum::operator*(const OperatorSum &other) const {
  std::vector<ProductOperator> product_terms;
  for (const auto &self_term : _terms) {
    for (const auto &other_term : other._terms) {
      product_terms.push_back(self_term * other_term);
    }
  }

  return OperatorSum(product_terms);
}

OperatorSum OperatorSum::operator/(const OperatorSum &other) const {
  std::vector<ProductOperator> divided_terms;
  for (const auto &term : _terms) {
    divided_terms.push_back(term / other);
  }

  return OperatorSum(divided_terms);
}

// OperatorSum + ScalarOperator
OperatorSum OperatorSum::operator+(const ScalarOperator &other) const {
  std::vector<ProductOperator> combined_terms = _terms;
  combined_terms.push_back(ProductOperator({other}));
  return OperatorSum(combined_terms);
}

OperatorSum OperatorSum::operator-(const ScalarOperator &other) const {
  return *this + (-1 * other);
}

OperatorSum OperatorSum::operator+=(const ScalarOperator &other) {
  *this = *this + other;
  return *this;
}

OperatorSum OperatorSum::operator-=(const ScalarOperator &other) {
  *this = *this - other;
  return *this;
}

// OperatorSum + ProductOperator
OperatorSum OperatorSum::operator+(const ProductOperator &other) const {
  std::vector<ProductOperator> combined_terms = _terms;
  combined_terms.push_back(other);
  return OperatorSum(combined_terms);
}

OperatorSum OperatorSum::operator-(const ProductOperator &other) const {
  return *this + (-1 * other);
}

OperatorSum OperatorSum::operator+=(const ProductOperator &other) {
  *this = *this + other;
  return *this;
}

OperatorSum OperatorSum::operator-=(const ProductOperator &other) {
  *this = *this - other;
  return *this;
}

// OperatorSum + ProductOperator
OperatorSum OperatorSum::operator+(const ElementaryOperator &other) const {
  std::vector<ProductOperator> combined_terms = _terms;
  combined_terms.push_back(ProductOperator({other}));
  return OperatorSum(combined_terms);
}

OperatorSum OperatorSum::operator-(const ElementaryOperator &other) const {
  return *this + (-1 * other);
}

OperatorSum OperatorSum::operator+=(const ElementaryOperator &other) {
  *this = *this + other;
  return *this;
}

OperatorSum OperatorSum::operator-=(const ElementaryOperator &other) {
  *this = *this - other;
  return *this;
}

complex_matrix OperatorSum::to_matrix(const std::map<int, int> &dimensions, const std::map<std::string, double> &params) const {

}

std::string OperatorSum::to_string() const {
  std::string result;
  for (const auto &term : _terms) {
    result += term.to_string() + " + ";
  }
  // Remove last " + "
  if (!result.empty())
    result.pop_back();
  return result;
}

// implement everything without `_evaluate` first

// OperatorSum(std::vector<ProductOperator> &terms) {
//   m_terms = terms;
// }

// TODO:
// (1) Elementary Operators
// (2) Scalar Operators

ElementaryOperator::ElementaryOperator(std::string operator_id,
                                       std::vector<int> degrees)
    : id(operator_id), degrees(degrees) {}

ElementaryOperator ElementaryOperator::identity(int degree) {
  std::string op_id = "identity";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  // NOTE: I don't actually think I need this if here because this
  // is a static method that creates a new ElementaryOperator (which
  // is what's being checked now) anyways.
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::vector<int> none, std::vector<Parameter> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      auto mat = complex_matrix(degree, degree);
      // Build up the identity matrix.
      for (std::size_t i = 0; i < degree; i++) {
        mat(i, i) = 1.0 + 0.0 * 'j';
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "done\n\n";
      return mat;
    };
    op.define(op_id, degrees, func);
  }
  return op;
}

ElementaryOperator ElementaryOperator::zero(int degree) {
  std::string op_id = "zero";
  std::vector<int> degrees = {degree};
  auto op = ElementaryOperator(op_id, degrees);
  if (op.m_ops.find(op_id) == op.m_ops.end()) {
    auto func = [&](std::vector<int> none, std::vector<Parameter> _none) {
      // Need to set the degree via the op itself because the
      // argument to the outer function goes out of scope when
      // the user invokes this later on via, e.g, `to_matrix()`.
      auto degree = op.degrees[0];
      auto mat = complex_matrix(degree, degree);
      mat.set_zero();
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
      return mat;
    };
    op.define(op_id, degrees, func);
  }
  return op;
}

complex_matrix
ElementaryOperator::to_matrix(std::vector<int> degrees,
                              std::vector<Parameter> parameters) {
  ReturnType result = m_ops[id].m_generator(degrees, parameters);

  if (std::holds_alternative<complex_matrix>(result)) {
    // Move the complex_matrix from the variant, which avoids copying
    return std::move(std::get<complex_matrix>(result));
  } else {
    // if it's a scalar, convert the scalar to a 1x1 matrix
    std::complex<double> scalar = std::get<std::complex<double>>(result);

    cudaq::complex_matrix scalar_matrix(1, 1);
    scalar_matrix(0, 0) = scalar;

    return scalar_matrix;
  }
}

ScalarOperator::ScalarOperator(const ScalarOperator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}
ScalarOperator::ScalarOperator(ScalarOperator &other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}
ScalarOperator::ScalarOperator(ScalarOperator &&other)
    : generator(other.generator), m_constant_value(other.m_constant_value) {}

OperatorSum ElementaryOperator::operator+(const ElementaryOperator &other) const {
  std::map<std::string, std::string> merged_params = aggregate_parameters(this->m_ops, other.m_ops);

  auto merged_func = [this, other](std::vector<int> degrees, std::vector<VariantArg> parameters) -> complex_matrix {
    complex_matrix result1 = this->func(degrees, parameters);
    complex_matrix result2 = other.func(degrees, parameters);

    for (size_t i = 0; i < result1.rows(); i++) {
      for (size_t j = 0; j < result1.cols(); j++) {
        result1(i, j) += result2(i, j);
      }
    }

    return result1;
  };

  return OperatorSum(merged_func);
}

/// @FIXME: The below function signature can be updated once
/// we support generalized function arguments.
/// @brief Constructor that just takes and returns a complex double value.
ScalarOperator::ScalarOperator(std::complex<double> value) {
  m_constant_value = value;
  auto func = [&](std::vector<std::complex<double>> _none) {
    return m_constant_value;
  };
  generator = scalar_callback_function(func);
}

std::complex<double>
ScalarOperator::evaluate(std::vector<std::complex<double>> parameters) {
  return generator(parameters);
}

// Arithmetic Operations.
ScalarOperator operator+(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) +
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator+(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) +
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator-(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) -
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator-(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) -
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator*(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) *
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator*(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) *
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator/(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[0].evaluate(selfParams) /
           returnOperator._operators_to_compose[1].evaluate({});
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

ScalarOperator operator/(std::complex<double> other, ScalarOperator &self) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);

  // Create an operator that we will store the result in and return to
  // the user.
  ScalarOperator returnOperator;

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  returnOperator._operators_to_compose.push_back(self);
  returnOperator._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return returnOperator._operators_to_compose[1].evaluate({}) /
           returnOperator._operators_to_compose[0].evaluate(selfParams);
  };

  returnOperator.generator = scalar_callback_function(newGenerator);
  return returnOperator;
}

void operator+=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) +
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator-=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) -
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator*=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) *
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

void operator/=(ScalarOperator &self, std::complex<double> other) {
  // Create an operator for the complex double value.
  auto otherOperator = ScalarOperator(other);
  // Need to move the existing generating function to a new
  // operator so that we can modify the generator in `self` in-place.
  ScalarOperator copy(self);

  // Store the previous generator functions in the new operator.
  // This is needed as the old generator functions would effectively be
  // lost once we leave this function scope.
  self._operators_to_compose.push_back(copy);
  self._operators_to_compose.push_back(otherOperator);

  /// FIXME: For right now, we will merge the arguments vector into one larger
  /// vector.
  // I think it should ideally take multiple arguments, however, in the order
  // that the arithmetic was applied. I.e, allow someone to keep each vector for
  // each generator packed up individually instead of concatenating them.
  // So if they had two generator functions that took parameter vectors, now
  // they would have two arguments to this new generator function.
  auto newGenerator = [&](std::vector<std::complex<double>> selfParams) {
    return self._operators_to_compose[0].evaluate(selfParams) /
           self._operators_to_compose[1].evaluate({});
  };

  self.generator = scalar_callback_function(newGenerator);
}

} // namespace cudaq