#include "cudaq/expressions.h"

#include "common/EigenDense.h"

#include <iostream>

namespace cudaq {

// implement everything without `_evaluate` first

// OperatorSum(std::vector<ProductOperator> &terms) {
//   m_terms = terms;
// }

// TODO:
// (1) Elementary Operators
// (2) Scalar Operators

Definition::Definition(){};
Definition::~Definition(){};

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
    // Issue: we need a capture lambda here, but we want to store
    // this as a std::function member on the Definition class. This
    // is explicitly not allowed, however, so I will need to think of
    // a workaround.
    auto func = [&](std::vector<int> none,
                    std::vector<std::complex<double>> _none) {
      auto mat = complex_matrix(degree, degree);
      // Build up the identity matrix.
      for (std::size_t i = 0; i < degree; i++) {
        mat(i, i) = 1.0 + 0.0j;
      }
      std::cout << "dumping the complex mat: \n";
      mat.dump();
      std::cout << "\ndone\n";
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
    auto func = [&](std::vector<int> none,
                    std::vector<std::complex<double>> _none) {
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
                              std::vector<std::complex<double>> parameters) {
  return m_ops[id].m_generator(degrees, parameters);
}

// delete me
complex_matrix ElementaryOperator::to_matrix() {
  return m_ops[id].m_generator({}, {});
}

} // namespace cudaq