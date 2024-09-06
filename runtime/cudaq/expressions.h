#include "cudaq/qis/state.h"
#include "matrix.h"

#include <functional>
#include <map>

namespace cudaq {

class OperatorSum;
class OperatorSum {};
class ProductOperator : public OperatorSum {};
class ScalarOperator : public ProductOperator {};

// Limit the signature of the users callback function to accept a vector of ints
// for the degree of freedom dimensions, and a vector of complex doubles for the
// concrete parameter values.
using Func = std::function<complex_matrix(std::vector<int>,
                                          std::vector<std::complex<double>>)>;

class callback_function {
private:
  // The user provided callback function that takes the degrees of
  // freedom and a vector of complex parameters.
  Func _callback_func;

public:
  callback_function() = default;

  template <typename Callable>
  callback_function(Callable &&callable) {
    static_assert(std::is_invocable_v<Callable, std::vector<int>,
                                      std::vector<std::complex<double>>>,
                  "Invalid callback function. Must have signature double(const "
                  "std::vector<int>, "
                  "std::vector<std::complex<double>>)");
    _callback_func = std::move(callable);
  }

  complex_matrix operator()(std::vector<int> degrees,
                            std::vector<std::complex<double>> parameters) {
    return _callback_func(degrees, parameters);
  }
};

/// @brief Object used to give an error if a Definition of an elementary
/// or scalar operator is instantiated by other means than the `define`
/// class method.
class Definition {
public:
  // Constructor.
  Definition();

  // Destructor.
  ~Definition();

  std::string m_id;
  std::vector<int> m_expected_dimensions;
  // The user-provided generator function should take a variable number of
  // complex doubles for the parameters. It should return a
  // `cudaq::complex_matrix` type representing the operator matrix.
  callback_function m_generator;

  // Convenience setter. May be able to just move this to the constructor
  // now that we've restricted the function signature and no longer need
  // a template on this function.
  void create_definition(std::string operator_id,
                         std::vector<int> expected_dimensions,
                         callback_function &&create) {
    // TODO: reproduce `with_dimension_check` from python ...
    m_id = operator_id;
    m_expected_dimensions = expected_dimensions;
    m_generator = callback_function(create);
  };
};

class ElementaryOperator : public ProductOperator {
private:
  std::map<std::string, Definition> m_ops;

public:
  // The constructor should never be called directly by the user:
  // Keeping it internally documentd for now, however.
  // / @brief Constructor.
  // / @arg operator_id : The ID of the operator as specified when it was
  // / defined.
  // / @arg degrees : the degrees of freedom that the operator acts upon.
  ElementaryOperator(std::string operator_id, std::vector<int> degrees);

  // Arithmetic overloads against all other operator types.
  OperatorSum operator+(OperatorSum other);
  OperatorSum operator-(OperatorSum other);
  OperatorSum operator+=(OperatorSum other);
  OperatorSum operator-=(OperatorSum other);
  ProductOperator operator*(OperatorSum other);
  ProductOperator operator*=(OperatorSum other);
  OperatorSum operator+(ScalarOperator other);
  OperatorSum operator-(ScalarOperator other);
  OperatorSum operator+=(ScalarOperator other);
  OperatorSum operator-=(ScalarOperator other);
  ProductOperator operator*(ScalarOperator other);
  ProductOperator operator*=(ScalarOperator other);
  OperatorSum operator+(ProductOperator other);
  OperatorSum operator-(ProductOperator other);
  OperatorSum operator+=(ProductOperator other);
  OperatorSum operator-=(ProductOperator other);
  ProductOperator operator*(ProductOperator other);
  ProductOperator operator*=(ProductOperator other);
  OperatorSum operator+(ElementaryOperator other);
  OperatorSum operator-(ElementaryOperator other);
  OperatorSum operator+=(ElementaryOperator other);
  OperatorSum operator-=(ElementaryOperator other);
  ProductOperator operator*(ElementaryOperator other);
  ProductOperator operator*=(ElementaryOperator other);
  /// @brief True, if the other value is an elementary operator with the same id
  /// acting on the same degrees of freedom, and False otherwise.
  bool operator==(ElementaryOperator other);

  /// @brief Return the `ElementaryOperator` as a string.
  std::string to_string() const;

  /// @brief Return the `ElementaryOperator` as a matrix.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  complex_matrix to_matrix(std::vector<int> degrees,
                           std::vector<std::complex<double>> parameters);

  /// FIXME Return the `ElementaryOperator`, given that it doesn't need
  /// any parameters or degrees. Just using this for convenience testing
  /// right now, should revisit if we need this moving forward.
  complex_matrix to_matrix();

  static ElementaryOperator identity(int degree);
  static ElementaryOperator zero(int degree);

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
  void define(std::string operator_id, std::vector<int> expected_dimensions,
              Func create) {
    if (m_ops.find(operator_id) != m_ops.end()) {
      // todo: make a nice error message to say op already exists
      throw;
    }
    auto defn = Definition();
    defn.create_definition(operator_id, degrees, create);
    m_ops[operator_id] = defn;
  }

  /// Attributes.

  /// @brief The number of levels, that is the dimension, for each degree of
  /// freedom in canonical order that the operator acts on. A value of zero or
  /// less indicates that the operator is defined for any dimension of that
  /// degree.
  std::vector<int> expected_dimensions;
  /// @brief The degrees of freedom that the operator acts on in canonical
  /// order.
  std::vector<int> degrees;
  /// @brief A map of the paramter names to their concrete, complex values.
  /// TODO: Will use positional arguments instead of keyword, so this should
  /// just be a vector of the parameter values, where the order will never
  /// change.
  // std::map<std::string, std::complex<double>> parameters;
  /// @brief In C++, I believe we will simply restrict the user-provided
  /// function to only take a vector of complex values.
  std::vector<std::complex<double>> parameters;
  std::string id;

  // /// @brief Creates a representation of the operator as `pauli_word` that
  // can be passed as an argument to quantum kernels.
  // pauli_word to_pauli_word ovveride();
};

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
// class OperatorSum {

// private:
//   std::vector<ProductOperator> m_terms;

// public:
//   /// @brief Construct a `cudaq::OperatorSum` given a sequence of
//   /// `cudaq::ProductOperator`'s.
//   /// This operator expression represents a sum of terms, where each term
//   /// is a product of elementary and scalar operators.
//   OperatorSum(std::vector<ProductOperator> &terms);

//   /// @brief Empty constructor.
//   OperatorSum();

//   // Arithmetic overloads against all other operator types.
//   OperatorSum operator+(double other);
//   OperatorSum operator-(double other);
//   OperatorSum operator*(double other);
//   OperatorSum operator+=(double other);
//   OperatorSum operator-=(double other);
//   OperatorSum operator*=(double other);
//   OperatorSum operator+(std::complex<double> other);
//   OperatorSum operator-(std::complex<double> other);
//   OperatorSum operator*(std::complex<double> other);
//   OperatorSum operator+=(std::complex<double> other);
//   OperatorSum operator-=(std::complex<double> other);
//   OperatorSum operator*=(std::complex<double> other);
//   OperatorSum operator+(OperatorSum other);
//   OperatorSum operator-(OperatorSum other);
//   OperatorSum operator*(OperatorSum other);
//   OperatorSum operator+=(OperatorSum other);
//   OperatorSum operator-=(OperatorSum other);
//   OperatorSum operator*=(OperatorSum other);
//   OperatorSum operator+(ScalarOperator other);
//   OperatorSum operator-(ScalarOperator other);
//   OperatorSum operator*(ScalarOperator other);
//   OperatorSum operator+=(ScalarOperator other);
//   OperatorSum operator-=(ScalarOperator other);
//   OperatorSum operator*=(ScalarOperator other);
//   OperatorSum operator+(ProductOperator other);
//   OperatorSum operator-(ProductOperator other);
//   OperatorSum operator*(ProductOperator other);
//   OperatorSum operator+=(ProductOperator other);
//   OperatorSum operator-=(ProductOperator other);
//   OperatorSum operator*=(ProductOperator other);
//   OperatorSum operator+(ElementaryOperator other);
//   OperatorSum operator-(ElementaryOperator other);
//   OperatorSum operator*(ElementaryOperator other);
//   OperatorSum operator+=(ElementaryOperator other);
//   OperatorSum operator-=(ElementaryOperator other);
//   OperatorSum operator*=(ElementaryOperator other);
//   /// @brief  True, if the other value is an OperatorSum with equivalent
//   terms,
//   /// and False otherwise. The equality takes into account that operator
//   /// addition is commutative, as is the product of two operators if they
//   /// act on different degrees of freedom.
//   /// The equality comparison does *not* take commutation relations into
//   /// account, and does not try to reorder terms blockwise; it may hence
//   /// evaluate to False, even if two operators in reality are the same.
//   /// If the equality evaluates to True, on the other hand, the operators
//   /// are guaranteed to represent the same transformation for all arguments.
//   bool operator==(OperatorSum other);

//   /// @brief Return the OperatorSum as a string.
//   std::string to_string() const;

//   /// @brief Return the `OperatorSum` as a matrix.
//   /// @arg `dimensions` : A mapping that specifies the number of levels,
//   ///                      that is, the dimension of each degree of freedom
//   ///                      that the operator acts on. Example for two,
//   2-level
//   ///                      degrees of freedom: `{0:2, 1:2}`.
//   /// @arg `parameters` : A map of the paramter names to their concrete,
//   complex
//   /// values.
//   complex_matrix
//   to_matrix(std::map<int, int> dimensions,
//             std::map<std::string, std::complex<double>> parameters);

//   /// @brief Creates a representation of the operator as a
//   `cudaq::pauli_word`
//   /// that can be passed as an argument to quantum kernels.
//   pauli_word to_pauli_word();

//   /// Attributes on the class.

//   /// @brief The degrees of freedom that the oeprator acts on in canonical
//   /// order.
//   std::vector<int> degrees;

//   /// @brief A map of the paramter names to their concrete, complex values.
//   std::map<std::string, std::complex<double>> parameters;
// };

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types that
/// can.
// class ProductOperator {

//   /// @brief Constructor for an operator expression that represents a product
//   /// of elementary operators.
//   /// @arg atomic_operators : The operators of which to compute the product
//   when
//   ///                         evaluating the operator expression.
//   ProductOperator(std::vector<ElementaryOperator> atomic_operators);

//   /// @brief Constructor for an operator expression that represents a product
//   /// of e scalar operators.
//   /// @arg atomic_operators : The operators of which to compute the product
//   when
//   ///                         evaluating the operator expression.
//   ProductOperator(std::vector<ScalarOperator> atomic_operators);

//   ProductOperator(std::vector<ElementaryOperator> elementary_operators,
//   std::vector<ScalarOperator> scalar_operators);

//   // Arithmetic overloads against all other operator types.
//   OperatorSum operator+(int other);
//   OperatorSum operator-(int other);
//   OperatorSum operator+=(int other);
//   OperatorSum operator-=(int other);
//   ProductOperator operator*(int other);
//   ProductOperator operator*=(int other);
//   OperatorSum operator+(double other);
//   OperatorSum operator-(double other);
//   OperatorSum operator+=(double other);
//   OperatorSum operator-=(double other);
//   ProductOperator operator*(double other);
//   ProductOperator operator*=(double other);
//   OperatorSum operator+(std::complex<double> other);
//   OperatorSum operator-(std::complex<double> other);
//   OperatorSum operator+=(dostd::complex<double> other);
//   OperatorSum operator-=(std::complex<double> other);
//   ProductOperator operator*(std::complex<double> other);
//   ProductOperator operator*=(std::complex<double> other);
//   OperatorSum operator+(OperatorSum other);
//   OperatorSum operator-(OperatorSum other);
//   OperatorSum operator+=(OperatorSum other);
//   OperatorSum operator-=(OperatorSum other);
//   ProductOperator operator*(OperatorSum other);
//   ProductOperator operator*=(OperatorSum other);
//   OperatorSum operator+(ScalarOperator other);
//   OperatorSum operator-(ScalarOperator other);
//   OperatorSum operator+=(ScalarOperator other);
//   OperatorSum operator-=(ScalarOperator other);
//   ProductOperator operator*(ScalarOperator other);
//   ProductOperator operator*=(ScalarOperator other);
//   OperatorSum operator+(ProductOperator other);
//   OperatorSum operator-(ProductOperator other);
//   OperatorSum operator+=(ProductOperator other);
//   OperatorSum operator-=(ProductOperator other);
//   ProductOperator operator*(ProductOperator other);
//   ProductOperator operator*=(ProductOperator other);
//   OperatorSum operator+(ElementaryOperator other);
//   OperatorSum operator-(ElementaryOperator other);
//   OperatorSum operator+=(ElementaryOperator other);
//   OperatorSum operator-=(ElementaryOperator other);
//   ProductOperator operator*(ElementaryOperator other);
//   ProductOperator operator*=(ElementaryOperator other);
//   /// @brief True, if the other value is an OperatorSum with equivalent
//   terms,
//   ///  and False otherwise. The equality takes into account that operator
//   ///  addition is commutative, as is the product of two operators if they
//   ///  act on different degrees of freedom.
//   ///  The equality comparison does *not* take commutation relations into
//   ///  account, and does not try to reorder terms blockwise; it may hence
//   ///  evaluate to False, even if two operators in reality are the same.
//   ///  If the equality evaluates to True, on the other hand, the operators
//   ///  are guaranteed to represent the same transformation for all arguments.
//   bool operator==(ProductOperator other);

//   /// @brief Return the `ProductOperator` as a string.
//   std::string to_string() const;

//   /// @brief Return the `OperatorSum` as a matrix.
//   /// @arg  `dimensions` : A mapping that specifies the number of levels,
//   ///                      that is, the dimension of each degree of freedom
//   ///                      that the operator acts on. Example for two,
//   2-level
//   ///                      degrees of freedom: `{0:2, 1:2}`.
//   /// @arg `parameters` : A map of the paramter names to their concrete,
//   complex
//   /// values.
//   complex_matrix to_matrix(
//       std::map<int, int> dimensions,
//       std::map<std::string, std::complex<double>> parameters);

//   /// @brief Creates a representation of the operator as a
//   `cudaq::pauli_word`
//   /// that can be passed as an argument to quantum kernels.
//   pauli_word to_pauli_word();

//   /// @brief The degrees of freedom that the oeprator acts on in canonical
//   /// order.
//   std::vector<int> degrees;

//   /// @brief A map of the paramter names to their concrete, complex values.
//   std::map<std::string, std::complex<double>> parameters;
// };

// class ScalarOperator {

// /// @brief Constructor.
// /// @arg generator: The value of the scalar operator as a function of its
// /// parameters. The generator may take any number of complex-valued arguments
// /// and must return a number.
// ScalarOperator(Callable generator,
//                std::map<std::string, std::complex<double>> parameters);

// // Arithmetic overloads against all other operator types.
// ScalarOperator operator+(int other);
// ScalarOperator operator-(int other);
// ScalarOperator operator+=(int other);
// ScalarOperator operator-=(int other);
// ScalarOperator operator*(int other);
// ScalarOperator operator*=(int other);
// ScalarOperator operator/(int other);
// ScalarOperator operator/=(int other);
// ScalarOperator operator+(double other);
// ScalarOperator operator-(double other);
// ScalarOperator operator+=(double other);
// ScalarOperator operator-=(double other);
// ScalarOperator operator*(double other);
// ScalarOperator operator*=(double other);
// ScalarOperator operator/(double other);
// ScalarOperator operator/=(double other);
// ScalarOperator operator+(std::complex<double> other);
// ScalarOperator operator-(std::complex<double> other);
// ScalarOperator operator+=(std::complex<double> other);
// ScalarOperator operator-=(std::complex<double> other);
// ScalarOperator operator*(std::complex<double> other);
// ScalarOperator operator*=(std::complex<double> other);
// ScalarOperator operator/(std::complex<double> other);
// ScalarOperator operator/=(std::complex<double> other);
// ScalarOperator operator+(OperatorSum other);
// ScalarOperator operator-(OperatorSum other);
// ScalarOperator operator+=(OperatorSum other);
// ScalarOperator operator-=(OperatorSum other);
// ScalarOperator operator*(OperatorSum other);
// ScalarOperator operator*=(OperatorSum other);
// ScalarOperator operator/(OperatorSum other);
// ScalarOperator operator/=(OperatorSum other);
// ScalarOperator operator+(ScalarOperator other);
// ScalarOperator operator-(ScalarOperator other);
// ScalarOperator operator+=(ScalarOperator other);
// ScalarOperator operator-=(ScalarOperator other);
// ScalarOperator operator*(ScalarOperator other);
// ScalarOperator operator*=(ScalarOperator other);
// ScalarOperator operator/(ScalarOperator other);
// ScalarOperator operator/=(ScalarOperator other);
// ScalarOperator pow(ScalarOperator other);
// ScalarOperator operator+(ProductOperator other);
// ScalarOperator operator-(ProductOperator other);
// ScalarOperator operator+=(ProductOperator other);
// ScalarOperator operator-=(ProductOperator other);
// ScalarOperator operator*(ProductOperator other);
// ScalarOperator operator*=(ProductOperator other);
// ScalarOperator operator/(ProductOperator other);
// ScalarOperator operator/=(ProductOperator other);
// ScalarOperator operator+(ElementaryOperator other);
// ScalarOperator operator-(ElementaryOperator other);
// ScalarOperator operator+=(ElementaryOperator other);
// ScalarOperator operator-=(ElementaryOperator other);
// ScalarOperator operator*(ElementaryOperator other);
// ScalarOperator operator*=(ElementaryOperator other);
// ScalarOperator operator/(ElementaryOperator other);
// ScalarOperator operator/=(ElementaryOperator other);

// /// @brief Returns true if other is a scalar operator with the same
// /// generator.
// bool operator==(ScalarOperator other);

// /// @brief Return the `OperatorSum` as a matrix.
// /// @arg  `dimensions` : A mapping that specifies the number of levels,
// ///                      that is, the dimension of each degree of freedom
// ///                      that the operator acts on. Example for two, 2-level
// ///                      degrees of freedom: `{0:2, 1:2}`.
// /// @arg `parameters` : A map of the paramter names to their concrete,
// complex
// /// values.
// complex_matrix to_matrix(
//     std::map<int, int> dimensions,
//     std::map<std::string, std::complex<double>> parameters);

// /// @brief Creates a representation of the operator as `pauli_word` that can
// /// be passed as an argument to quantum kernels.
// pauli_word to_pauli_word ovveride();

// /// @brief The number of levels, that is the dimension, for each degree of
// /// freedom in canonical order that the operator acts on. A value of zero or
// /// less indicates that the operator is defined for any dimension of that
// /// degree.
// std::vector<int> dimensions;
// /// @brief The degrees of freedom that the operator acts on in canonical
// /// order.
// std::vector<int> degrees;
// /// A map of the paramter names to their concrete, complex
// /// values.
// std::map<std::string, std::complex<double>> parameters;

// /// @brief The function that generates the value of the scalar operator.
// /// The function can take any number of complex-valued arguments
// /// and returns a number.
// Callable generator;
// };

} // namespace cudaq