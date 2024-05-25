import cudaq

import qutip
from qutip import Qobj, QobjEvo

import numpy as np

# TODO:
# Write a "kernel" containing analog hamiltonian terms that
# will build up an underlying QuTip QOBJ.

global operators
operators = []


# Variables used in the construction of parameterized Hamiltonian
# operators.
class Variable:
    pass


cudaq.Variable = Variable


class Time(Variable):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # FIXME
        return 1.0


cudaq.Time = Time


class Coefficient(Variable):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # FIXME
        return 1.0


cudaq.Coefficient = Coefficient


# Kernel-level spin operators.
class operator:
    pass


cudaq.operator = operator


class X(operator):

    def __init__(self, qubit: int):
        self.qubit = qubit
        self.operator_term = cudaq.spin.x(qubit)
        operators.append(self.operator_term)

    def __mul__(self, value: float):
        self.operator_term *= value

    def __mul__(self, values: np.ndarray):
        # FIXME: This will construct in the necessary form
        # for a QobjEvo as to ensure it builds up a time-dependent
        # term.
        pass

    def __rmul__(self, value: float):
        self.operator_term *= value

    def __add__(self, value: float):
        self.operator_term += value

    def __radd__(self, value: float):
        self.operator_term += value

    def __sub__(self, value: float):
        self.operator_term -= value

    def __rsub__(self, value: float):
        self.operator_term -= value


cudaq.operator.X = X


class Y(operator):

    def __init__(self, qubit: int):
        self.qubit = qubit
        self.operator_term = cudaq.spin.y(qubit)
        operators.append(self.operator_term)

    def __mul__(self, value: float):
        self.operator_term *= value

    def __mul__(self, values: np.ndarray):
        # FIXME: This will construct in the necessary form
        # for a QobjEvo as to ensure it builds up a time-dependent
        # term.
        pass

    def __rmul__(self, value: float):
        self.operator_term *= value

    def __add__(self, value: float):
        self.operator_term += value

    def __radd__(self, value: float):
        self.operator_term += value

    def __sub__(self, value: float):
        self.operator_term -= value

    def __rsub__(self, value: float):
        self.operator_term -= value


cudaq.operator.Y = Y


class Z(operator):

    def __init__(self, qubit: int):
        self.qubit = qubit
        self.operator_term = cudaq.spin.z(qubit)
        operators.append(self.operator_term)

    def __mul__(self, value: float):
        self.operator_term *= value

    def __mul__(self, values: np.ndarray):
        # FIXME: This will construct in the necessary form
        # for a QobjEvo as to ensure it builds up a time-dependent
        # term.
        pass

    def __rmul__(self, value: float):
        self.operator_term *= value

    def __add__(self, value: float):
        self.operator_term += value

    def __radd__(self, value: float):
        self.operator_term += value

    def __sub__(self, value: float):
        self.operator_term -= value

    def __rsub__(self, value: float):
        self.operator_term -= value


cudaq.operator.Z = Z


class I(operator):

    def __init__(self, qubit: int):
        self.qubit = qubit
        self.operator_term = cudaq.spin.i(qubit)
        operators.append(self.operator_term)

    def __mul__(self, value):
        self.operator_term *= value

    def __mul__(self, values: np.ndarray):
        # FIXME: This will construct in the necessary form
        # for a QobjEvo as to ensure it builds up a time-dependent
        # term.
        pass

    def __rmul__(self, value):
        self.operator_term *= value

    def __add__(self, value):
        self.operator_term += value

    def __radd__(self, value):
        self.operator_term += value

    def __sub__(self, value):
        self.operator_term -= value

    def __rsub__(self, value):
        self.operator_term -= value


cudaq.operator.I = I


# Analog kernel decorator -- akin to the `@cudaq.kernel` decorator.
class analog_kernel:

    def __init__(self, kernel_function):
        self.kernel_function = kernel_function
        # We will store the hamiltonian terms in this variable, then
        # convert each to a QuTip Qobj for simulation.
        self.operator_terms = []

    def __call__(self, *args, **kwargs):
        return self.kernel_function(*args, **kwargs)

    def __compile__(self, *args, **kwargs):
        """ Lowers the analog kernel to a list of qutip Qobj's. """
        self.stitched_operators = sum(operators)

        self.qobjects = []
        for term in self.stitched_operators:
            self.qobjects.append(Qobj(np.asarray(term.to_matrix())))

        # TODO: Also need to be able to handle coefficients, and time-parameterized
        # coefficients.

        # This will require building up a `QobjEvo` instead of a `Qobj`.


cudaq.analog_kernel = analog_kernel


# Updated observe function to handle the analog kernel.
def observe(kernel, *args, debug=False, **kwargs):
    kernel.__call__(*args, **kwargs)
    kernel.__compile__(*args, **kwargs)
    # Just a temporary flag.
    if (debug):
        for obj in kernel.qobjects:
            print(obj)
            print("\n")


cudaq.observe = observe
