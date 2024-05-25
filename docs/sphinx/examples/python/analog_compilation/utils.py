import cudaq

import qutip
from qutip import Qobj, QobjEvo

import numpy as np
import functools

# Storing the `cudaq.operator`'s that we build up here in this global variable.
# This is not a great solution, but it prevents me from having to pass extra
# arguments around to the X/Y/Z/I operators on the mock front end.
global operators
operators = []


# Variables used in the construction of parameterized Hamiltonian
# operators.
class Variable:
    pass


cudaq.Variable = Variable


class Time(Variable):

    def __init__(self, *args, **kwargs):
        # TODO: Add member variables for keeping track of timing information:
        # (start_time, stop_time, dt)
        self.sample_times = None

    def set_system_sample_times(self, times: np.ndarray):
        self.sample_times = times

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


# Parent class for the operators that we will use to construct
# complex, time-dependent hamiltonians.
class operator:

    def __init__(self, qubit: int):
        self.qubit = qubit
        self.operator_term = cudaq.spin.x(qubit)
        operators.append(self)
        # Store any coefficients that will be time-dependent; e.g, we don't
        # just have a single value for them. This may either be a function
        # that is called at each time step, or an array where each respective
        # member (coefficient) belongs to its own time step.
        self.operator_coefficients = None

    def from_data(self, *args, **kwargs):
        """
            *args : the `cudaq.operator`'s that we will be starting from.
        """
        pass


    def __mul__(self, coefficients):
        """ 
        Support:

            `float * OPERATOR(qubit)`
            `np.ndarray[time] * OPERATOR(qubit)`
            `foo(time) * OPERATOR(qubit)
         
        This is needed in one function because python doesn't do
        a great job of rerouting overload calls if the function
        signatures have the similar argument structures.

        This leads to the undesirable variable name, `coefficients`,
        which should really be `coefficient` if it's only a float,
        and `coefficients` if it's a vector or function.
        """
        incoming_type = type(coefficients)
        # If someone passed us a list of coefficients in time, or a time-dependent
        # coefficient function, we set the member variable.
        same_types = [X, Y, Z, I]
        if incoming_type != float and incoming_type not in same_types:
            self.operator_coefficients = coefficients
        # If it's another operator, multiply them at the `cudaq::spin` level.
        elif (incoming_type in same_types):
            print(
                "WARNING: Same type operation that may not handle coefficients properly.\n"
            )
            self.operator_term *= coefficients.operator_term
            # FIXME: Collapse that operator into self and remove the other.
            # operators.remove(coefficients.operator_term)
            return self  # FIXME return operator(qubit, )
        # Otherwise, it's just a conventional coefficient, so multiply through.
        else:
            self.operator_term *= coefficients

    # NOTE: There is this weird bug in numpy where, if you use __rmul__
    # between a numpy array and the cudaq.operator: `np.array([1,2,3]) * X(qubit)`,
    # it will use the `numpy.array.__mul__()` method instead of our rmul, giving us a
    # bug. If I jack this `__array_priority__` variable up, however, it solves the
    # issue.
    __array_priority__ = 10000

    def __rmul__(self, coefficients):
        self.__mul__(coefficients)

    def __add__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            print(
                "WARNING: Same type operation that may not handle coefficients properly.\n"
            )
            self.operator_term += value.operator_term

        else:
            self.operator_term += value

    def __radd__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            print(
                "WARNING: Same type operation that may not handle coefficients properly.\n"
            )
            value.operator_term += self.operator_term
        else:
            value += self.operator_term

    def __sub__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            print(
                "WARNING: Same type operation that may not handle coefficients properly.\n"
            )
            self.operator_term -= value.operator_term
        else:
            self.operator_term -= value

    def __rsub__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            print(
                "WARNING: Same type operation that may not handle coefficients properly.\n"
            )
            value.operator_term -= self.operator_term
        else:
            value -= self.operator_term


cudaq.operator = operator

# Define the `X,Y,Z,I` quantum operators.


class X(cudaq.operator):
    pass


class Y(cudaq.operator):
    pass


class Z(cudaq.operator):
    pass


class I(cudaq.operator):
    pass


cudaq.operator.X = X
cudaq.operator.Y = Y
cudaq.operator.Z = Z
cudaq.operator.I = I


# Analog kernel decorator -- akin to the `@cudaq.kernel` decorator.
class analog_kernel:

    def __init__(self, kernel_function: callable):
        self.kernel_function = kernel_function
        self.qubit_count = None

    def __call__(self, *args, **kwargs):
        return self.kernel_function(*args, **kwargs)

    def __compile__(self, time_steps: np.ndarray, *args, **kwargs):
        """ 
        Lowers the analog kernel to a list of qutip Qobj's
        containing the parameterized coefficients.

        TODO: Eventually just call this from the `__call__` function
        and replace the start of the observe function with just
        ``` kernel.__call__(*args, **kwargs) ```
        """
        # Get the total number of qubits in the user-created operator.
        # We have to sum all of the terms here since we've kept track
        # of them individually, and therefore, not all of them have been
        # projected to the full Hilbert space yet.
        self.qubit_count = sum([term.operator_term for term in operators
                               ]).get_qubit_count()

        self.qobjects = []
        for term in operators:
            # Each operator term was constructed in isolation, therefore
            # they have no clue of the total size that the Hilbert space
            # has grown to. This will cast everything to the full Hilbert
            # space, such that the terms are all of the same dimension.
            term.operator_term.project_hilbert_space(self.qubit_count)

            # If we had a function or array for our coefficients, we must
            # pass those along separately while building up our list of
            # qobjects.
            if term.operator_coefficients is not None:
                # If the coefficient was a function, let's just create
                # the array of samples upfront ourselves to save the
                # overhead.
                if (type(term.operator_coefficients) != np.ndarray):
                    coefficients = np.array([
                        term.operator_coefficients(time) for time in time_steps
                    ])
                    self.qobjects.append([
                        Qobj(np.asarray(term.operator_term.to_matrix())),
                        coefficients
                    ])
                else:
                    self.qobjects.append([
                        Qobj(np.asarray(term.operator_term.to_matrix())),
                        term.operator_coefficients
                    ])
            # Otherwise, if we just had a plain old spin term with the constant
            # coefficient already attached to it, just append that Qobj directly.
            else:
                self.qobjects.append(
                    Qobj(np.asarray(term.operator_term.to_matrix())))


cudaq.analog_kernel = analog_kernel


# Updated observe function to handle the analog kernel.
def observe(kernel, time_steps, *args, verbose=False, **kwargs):
    """
    FIXME: Extend to sequential waveforms.

    1.  The big clock of sample times can become our global time.
        Then we can have micro clocks within that that can kick off sub-routines, or
        allow for multiple operations on a qubit within the program.
        This would make the global time list potentially non-linear.

    ```
    # Example with 3 sequential waveforms:

    global_times = [
        np.arange(0,T_1,steps=dt), 
        np.arange(0,T_2,steps=dt), 
        np.arange(0,T_3,steps=dt)
    ]
    # Our first initial state for the complete workflow is `|0>`
    psi = qutip.basis(2**(kernel.qubit_count), 0)
    for time in global_times:
        evolved_state = qutip.mesolve(..., psi, time, ...)
        psi = evolved_state

    # The `psi` that we have after the end of the loop should
    # be the completely evolved state vector after the 3 sequential
    # signals.
    ```

    """
    kernel.__call__(*args, **kwargs)
    kernel.__compile__(time_steps, *args, **kwargs)

    # Just a temporary flag.
    if (verbose):
        for obj in kernel.qobjects:
            print(obj)
            print("\n")

    # Delegate the simulation to the QuTip solver backend.
    # You could imagine the user setting which solver to use via:
    #   `cudaq.set_target('qutip-SESolver')`
    #   `cudaq.set_target('qutip-mesolve')`
    #   ...

    # FIXME: Just starting from the 0-state always for now.
    psi_initial = qutip.basis(2**(kernel.qubit_count), 0)

    # Save off the state vectors during simulation.
    solver_options = {
        "store_final_state": True,
        "store_states": False
        # "progress_bar": "enhanced"
    }
    # FIXME: Just hard-coding to return `<psi | Z | psi>`.
    expectation_operators = [
        cudaq.spin.z(qubit) for qubit in range(kernel.qubit_count)
    ]
    sigma_z = Qobj(np.asarray(sum(expectation_operators).to_matrix()))

    # Execute the simulation on the Qutip backend.
    result = qutip.sesolve(kernel.qobjects,
                           psi_initial,
                           time_steps,
                           e_ops=sigma_z,
                           options=solver_options)

    print(f"expectation values = {result.expect}\n")
    print(f"psi_final = {result.final_state}\n")
    # print(f"stats = {result.stats}\n")


cudaq.observe = observe


# Returns the measured distribution after simulation.
def sample(kernel, *args, debug=False, **kwargs):
    pass


cudaq.sample = sample


# Returns the final state vector from simulation.
def get_state(kernel, *args, debug=False, **kwargs):
    pass


cudaq.get_state = get_state
