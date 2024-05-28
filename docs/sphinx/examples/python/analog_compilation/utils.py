import cudaq

import qutip
import numpy as np

# Storing the `cudaq.operator`'s that we build up here in this global variable.
# This is not a great solution, but it prevents me from having to pass extra
# arguments around to the X/Y/Z/I operators on the mock front end.
global operators
operators = []


# Parent class for the operators that we will use to construct
# complex, time-dependent hamiltonians.
class operator:

    def __init__(self, qubit: int, term: str = None, start_empty=False):
        if not start_empty:
            self.qubit = qubit
            # Instead of overloading this constructor for each
            # child class, I just use a hacky f-string method.
            exec(f"self.operator_term = cudaq.spin.{term}(qubit)")
            operators.append(self)
            # Store any coefficients that will be time-dependent; e.g, we don't
            # just have a single value for them. This may either be a function
            # that is called at each time step, or an array where each respective
            # member (coefficient) belongs to its own time step.
            self.operator_coefficients = None
        else:
            # Used in conjunction with the below `from_other_operations` method.
            # Code this ugly wouldn't exist in C++ because I'd just be able to
            # overload the constructor there and not have to use this `start_empty`
            # function argument :-)
            self.qubit = qubit
            self.operator_term = None
            self.operator_coefficients = None

    def from_other_operations(self,
                              other_operations=[],
                              arithmetic_op="__mul__"):
        """
        When someone does arithmetic between two operators in their kernel,
        we need to merge the joined `cudaq.SpinOperator` into one `cudaq.operator`.
        This function allows you to pass those other operations (just 2 for right now)
        in to the operator.


        *args : The `cudaq.operator`'s that we will be starting from. We've already
                asserted that all of the previous operators are on the same qubit.
        arithmetic_op : The binary operator used to merge the different operation data
                        together.
        """
        assert len(other_operations) == 2
        self.qubit = other_operations[0].qubit
        term_0 = other_operations[0].operator_term
        term_1 = other_operations[1].operator_term
        execution_string = f"self.operator_term = term_0.{arithmetic_op}(term_1)"
        exec(execution_string)
        # Since the other operations should've been made on the spot
        # in the analog kernel, they shouldn't have taken on any coefficients
        # yet.
        self.operator_coefficients = None

        # Will have to remove the original operators that were spun up and added
        # to the operators list, as they were constructed at the same time as we
        # were performing the arithmetic operation that calls this constructor.
        for operation in other_operations:
            operators.remove(operation)

        # Add this operator in their place.
        operators.append(self)

    def __mul__(self, coefficients):
        """ 
        Support:

            `float * OPERATOR(qubit)`
            `np.ndarray[time] * OPERATOR(qubit)`
            `foo(time) * OPERATOR(qubit)
         
        This is needed in one function because python doesn't do
        a great job of rerouting overload calls if the function
        signatures have similar argument structures.

        This leads, however, to the undesirable variable name, 
        `coefficients`,  which should really be `coefficient` if 
        it's only a float, `coefficients` if it's a vector
        or function, and `other` if it's another `cudaq.operator`.
        """
        incoming_type = type(coefficients)
        # If someone passed us a list of coefficients in time, or a time-dependent
        # coefficient function, we set the member variable.
        same_types = [X, Y, Z, I]
        if incoming_type != float and incoming_type not in same_types:
            self.operator_coefficients = coefficients
        # If it's another operator, multiply them at the `cudaq::spin` level.
        elif (incoming_type in same_types):
            assert (self.qubit == coefficients.qubit)
            new_operator = cudaq.operator(self.qubit, start_empty=True)
            new_operator.from_other_operations(
                other_operations=[self, coefficients])
            return new_operator
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
            assert (self.qubit == value.qubit)
            new_operator = cudaq.operator(self.qubit, start_empty=True)
            new_operator.from_other_operations(other_operations=[self, value],
                                               arithmetic_op="__add__")
            return new_operator
        else:
            self.operator_term += value

    def __radd__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            assert (self.qubit == value.qubit)
            new_operator = cudaq.operator(self.qubit, start_empty=True)
            new_operator.from_other_operations(other_operations=[self, value],
                                               arithmetic_op="__radd__")
            return new_operator
        else:
            value += self.operator_term

    def __sub__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            assert (self.qubit == value.qubit)
            new_operator = cudaq.operator(self.qubit, start_empty=True)
            new_operator.from_other_operations(other_operations=[self, value],
                                               arithmetic_op="__sub__")
            return new_operator
        else:
            self.operator_term -= value

    def __rsub__(self, value):
        incoming_type = type(value)
        same_types = [X, Y, Z, I]
        if (incoming_type in same_types):
            assert (self.qubit == value.qubit)
            new_operator = cudaq.operator(self.qubit, start_empty=True)
            new_operator.from_other_operations(other_operations=[self, value],
                                               arithmetic_op="__rsub__")
            return new_operator
        else:
            value -= self.operator_term


cudaq.operator = operator

# Define the `X,Y,Z,I` quantum operators.


class X(cudaq.operator):

    def __init__(self, qubit: int, start_empty=False):
        super(X, self).__init__(qubit, term="x", start_empty=False)


class Y(cudaq.operator):

    def __init__(self, qubit: int, start_empty=False):
        super(Y, self).__init__(qubit, term="y", start_empty=False)


class Z(cudaq.operator):

    def __init__(self, qubit: int, start_empty=False):
        super(Z, self).__init__(qubit, term="z", start_empty=False)


class I(cudaq.operator):

    def __init__(self, qubit: int, start_empty=False):
        super(I, self).__init__(qubit, term="i", start_empty=False)


cudaq.operator.X = X
cudaq.operator.Y = Y
cudaq.operator.Z = Z
cudaq.operator.I = I


def gate_fidelity(want_gate, got_gate):
    """
    Returns the overlap between the desired gate and the
    evolved gate. A value of 0.0 represents no overlap 
    and 1.0 represents perfect overlap.

    F = (1/(d**2)) * | tr(U_want^dag * U_got) | **2

    where d is the dimension of our gate
    """
    fidelity = (1 / want_gate.size) * (np.abs(
        np.trace(np.dot(np.conj(want_gate).T, got_gate)))**2)
    return fidelity


cudaq.operator.gate_fidelity = gate_fidelity


# Analog kernel decorator -- akin to the `@cudaq.kernel` decorator.
class analog_kernel:

    def __init__(self, kernel_function: callable):
        self.kernel_function = kernel_function
        self.qubit_count = None
        # TODO: might need a flag like this for re-compiling
        # self.already_initialized = True

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
                        qutip.Qobj(np.asarray(term.operator_term.to_matrix())),
                        coefficients
                    ])
                else:
                    self.qobjects.append([
                        qutip.Qobj(np.asarray(term.operator_term.to_matrix())),
                        term.operator_coefficients
                    ])
            # Otherwise, if we just had a plain old spin term with the constant
            # coefficient already attached to it, just append that Qobj directly.
            else:
                self.qobjects.append(
                    qutip.Qobj(np.asarray(term.operator_term.to_matrix())))

        # TODO:
        # def __recompile__(): pass


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

    # FIXME: Always starting from the 0-state for now.
    hilbert_space_size = 2**(kernel.qubit_count)
    psi_initial = qutip.basis(hilbert_space_size, 0)

    # Save off the state vectors during simulation.
    solver_options = {
        "store_final_state": True,
        "store_states": False,
        # "nsteps": 5000,
        # "progress_bar": "enhanced"
    }
    # FIXME: Just hard-coding to return `<psi | Z | psi>`.
    expectation_operators = [
        cudaq.spin.z(qubit) for qubit in range(kernel.qubit_count)
    ]
    sigma_z = qutip.Qobj(np.asarray(sum(expectation_operators).to_matrix()))

    # Execute the simulation on the Qutip backend.
    result = qutip.sesolve(H=kernel.qobjects,
                           rho0=psi_initial,
                           tlist=time_steps,
                           e_ops=sigma_z,
                           options=solver_options)

    if (verbose):
        print(f"expectation values = {result.expect}\n")
        print(f"psi_final = {result.final_state}\n")

    # Return the expectation value with respect to Z.
    return result.expect[0][-1]


cudaq.observe = observe


# Returns the measured distribution after simulation.
def sample(kernel, time_steps, *args, verbose=False, **kwargs):
    kernel.__call__(*args, **kwargs)
    kernel.__compile__(time_steps, *args, **kwargs)

    # Just a temporary flag.
    if (verbose):
        for obj in kernel.qobjects:
            print(obj)
            print("\n")

    # Save off the state vectors during simulation.
    solver_options = {
        "store_final_state": True,
        "store_states": False,
        "nsteps": 1000,
        # "progress_bar": "enhanced"
    }

    hilbert_space_size = 2**(kernel.qubit_count)
    rho0 = qutip.Qobj(np.eye(hilbert_space_size))
    # FIXME: Once every ~500 runs or so, we generate a
    # set of parameters that break the underlying scipy
    # optimizer from QuTip. The solution is to increase
    # the number of steps for the optimizer, but keeping the
    # steps this high would be super inneficient for the other
    # ~499 iterations. So we just keep trying with iteratively
    # more iterations until it works.
    try:
        result = qutip.sesolve(H=kernel.qobjects,
                               psi0=rho0,
                               tlist=time_steps,
                               options=solver_options)
    except:
        try:
            solver_options = {
                "store_final_state": True,
                "store_states": False,
                "nsteps": 2500,
            }
            result = qutip.sesolve(H=kernel.qobjects,
                                   psi0=rho0,
                                   tlist=time_steps,
                                   options=solver_options)
        except:
            try:
                solver_options = {
                    "store_final_state": True,
                    "store_states": False,
                    "nsteps": 5000,
                }
                result = qutip.sesolve(H=kernel.qobjects,
                                       psi0=rho0,
                                       tlist=time_steps,
                                       options=solver_options)
            except:
                try:
                    solver_options = {
                        "store_final_state": True,
                        "store_states": False,
                        "nsteps": 10000,
                    }
                    result = qutip.sesolve(H=kernel.qobjects,
                                           psi0=rho0,
                                           tlist=time_steps,
                                           options=solver_options)
                except:
                    try:
                        solver_options = {
                            "store_final_state": True,
                            "store_states": False,
                            "nsteps": 20000,
                        }
                        result = qutip.sesolve(H=kernel.qobjects,
                                               psi0=rho0,
                                               tlist=time_steps,
                                               options=solver_options)
                    except:
                        solver_options = {
                            "store_final_state": True,
                            "store_states": False,
                            "nsteps": 40000,
                        }
                        result = qutip.sesolve(H=kernel.qobjects,
                                               psi0=rho0,
                                               tlist=time_steps,
                                               options=solver_options)

    # Return the evolution propagator at time T: `U(T)`
    return np.asarray(result.final_state.full())


cudaq.sample = sample


# Returns the final state vector from simulation.
def get_state(kernel, time_steps, *args, verbose=False, **kwargs):
    pass


cudaq.get_state = get_state
