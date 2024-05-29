# This presents an example of defining a time-independent
# spin hamiltonian within an analog kernel, then "lowering"
# from the CUDA-Q representation to Qutip data-types.
# The qutip `SESolver` is then used as the target simulator
# for the new analog `cudaq::observe` overload.

import cudaq
from utils import *

import numpy as np


# The analog kernel will define the system Hamiltonian,
# thereby determining the instructions for the program.
# Each operator and its coefficient become an instruction
# in the quantum program.

# Some of these instructions may be static -- i.e, they are
# moreso setting constant parameters for the system, such as
# hardware frequencies, etc.

# Other instructions may be time-dependent -- i.e, they intend
# to invoke some desired unitary operation on the qubit via
# laser, AWG, etc. These time-dependent coefficients may be
# defined as either a vector of waveform samples, or a function
# that is invoked at each time step of the program.

@cudaq.analog_kernel
def kernel(waveform: np.ndarray, waveform_function: callable,
           constant_coefficients: list[float]):

    """
    Our kernel will define the hamiltonian for the system that we're
    running on.

    We then control the dynamics of that system by adding additional operator 
    terms to it, represented by spin terms with coefficients.
    This is much like we can allocate qubits in the circuit model, then manipulate
    their time evolution via gate applications.

    Here, instead of saying 
        `x(q0)` == rotating qubit 0 via X axis"
    we're saying 
        `waveform * X(0)` == apply the waveform to qubit 0 on the X-axis
    
    The actual program is determined by the unitary evolution that results
    from allowing the defined Hamiltonian to evolve over a finite time duration.
    """

    # hamiltonian = cudaq.Hamiltonian(qubit_count=2)

    constant_coefficients[0] * X(0)
    constant_coefficients[1] * Y(1)
    constant_coefficients[2] * Z(2)

    # Some control term that is `waveform[t] * X(qubit)` where the `waveform`
    # is an array of signal amplitude values.
    waveform * X(0)
    waveform * Y(1)
    # Some control term that is `waveform[t] * Y(qubit)` where the `waveform`
    # is a function that is called for each t, and returns the signal amplitude
    # at that time.
    waveform_function * Y(0)
    waveform_function * X(1)

    waveform_function_new * Y(0)

    # The idea is that the user just expresses which operator the entire
    # signal will act upon, then we will handle generating them as time-
    # dependent operator coefficients for them.


    # other_hamiltonian(....)


# Both test waveform formats just return constant values of 1.0
waveform = np.ones(20)
waveform_function = lambda t: 1.0
constants = [1.0, 2.0, 3.0]

result = cudaq.observe(kernel,
                       time_steps=np.linspace(0.0, 2.0, 20),
                       waveform=waveform,
                       waveform_function=waveform_function,
                       constant_coefficients=constants,
                       verbose=True)
