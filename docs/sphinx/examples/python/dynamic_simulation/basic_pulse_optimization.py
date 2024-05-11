import cudaq

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import stats

from new_utility_functions import *


########################### Hamiltonian Definition #############################

# A static term containing information about our ground
# state frequency. Setting it to 1 for simplicity.
omega_0 = 1.0  # arbitrary value in hz.

# `cudaq.Time`
time_variable = cudaq.Time()
time_variable.max_time = 5.  # total time, T. arbitrary value in s.
time_variable.resolution = 0.25  # time duration for each chunk of time evolution. arbitrary value in s.
# The number of time chunks that we will optimize amplitudes for.
chunks = int(time_variable.max_time / time_variable.resolution)

# `cudaq.ControlSignal`
control_signal = cudaq.ControlSignal(time=time_variable)

# `H_constant = [[0,0], [0, omega_0]]`
H_constant = (omega_0 / 2) * (cudaq.spin.i(0) - cudaq.spin.z(0))
# `H_control = waveform_amplitude_x * X + waveform_amplitude_y * Y``
#       where `waveform_amplitude_x = amplitude * np.cos(omega_0 * time)`
#       and   `waveform_amplitude_y = amplitude * np.sin(omega_0 * time)`
Hamiltonian = lambda t: np.asarray((H_constant + (control_signal(t) * np.cos(
    omega_0 * time_variable(t))) * cudaq.spin.x(0) + (control_signal(
        t) * np.sin(omega_0 * time_variable(t))) * cudaq.spin.y(0)).to_matrix())

##################################################################################

def calculate_state_fidelity(want_state, got_state):
    """
    Returns the overlap between the desired state and the
    evolved state. A value of 0.0 represents no overlap 
    (orthogonal states), and 1.0 represents perfect overlap.

    F = |<psi_want | psi_got>| ** 2
    """
    fidelity = np.abs(np.dot(np.conj(want_state).T, got_state))**2
    return fidelity

def optimization_function(parameters: np.ndarray, want_state):
    # In this case, we'll just let each individual parameter
    # represent the amplitude at a time chunk. So we will
    # have 1 parameter for each chunk.
    waveform = parameters
    control_signal.set_sample_values(waveform)

    # Synthesize the unitary operations for this Hamiltonian with
    # the provided control amplitudes.
    unitary_operations = cudaq.synthesize_unitary(Hamiltonian, time_variable)

    # Allocate a qubit to a kernel and apply the registered unitary operations
    # (taken from the global dict), to the kernel.
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    for unitary_operation in unitary_operations.keys():
      evaluation_string = "kernel." + unitary_operation + "(qubit)"
      eval(evaluation_string)

    got_state = np.asarray(cudaq.get_state(kernel))
    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_state_fidelity(want_state, got_state)
    print(cost)
    return cost

def calculate_want_state(want_gate: np.ndarray):
  # Start in |0> state.
  initial_state = np.array([1.,0.])
  # want_state = gate * |0>
  return np.dot(want_gate, initial_state)

def run_optimization(unitary_gate: np.ndarray):
    """
    Closed loop optimization of the waveform:
    Will optimize the waveform to produce the provided `unitary_gate`.
    """
    # Semi-arbitrary bounds on the amplitude of the waveform.
    lower = -1.
    upper = 1.
    bounds = optimize.Bounds(lb=lower, ub=upper)
    want_state = calculate_want_state(unitary_gate)
    # Just using random numbers to start with on our waveform.
    initial_waveform = np.random.uniform(low=lower, high=upper, size=(chunks,))
    optimized_result = optimize.minimize(optimization_function,
                                         initial_waveform,
                                         args=(want_state),
                                         bounds=bounds,
                                         method="Nelder-Mead")
    return optimized_result


# Optimize for an X-gate.
x_gate = np.array([[0, 1], [1, 0]])
run_optimization(x_gate)