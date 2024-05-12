import cudaq

import time
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


def optimization_function(x: np.ndarray, *args):
    # In this case, we'll just let each individual parameter
    # represent the amplitude at a time chunk. So we will
    # have 1 parameter for each chunk.
    waveform = x
    control_signal.set_sample_values(waveform)

    # Synthesize the unitary operations for this Hamiltonian with
    # the provided control amplitudes.
    start = time.time()
    unitary_operations = cudaq.synthesize_unitary(Hamiltonian, time_variable)
    stop = time.time()
    # print(f"unitary synthesis took {stop-start} seconds")

    # Allocate a qubit to a kernel and apply the registered unitary operations
    # (taken from the global dict), to the kernel.
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    for unitary_operation in unitary_operations.keys():
        evaluation_string = "kernel." + unitary_operation + "(qubit)"
        eval(evaluation_string)

    start = time.time()
    got_state = np.asarray(cudaq.get_state(kernel))
    stop = time.time()
    got_states.append(got_state)
    # print(f"kernel execution took {stop-start} seconds")
    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_state_fidelity(want_state, got_state)
    print(f"cost = {cost}")

    return cost


def calculate_want_state(want_gate: np.ndarray):
    # Start in |0> state.
    initial_state = np.array([1., 0.])
    # want_state = gate * |0>
    return np.dot(want_gate, initial_state)


def run_optimization(unitary_gate: np.ndarray):
    """
    Closed loop optimization of the waveform:
    Will optimize the waveform to produce the provided `unitary_gate`.
    """
    # Semi-arbitrary bounds on the amplitude of the waveform.
    lower = [-1.] * chunks
    upper = [1.] * chunks
    bounds = optimize.Bounds(lb=lower, ub=upper)
    global want_state
    want_state = calculate_want_state(unitary_gate)
    # Just using random numbers to start with on our waveform.
    # initial_waveform = np.random.uniform(low=lower, high=upper, size=(chunks,))
    initial_waveform = np.full(chunks, fill_value=upper[0])
    # Use Simulated Annealing to minimize the function.
    # An alternative optimizer to try is `scipy.basinhopping`
    optimized_result = optimize.dual_annealing(
        func=optimization_function,
        x0=initial_waveform,
        bounds=list(zip(lower, upper)),
        #  visit=1.25,
        accept=-1e4,
        no_local_search=True,
        #  seed=np.random.default_rng(4),
        maxiter=100)
    return optimized_result


# Global variable to store states in so I can double check them later.
got_states = []

# # Optimize for an X-gate.
# x_gate = np.array([[0, 1], [1, 0]])
# x_result = run_optimization(x_gate)
# x_waveform = x_result.x
# x_state = got_states[-1]
# print("x_state = ", x_state)
# print("abs(x_state) = ", np.abs(x_state))
# print("fidelity = ", 1.0 - x_result.fun, "\n")

# Optimize for a Hadamard-gate.
hadamard_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
h_result = run_optimization(hadamard_gate)
hadamard_waveform = h_result.x
h_state = got_states[-1]
print("h_state = ", h_state)
print("abs(h_state) = ", np.abs(h_state))
print("fidelity = ", 1.0 - h_result.fun, "\n")

# Optimize for a randomly generated Unitary matrix.
random_gate = stats.unitary_group.rvs(2)
print("random_gate = ", random_gate)
random_result = run_optimization(random_gate)
random_waveform = random_result.x
random_state = got_states[-1]
print("random_state = ", random_state)
print("abs(random_state) = ", np.abs(random_state))
print("fidelity = ", 1.0 - random_result.fun, "\n")

#################################################################################################################

# # Make pretty pictures of the waveforms
# fig, axs = plt.subplots(3)
# axs[0].set_title(f"Optimized X Waveform\n fidelity={1.0 - x_result.fun}")
# axs[0].step(time_variable.time_series(), x_waveform)
# axs[1].set_title(f"Optimized Hadamard Waveform\n fidelity={1.0 - h_result.fun}")
# axs[1].step(time_variable.time_series(), hadamard_waveform)
# axs[1].set_ylabel("Waveform Amplitude")
# axs[2].set_title(
#     f"Optimized Random Unitary Waveform\n fidelity={1.0 - random_result.fun}")
# axs[2].step(time_variable.time_series(), random_waveform)
# axs[2].set_xlabel("Time\n(N_time_chunks * dt = T)")
# fig.tight_layout(pad=1.0)

# plt.savefig("out.png", bbox_inches="tight", pad_inches=0.5)
