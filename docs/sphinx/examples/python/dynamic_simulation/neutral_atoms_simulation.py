import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import itertools
# from multiprocessing import Pool
from multiprocess import Pool

######################################### Timing Parameters ###################################################

T = 1.5  # total time, T. aritrary value in s.
dt = 0.25  # time duration of each waveform chunk. arbitrary value in s.
chunks = int(T / dt)  # number of time chunks we will solve for

################################################################################################################


def hamiltonian(amplitude: float, phase: float, detuning: float) -> np.ndarray:
    """
    Hamiltonian for a single Rydberg atom.

    Will extend to a chain of neutral atoms eventually.

    Returns a snapshot of the Hamiltonian at a single time step,
    provided the laser amplitude, phase, and detuning at that time
    instant.
    """
    term_1 = np.exp(1j * phase) * np.outer(np.array([1., 0]), np.array([0., 1.
                                                                       ]))
    H_control = (amplitude / 2.) * (term_1 + np.conj(term_1).T)
    H_detuning = detuning * np.outer(np.array([0., 1.]), np.array([0., 1.]))
    # Have no interaction term because only working with 1-qubit.
    return H_control + H_detuning


################################################################################################################


def unitary_step(time_step, amplitude, phase, detuning):
    time = dt * time_step
    print("time = ", time_step)
    U_slice = sp.linalg.expm(-1j * dt * hamiltonian(amplitude, phase, detuning))
    return U_slice


def parallel_unitary_evolution(waveform: np.ndarray, phase: np.ndarray,
                               detuning: np.ndarray, chunks: int) -> np.ndarray:
    """
    Calculates the unitary time evolution given the hamiltonian, a 
    waveform, and a number of time chunks.

    U(T) = U(T) * U(T-1) * U(T-2) * ... * U(t=0)
    where
    U(t) = exp(-1j * dt * hamiltonian(t))

    The evolved state vector will later on be calculated as:
    |psi(t)> = U(t) |psi_initial>
    """
    # Starting with the identity, then will multiply
    # from U(T) down to U(0)
    U_t = np.array([[1, 0], [0, 1]])

    pool = Pool()

    time_steps = np.flip(np.arange(start=0, stop=chunks, dtype=int))
    time_reverse_waveform = np.flip(waveform)
    time_reverse_phase = np.flip(phase)
    time_reverse_detuning = np.flip(detuning)
    _args = zip(time_steps, time_reverse_waveform, time_reverse_phase,
                time_reverse_detuning)
    unitary_matrices = pool.starmap(unitary_step, _args)

    # Close the process pool
    pool.close()
    pool.join()

    # TODO: Can partially paralellize this eventually.
    for unitary in unitary_matrices:
        U_t = np.matmul(U_t, unitary)
    return U_t

    print(f"U(T) =\n {np.matrix(np.round(np.abs(U_t), decimals=5))}")
    return U_t


################################################################################################################


def calculate_gate_fidelity(want_gate, got_gate):
    """
    Returns the overlap between the desired gate and the
    evolved gate. A value of 0.0 represents no overlap 
    and 1.0 represents perfect overlap.

    F = (1/(d**2)) * | tr(U_want^dag * U_got) | **2

    where d is the dimension of our gate (2)
    """
    fidelity = (1 / 4) * (np.abs(
        np.trace(np.dot(np.conj(want_gate).T, got_gate)))**2)
    print(f"fidelity = {fidelity}")
    return fidelity


def calculate_state_fidelity(want_state, got_state):
    """
    Returns the overlap between the desired state and the
    evolved state. A value of 0.0 represents no overlap 
    (orthogonal states), and 1.0 represents perfect overlap.

    F = |<psi_want | psi_got>| ** 2
    """
    fidelity = np.abs(np.dot(np.conj(want_state).T, got_state))**2
    return fidelity


################################################################################################################


def optimization_function(parameters: np.ndarray, *args):
    # In this case, we'll just let each individual parameter
    # represent the amplitude at a time chunk. So we will
    # have 1 parameter for each chunk.
    waveform = parameters[0:chunks]
    phase = parameters[chunks:chunks + chunks]
    detuning = parameters[chunks + chunks:len(parameters)]

    # Get the evolved unitary matrix.
    got_gate = parallel_unitary_evolution(waveform, phase, detuning, chunks)

    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_gate_fidelity(gate_to_optimize, got_gate)
    print(f"cost = {cost}")
    return cost


def run_optimization(want_gate: np.ndarray):
    """
    Closed loop optimization of the waveform:
    Will run the optimization on the provided gate
    """
    global gate_to_optimize
    gate_to_optimize = want_gate

    # Semi-arbitrary bounds on the amplitude of the waveform.
    lower_amplitude = [0.] * chunks
    upper_amplitude = [5. * np.pi] * chunks
    amplitude_bounds = list(zip(lower_amplitude, upper_amplitude))
    # amplitude_bounds = optimize.Bounds(lb=lower_amplitude, ub=upper_amplitude)

    # Phase bounds.
    lower_phase = [0.] * chunks
    upper_phase = [np.pi] * chunks
    phase_bounds = list(zip(lower_phase, upper_phase))
    # phase_bounds = optimize.Bounds(lb=lower_amplitude, ub=upper_amplitude)

    # Detuning bounds.
    lower_detuning = [0.] * chunks  # MHz.
    upper_detuning = [10.] * chunks  # MHz.
    detuning_bounds = list(zip(lower_detuning, upper_detuning))
    # detuning_bounds = optimize.Bounds(lb=lower_detuning, ub=upper_detuning)

    bounds = np.concatenate((amplitude_bounds, phase_bounds, detuning_bounds))

    # Just using random numbers to start with on our waveform.
    initial_waveform = np.random.uniform(low=lower_amplitude,
                                         high=upper_amplitude,
                                         size=(chunks,))
    initial_phases = np.random.uniform(low=lower_phase,
                                       high=upper_phase,
                                       size=(chunks,))
    initial_detunings = np.random.uniform(low=lower_detuning,
                                          high=upper_detuning,
                                          size=(chunks,))

    initial_controls = []
    initial_controls = np.concatenate(
        (initial_waveform, initial_phases, initial_detunings))

    optimized_result = optimize.minimize(optimization_function,
                                         initial_controls,
                                         args=(want_gate),
                                         bounds=bounds,
                                         method="Nelder-Mead")
    return optimized_result


################################################################################################################

# # Optimize for an X-rotation from an arbitrary initial state:
# want_gate = np.array([[0, 1], [1, 0]])

# result = run_optimization(want_gate)
# # Now let's take our final parameters (the waveform) and extract the optimized
# # unitary.
# x_waveform = result.x
# got_unitary = unitary_evolution(x_waveform, chunks)

# ################################################################################################################

# Let's try to optimize another gate!
hadamard_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
hadamard_result = run_optimization(hadamard_gate)
hadamard_waveform = hadamard_result.x
got_hadamard = unitary_evolution(hadamard_waveform, chunks)
# Let's evolve the initial state (|0>) and check how close it is to our desired state:
# (1/root(2) * [|0> + |1>]).
psi_want_hadamard = (1 / np.sqrt(2)) * np.array([1, 1])
psi_got_hadamard = np.abs(np.dot(got_hadamard, psi_initial))

# ################################################################################################################

# # Now let's optimize for a randomly generated Unitary matrix.

# random_gate = stats.unitary_group.rvs(2)
# random_result = run_optimization(random_gate)
# random_waveform = random_result.x
# got_random = unitary_evolution(random_waveform, chunks)
# # Let's evolve the initial state (|0>) and check how close it is to our desired state
# psi_want_random = np.abs(np.dot(np.conj(random_gate), psi_initial))
# psi_got_random = np.abs(np.dot(got_random, psi_initial))

# ################################################################################################################

# print(f"\n\n\nOptimized State (X-gate): {np.round(psi_got, decimals=5)}")
# print(f"Desired State (X-gate) {psi_want}")
# # And the state fidelity: F = |<psi_want | psi_got>| ** 2
# print("Optimized Fidelity (X-gate): ",
#       calculate_state_fidelity(np.abs(psi_want), np.abs(psi_got)))

# print(
#     f"\n\n\nOptimized State (Hadamard): {np.round(psi_got_hadamard, decimals=5)}"
# )
# print(f"Desired State (Hadamard) {psi_want_hadamard}")
# print("Optimized Fidelity: ",
#       calculate_state_fidelity(psi_want_hadamard, psi_got_hadamard))

# print(
#     f"\n\n\nOptimized State (Random Unitary): {np.round(psi_got_random, decimals=5)}"
# )
# print(f"Desired State (Random) {psi_want_random}")
# print("Optimized Fidelity: ",
#       calculate_state_fidelity(psi_want_random, psi_got_random))

# ################################################################################################################

# # Make pretty pictures of the waveforms

# time_values = np.linspace(0, T, chunks)

# fig, axs = plt.subplots(3)
# axs[0].set_title("Optimized X Waveform")
# axs[0].step(time_values, x_waveform)
# axs[1].set_title("Optimized Hadamard Waveform")
# axs[1].step(time_values, hadamard_waveform)
# axs[1].set_ylabel("Waveform Amplitude")
# axs[2].set_title("Optimized Random Waveform")
# axs[2].step(time_values, random_waveform)
# axs[2].set_xlabel("Time\n(N_time_chunks * dt = T)")
# fig.tight_layout(pad=1.0)

# plt.savefig("out.png", bbox_inches="tight", pad_inches=0.5)
