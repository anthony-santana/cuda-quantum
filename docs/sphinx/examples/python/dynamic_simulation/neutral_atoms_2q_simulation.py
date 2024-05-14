import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import itertools
from multiprocess import Pool

import cudaq

# TODO:
# 1. Speed up certain calculations in parallel
# 2. Get rid of inneficiencies in certain calculations
#    (such as how I pack and unpack params)
# 3. Move linear algebra to cupy for speed up
# 4. Make the actual qubit positions an optimizable parameter
# 5. Create data types to make the code more generally understandable
# 6. Model their common sources of noise within the Hamiltonian to create
#    a more realistic simulation.
# 7. Translate all of this to C++ code written in CUDA
# 8. Area underneath waveform should be angle of single qubit rotation

# NEED TO Have an (x,y) coordinate pair for each qubit
# This will be used to do the distance calcualtions between
# them for the interaction terms.

######################################### Timing Parameters ###################################################

T = 3.5  # total time, T. aritrary value in s.
dt = 0.25  # time duration of each waveform chunk. arbitrary value in s.
chunks = int(T / dt)  # number of time chunks we will solve for

################################################################################################################


def single_qubit_hamiltonian(qubit, waveform_amplitude, phase, detuning):
    """ """
    basis_term_01 = 0.5 * (cudaq.spin.x(qubit) - (1j * cudaq.spin.y(qubit)))
    basis_term_01_conj = (0.5 * (cudaq.spin.x(qubit) +
                                 (1j * cudaq.spin.y(qubit))))
    basis_n_i = 0.5 * (cudaq.spin.i(qubit) - cudaq.spin.z(qubit))
    phase_factor = np.exp(1j * phase)
    # Define the parameterized Hamiltonian using CUDA-Q spin operators.
    single_qubit_hamiltonian = ((waveform_amplitude / 2.) *
                                ((phase_factor * basis_term_01) +
                                 (np.conj(phase_factor) * basis_term_01_conj)))
    single_qubit_hamiltonian -= (detuning * basis_n_i)
    return single_qubit_hamiltonian


def hamiltonian(amplitudes: tuple[float],
                phases: tuple[float],
                detunings: tuple[float],
                V_ij: float = 1.0) -> np.ndarray:
    """
    Hamiltonian for multiple Rydberg atoms.

    --> This means I'm redefining the spin op at every time step from
        scratch which is certainly time inneficient.

    --> Would it be more efficient to calculate all time steps of this
        at once, while the grunt work has already been done, store them
        away, them access them 1 at a time later when I need them.  

    Returns a snapshot of the Hamiltonian at a single time step,
    provided each lasers amplitude, phase, and detuning at that time
    instant.


    FIXME:
    I'm currently paralellizing the function that calls this,
    so I can't paralellize this as well. If it turns out that
    outer loop is faster than I expect, I can flip it.
    """
    hamiltonian = 0.0
    for qubit in range(qubit_count):
        waveform_amplitude = amplitudes[qubit]
        phase = phases[qubit]
        detuning = detunings[qubit]
        hamiltonian += single_qubit_hamiltonian(qubit, waveform_amplitude,
                                                phase, detuning)

    # Don't really need individual functions here for the n i and j terms
    # since they're the same with different indices. But it's helpful to be
    # explicit while I work this code out.
æ
    return np.asarray(hamiltonian.to_matrix())


################################################################################################################


def unitary_step(time_step, amplitudes, phases, detunings):
    time = dt * time_step
    print("time = ", time_step)
    U_slice = sp.linalg.expm(-1j * dt *
                             hamiltonian(amplitudes, phases, detunings))
    return U_slice


def parallel_unitary_evolution(waveforms: np.ndarray, phases: np.ndarray,
                               detunings: np.ndarray,
                               chunks: int) -> np.ndarray:
    """
    Calculates the unitary time evolution given the hamiltonian, a 
    waveform, and a number of time chunks.

    Accepts:
    `waveforms` : array of tuple of waveform samples. tuple contains a sample for each control .
    `phases` : array of each tuple of laser phases
    `detunings` : array of each tuple of laser detunings
    `chunks` : the number of time steps.


    U(T) = U(T) * U(T-1) * U(T-2) * ... * U(t=0)
    where
    U(t) = exp(-1j * dt * hamiltonian(t))

    The evolved state vector will later on be calculated as:
    |psi(t)> = U(t) |psi_initial>
    """
    # Starting with the identity, then will multiply
    # from U(T) down to U(0)
    U_t = np.identity(2**qubit_count)

    pool = Pool()

    time_steps = np.flip(np.arange(start=0, stop=chunks, dtype=int))

    # TODO:
    # I've already passed the waveforms and everything into here
    # split up and stacked so I don't have to do any extra formatting.
    # May move that code logic here in the future but idk.
    _args = zip(time_steps, waveforms, phases, detunings)
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

    where d is the dimension of our gate
    """
    fidelity = (1 / want_gate.size) * (np.abs(
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
    # Flipping these ahead of time for time ordering, since they'll all
    # be nested away and more inconvenient to flip then.
    flipped_parameters = np.flip(parameters)
    # TODO: Write a comment about the convention I used to unpack these.
    # TODO: Write this code in a more efficient manner. There's a lot of data
    #       splitting, merging, stacking, etc. that could be avoided with better
    #       code.
    detunings = np.split(flipped_parameters[0:qubit_count * chunks],
                         qubit_count)
    phases = np.split(
        flipped_parameters[qubit_count * chunks:2 * qubit_count * chunks],
        qubit_count)
    waveforms = np.split(
        flipped_parameters[2 * qubit_count * chunks:len(parameters)],
        qubit_count)

    # Stack all of the simulataneous samples across all qubits into
    # tuples so we can paralellize.
    stacked_detunings = np.column_stack(tuple(detunings))
    stacked_phases = np.column_stack(tuple(phases))
    stacked_waveform_samples = np.column_stack(tuple(waveforms))

    # Get the evolved unitary matrix.
    got_gate = parallel_unitary_evolution(stacked_waveform_samples,
                                          stacked_phases, stacked_detunings,
                                          chunks)

    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_gate_fidelity(gate_to_optimize, got_gate)
    print(f"cost = {cost}")
    return cost


def run_optimization(want_gate: np.ndarray):
    """
    Closed loop optimization of the control signals:
    Will run the optimization for the provided unitary operation
    """
    global gate_to_optimize
    gate_to_optimize = want_gate
    global qubit_count
    qubit_count = int(np.log2(want_gate.shape[0]))

    # TODO:
    # 1. Lessen the parameter landscape by picking a fixed value
    #    for things like the phase, etc. and doing small modulations
    #    to it (smaller parameter range).

    # Bounds on the amplitude of the laser.
    lower_amplitude = [0.] * (qubit_count * chunks)
    upper_amplitude = [5. * np.pi] * (qubit_count * chunks)
    amplitude_bounds = list(zip(lower_amplitude, upper_amplitude))

    # Phase bounds.
    lower_phase = [0.] * (qubit_count * chunks)
    upper_phase = [np.pi] * (qubit_count * chunks)
    phase_bounds = list(zip(lower_phase, upper_phase))

    # Detuning bounds.
    lower_detuning = [0.] * (qubit_count * chunks)  # MHz.
    upper_detuning = [16.33] * (qubit_count * chunks)  # MHz.
    detuning_bounds = list(zip(lower_detuning, upper_detuning))

    bounds = (amplitude_bounds + phase_bounds + detuning_bounds)

    # Just using random numbers to start with on our waveform.
    initial_waveform = np.random.uniform(low=lower_amplitude,
                                         high=upper_amplitude,
                                         size=(qubit_count * chunks,))
    initial_phases = np.random.uniform(low=lower_phase,
                                       high=upper_phase,
                                       size=(qubit_count * chunks,))
    initial_detunings = np.random.uniform(low=lower_detuning,
                                          high=upper_detuning,
                                          size=(qubit_count * chunks,))

    initial_controls = []
    initial_controls = np.concatenate(
        (initial_waveform, initial_phases, initial_detunings))

    optimization_function(initial_controls)

    # optimized_result = optimize.minimize(optimization_function,
    #                                      initial_controls,
    #                                      args=(want_gate),
    #                                      bounds=bounds,
    #                                      method="Nelder-Mead")
    # return optimized_result


################################################################################################################

# Now let's optimize for a randomly generated Unitary matrix.
qubit_count = 3
random_gate = stats.unitary_group.rvs(2**qubit_count)
random_result = run_optimization(random_gate)
random_waveform = random_result.x
got_random = unitary_evolution(random_waveform, chunks)
# Let's evolve the initial state (|0>) and check how close it is to our desired state
psi_want_random = np.abs(np.dot(np.conj(random_gate), psi_initial))
psi_got_random = np.abs(np.dot(got_random, psi_initial))

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
