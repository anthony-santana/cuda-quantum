import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import functools
import itertools
from multiprocess import Pool

# import optax
# import tensorflow as tf
# import tensorflow_probability as tfp

import cudaq

# import cupy as cp
# from cupyx.scipy.linalg import expm

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
# 8. Area underneath signal should be angle of single qubit rotation

# NEED TO Have an (x,y) coordinate pair for each qubit
# This will be used to do the distance calcualtions between
# them for the interaction terms.

# When thinking about rewriting with CUDA in C++, much of
# these calculations can just stay on the GPU and all I should
# need to bring back to the CPU is the actual cost value.

######################################### Timing Parameters ###################################################

# T = 0.25  # total time, T in microseconds.
# dt = 0.25  # time duration of each signal chunk in microseconds.
# chunks = int(T / dt)  # number of time chunks we will solve for

T = 4.0  # total time, T in microseconds.
global chunks
chunks = 30  # number of time chunks (samples) contained in our signals
global dt
dt = T / chunks  # time duration of each signal chunk in microseconds.

################################################################################################################


def single_qubit_hamiltonian(time, qubit, signal_amplitude, phase, detuning):
    """ 
    Returns the spin hamiltonian for a single term, at a single time step,
    in an array of Rydberg atoms. 
    
    Args
    ----
        `qubit` : the index of the qubit to create the Hamiltonian term for.
        `signal_amplitude` : the amplitude of this qubits signal/laser at this
                               time step.
        `phase` : the phase of the laser at this time step.
        `detuning` : the laser detuning at this time step.                      
    """
    basis_term_01 = 0.5 * (cudaq.spin.x(qubit) - (1j * cudaq.spin.y(qubit)))
    basis_term_01_conj = (0.5 * (cudaq.spin.x(qubit) +
                                 (1j * cudaq.spin.y(qubit))))
    basis_n_i = 0.5 * (cudaq.spin.i(qubit) - cudaq.spin.z(qubit))
    phase_factor = np.exp(1j * phase)
    # Define the parameterized Hamiltonian using CUDA-Q spin operators.
    single_qubit_hamiltonian = ((signal_amplitude / 2.) *
                                ((phase_factor * basis_term_01) +
                                 (np.conj(phase_factor) * basis_term_01_conj)))
    single_qubit_hamiltonian -= (detuning * basis_n_i)
    return single_qubit_hamiltonian


def hamiltonian(time: float,
                amplitudes: tuple[float],
                phases: tuple[float],
                detunings: tuple[float],
                V_ij: float = 1.0) -> np.ndarray:
    """
    Hamiltonian for multiple Rydberg atoms (currently only support chain).

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
    # FIXME: Just using the same fixed distance for each atom in
    # a chain right now.
    interaction_energy_c6 = 862690 * 2 * np.pi  # MHz * micrometer^6
    atom_distance = 4  # micrometer
    V_ij = interaction_energy_c6 / (atom_distance**6)

    hamiltonian = 0.0
    for qubit in range(qubit_count):
        signal_amplitude = amplitudes[qubit]
        phase = phases[qubit]
        detuning = detunings[qubit]
        hamiltonian += single_qubit_hamiltonian(time, qubit, signal_amplitude,
                                                phase, detuning)

    # FIXME: This term doesn't change across time steps so I should
    #        really calculate it once and just insert it here every time.

    # Don't really need individual functions here for the n i and j terms
    # since they're the same with different indices. But it's helpful to be
    # explicit while I work this code out.
    basis_n_i = lambda i_qubit: 0.5 * (cudaq.spin.i(i_qubit) - cudaq.spin.z(
        i_qubit))
    basis_n_j = lambda j_qubit: 0.5 * (cudaq.spin.i(j_qubit) - cudaq.spin.z(
        j_qubit))
    # Interaction term is: `Sum (i < j) V_ij * n_i * n_j` .
    for j in range(qubit_count):
        for i in range(j - 1):
            hamiltonian += (V_ij * basis_n_i(i) * basis_n_j(j))
    return np.asarray(hamiltonian.to_matrix())


################################################################################################################


def unitary_step(time_step, amplitudes, phases, detunings):
    time = dt * time_step
    U_slice = sp.linalg.expm(-1j * dt *
                             hamiltonian(time, amplitudes, phases, detunings))
    # print(U_slice)
    return U_slice


def parallel_unitary_evolution(signals: np.ndarray, phases: np.ndarray,
                               detunings: np.ndarray,
                               chunks: int) -> np.ndarray:
    """
    Calculates the unitary time evolution given the hamiltonian, a 
    signal, and a number of time chunks.

    Args
    ----
        `signals` : array of tuple of signal samples. 
                      tuple contains a sample for each control .
        `phases` : array of each tuple of laser phases
        `detunings` : array of each tuple of laser detunings
        `chunks` : the number of time steps.


    Uses Trotter-Suzuki decomposition:

    U(T) = U(T) * U(T-1) * U(T-2) * ... * U(t=0)
    where
    U(t) = exp(-1j * dt * hamiltonian(t))

    The evolved state vector will later on be calculated as:
    |psi(t)> = U(t) |psi_initial>
    """
    # Setting the total number of processes to 1 for now
    # since it's quicker when only doing very coarse time
    # evolution for a small number of time steps.
    # pool = Pool(processes=1)

    time_steps = np.flip(np.arange(start=0, stop=chunks, dtype=int))

    # TODO:
    # I've already passed the signals and everything into here
    # split up and stacked so I don't have to do any extra formatting.
    # May move that code logic here in the future but idk.
    _args = zip(time_steps, signals, phases, detunings)
    # unitary_matrices = pool.starmap(unitary_step, _args)
    unitary_matrices = [unitary_step(*_arg) for _arg in _args]

    # # Close the process pool
    # pool.close()
    # pool.join()

    # Multiply through all unitary slices to get full time evolution.
    U_T = functools.reduce(np.matmul, unitary_matrices)

    # print(f"U(T) =\n {np.matrix(np.round(np.abs(U_T), decimals=5))}")
    return U_T


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
    # numpy_parameters = parameters.numpy()
    # flipped_parameters = np.flip(numpy_parameters)
    flipped_parameters = np.flip(parameters)
    # TODO: Write this code in a more efficient manner. There's a lot of data
    #       splitting, merging, stacking, etc. that could be avoided with better
    #       code.
    # The flattened array of parameters comes into us with the leftmost entry
    # being the first sample of the signal on qubit 0, and the rightmost
    # entry being the last sample of the detuning "signal" on qubit N.
    # To properly time order these when simulating time evolution, we flip the
    # parameters array and partition the parameters to their respective location.
    detunings = np.split(flipped_parameters[0:qubit_count * chunks],
                         qubit_count)
    phases = np.split(
        flipped_parameters[qubit_count * chunks:2 * qubit_count * chunks],
        qubit_count)
    signals = np.split(
        flipped_parameters[2 * qubit_count * chunks:len(parameters)],
        qubit_count)

    # Stack all of the simulataneous samples across all qubits into
    # tuples so we can paralellize.
    stacked_detunings = np.column_stack(tuple(detunings))
    stacked_phases = np.column_stack(tuple(phases))
    stacked_signal_samples = np.column_stack(tuple(signals))

    # Get the evolved unitary matrix.
    got_gate = parallel_unitary_evolution(stacked_signal_samples,
                                          stacked_phases, stacked_detunings,
                                          chunks)

    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_gate_fidelity(gate_to_optimize, got_gate)
    print(f"cost = {cost}")
    # return tf.cast(cost, tf.float64)
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

    # Just using random numbers to start with on our signal.
    initial_signal = np.random.uniform(low=lower_amplitude,
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
        (initial_signal, initial_phases, initial_detunings))

    # initial_controls_cast = tuple([tf.constant(value, tf.float64) for value in initial_controls])
    # # Optimizing as one large variable that I manually unpack into
    # # the proper place. This is to keep this script compatible with
    # # scipy optimizers, but could be reworked at some point.
    # controls_to_optimize = tf.Variable(initial_controls)
    # print(controls_to_optimize)
    # optim_results = tfp.optimizer.differential_evolution_minimize(
    #     optimization_function,
    #     initial_position=initial_controls_cast,
    #     seed=43210)
    # print(optim_results.converged)

    optimized_result = optimize.minimize(
        optimization_function,
        initial_controls,
        args=(want_gate),
        bounds=bounds,
        #  method="SLSQP")
        #  method="trust-constr")
        method="Nelder-Mead")

    # optimized_result = optimize.dual_annealing(func=optimization_function,
    #                                            x0=initial_controls,
    #                                            bounds=bounds)  #,
    # visit=1.25,
    # no_local_search=True)
    return optimized_result


def broadcast_single_qubit_operation(single_qubit_operation: np.ndarray,
                                     qubit_count: int):
    """
    Returns the matrix for a single-qubit gate, broadcast
    to be applied to the entire Hilbert space.
    """
    return functools.reduce(np.kron, [single_qubit_operation] * qubit_count)


def broadcast_individual_operations(operations: list[np.ndarray]):
    # TODO: Need to broadcast each against identity operations
    # before I add them each up.
    full_gate = []
    single_qubit_identity = np.eye(2)
    broadcast_list = [single_qubit_identity] * len(operations)
    print("operations = ", operations)
    print(broadcast_list)
    for operation_index, operation in enumerate(operations):
        print("index = ", operation_index)
        broadcast_list[operation_index] = operation
        print(broadcast_list)
        # print(functools.reduce(np.kron, broadcast_list))
        full_gate.append(functools.reduce(np.kron, broadcast_list))
        broadcast_list[operation_index] = single_qubit_identity
    return sum(full_gate)


################################################################################################################

qubit_count = 2

######################################## X-180 rotation of system ##############################################

# # Now let's optimize for an X-rotation
# x_180_rotation = np.fliplr(np.eye(2**qubit_count))
# x_180_result = run_optimization(x_180_rotation)
# x_180_signal = x_180_result.x
# print("final X-180 cost = ", x_180_result.fun, "\n")

########################################## Hadamard operation on system ##########################################

# single_hadamard_operation = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])
# hadamard_operation = broadcast_single_qubit_operation(single_hadamard_operation,
#                                                       qubit_count)
# hadamard_operation_result = run_optimization(hadamard_operation)
# hadamard_operation_signal = hadamard_operation_result.x
# print("final Hadamard cost = ", hadamard_operation_result.fun, "\n")

################################################## CZ-Gate #####################################################

# Now let's optimize for a CZ gate
cz_operation = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                         [0., 0., 0., -1.]])
cz_result = run_optimization(cz_operation)
cz_signal = cz_result.x
print("final CZ cost = ", cz_result.fun, "\n")

########################################## Operations on isolated atoms ##########################################

# FIXME: Produces the wrong operation

# single_x_180_rotation = np.fliplr(np.eye(2))
# single_hadamard_operation = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])
# # Will apply an X-180 rotation to every other atom, and a Hadamard operation
# # to the untouched atoms.
# individual_operation_list = [None] * qubit_count
# for qubit in range(qubit_count):
#     if ((qubit % 2) != 0):
#         individual_operation_list[qubit] = single_hadamard_operation
#     else:
#         individual_operation_list[qubit] = single_x_180_rotation
# full_operation = broadcast_individual_operations(individual_operation_list)
# print(full_operation)

# zero = np.array([1.,0.,0.,0.])
# print("\n", np.dot(full_operation, zero), "\n")

# # result = run_optimization(full_operation)
# # signal = result.x
# # print("final mixed gates cost = ", result.fun, "\n")

################################################################################################################

# # Now let's optimize for a randomly generated Unitary matrix.
# random_gate = stats.unitary_group.rvs(2**qubit_count)
# random_result = run_optimization(random_gate)
# random_signal = random_result.x
# print("final cost = ", random_result.fun)

# ################################################################################################################

# # Make pretty pictures of the signals

# time_values = np.linspace(0, T, chunks)

# fig, axs = plt.subplots(3)
# axs[0].set_title("Optimized X Signal")
# axs[0].step(time_values, x_signal)
# axs[1].set_title("Optimized Hadamard Signal")
# axs[1].step(time_values, hadamard_signal)
# axs[1].set_ylabel("signal Amplitude")
# axs[2].set_title("Optimized Random Signal")
# axs[2].step(time_values, random_signal)
# axs[2].set_xlabel("Time\n(N_time_chunks * dt = T)")
# fig.tight_layout(pad=1.0)

# plt.savefig("out.png", bbox_inches="tight", pad_inches=0.5)
