import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import functools
import itertools
from multiprocess import Pool

import cudaq

# Find the signals, phases, and detunings for N-Rydberg atoms in an
# array that minimizes the energy expectation value of the system.

######################################### Timing Parameters ###################################################

T = 1.  # total time, T in microseconds.
dt = 0.25  # time duration of each signal chunk in microseconds.
chunks = int(T / dt)  # number of time chunks we will solve for

################################################################################################################


def single_qubit_hamiltonian(qubit, signal_amplitude, phase, detuning):
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


def hamiltonian(amplitudes: tuple[float],
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
        hamiltonian += single_qubit_hamiltonian(qubit, signal_amplitude, phase,
                                                detuning)

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
                             hamiltonian(amplitudes, phases, detunings))
    # TODO: Would like to register the unitaries with CUDA-Q here to avoid
    # having to loop through every matrix again, but it goes out of scope
    # after return.
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
    pool = Pool(processes=1)

    time_steps = np.flip(np.arange(start=0, stop=chunks, dtype=int))

    # TODO:
    # I've already passed the signals and everything into here
    # split up and stacked so I don't have to do any extra formatting.
    # May move that code logic here in the future but idk.
    _args = zip(time_steps, signals, phases, detunings)
    unitary_matrices = pool.starmap(unitary_step, _args)

    # Close the process pool
    pool.close()
    pool.join()

    initial_name = "custom_operation_"
    for unitary, time_step in zip(unitary_matrices, time_steps):
        cudaq.register_operation(unitary,
                                 operation_name=initial_name + str(time_step))
    return cudaq.globalRegisteredUnitaries.keys()


def get_energy_expectation(unitary_operation_names):
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qubit_count)
    # Loop through and apply the registered unitary operations
    # to the kernel.
    for unitary_operation in unitary_operation_names:
        execution_string = f'''
qubit_args = []
for qubit_index in range({qubits.size()}):
  qubit_args.append(qubits[qubit_index])
kernel.{unitary_operation}(*qubit_args)
'''
        exec(execution_string)
    # Sample the kernel and return its expectation value.
    return cudaq.sample(kernel).expectation()


################################################################################################################


def optimization_function(parameters: np.ndarray):
    # Flipping these ahead of time for time ordering, since they'll all
    # be nested away and more inconvenient to flip then.
    # numpy_parameters = parameters.numpy()
    # flipped_parameters = np.flip(numpy_parameters)
    flipped_parameters = np.flip(parameters)
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

    # Get the name for each unitary time step that we will
    # apply to our system.
    unitary_operations = parallel_unitary_evolution(stacked_signal_samples,
                                                    stacked_phases,
                                                    stacked_detunings, chunks)

    print("energy = ", get_energy_expectation(unitary_operations))

    return get_energy_expectation(unitary_operations)


def run_optimization(_qubit_count: int):
    """
    Closed loop optimization of the control signals:
    Will run the optimization for the provided unitary operation
    """
    global qubit_count
    qubit_count = _qubit_count

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

    # optimized_result = optimize.minimize(optimization_function,
    #                                      initial_controls,
    #                                      bounds=bounds,
    #                                      method="Nelder-Mead")

    optimized_result = optimize.dual_annealing(func=optimization_function,
                                               x0=initial_controls,
                                               bounds=bounds)  #,
    # visit=1.25,
    # no_local_search=True)
    return optimized_result


######################################## Minimize Energy #######################################################

qubit_count = 6
energy_minimized_signals = run_optimization(qubit_count)
print("final minimized energy = ", energy_minimized_signals.fun, "\n")
