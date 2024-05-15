import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import functools
import itertools
from multiprocess import Pool

import cudaq

######################################### Timing Parameters ###################################################

T = 2.0  # total time, T in microseconds.
dt = 0.125  # time duration of each signal chunk in microseconds.
global chunks
chunks = int(T / dt)  # number of time chunks we will solve for

################################################################################################################


def gaussian_square_signal(amplitude, square_sample_count, sigma):
    """ 
    Args
    ----
        signal_sample_count : the number of samples of the entire signal
                              Keeping this value fixed for now.
        amplitude : amplitude of flattop portion of signal
        square_sample_count : the number of samples of the square portion
        sigma : width of gaussian risefall
    

    Square wave with Gaussian rise/fall.

            exp( -0.5 * ((x - risefall)**2)/(sigma**2))         , x < risefall
    f'(x) = 1                                                   , risefall <= x < risefall + width
            exp( -0.5 * ((x - (risefall+width))**2)/(sigma**2)) , risefall + width <= x
    
    f(x) = amplitude * (f_prime(x) - f_prime(-1)) / (1. - f_prime(-1))
    """

    signal_sample_count = chunks

    risefall = (signal_sample_count - square_sample_count) / 2.

    def f_prime(x):
        if (x < risefall):
            return np.exp(-0.5 * ((x - risefall)**2) / (sigma**2))
        elif ((risefall + square_sample_count) <= x):
            return np.exp(-0.5 * ((x - (risefall + square_sample_count))**2) /
                          (sigma**2))
        return 1

    def f(x):
        return amplitude * (f_prime(x) - f_prime(-1)) / (1. - f_prime(-1))

    return [f(x) for x in np.arange(start=0.0, stop=signal_sample_count)]


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

    detunings = np.split(flipped_parameters[0:qubit_count * chunks],
                         qubit_count)

    phases = np.split(
        flipped_parameters[qubit_count * chunks:2 * qubit_count * chunks],
        qubit_count)

    # Parameter packing convention:
    # """
    # T = 1.25  # total time, T in microseconds.
    # dt = 0.25  # time duration of each signal chunk in microseconds.
    # chunks = int(T / dt)  # number of time chunks we will solve for

    # def gaussian_square_signal(amplitude, signal_sample_count, square_sample_count,
    #                         sigma):
    # """
    signal_parameters = np.split(np.flip(flipped_parameters[2 * qubit_count * chunks:]), qubit_count)
    signals = [gaussian_square_signal(*_arg) for _arg in signal_parameters]
    # print(signals)

    # for signal in signals:
    #     plt.plot(signal)
    # plt.savefig(f"out.png")

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

    # parameters:
    # for each qubit have:
    # [(gaussian_signal_i, ...,  phase_signal_i, ..., detuning_signal_i, ...)]
    # <==>
    # [ ( (amplitude, square_width, sigma), ..., ( chunks_width_signal ), ..., (chunks_width_signal), ... )]

    # Gaussian signal bounds.
    # Bounds on the amplitude of the laser.
    lower_amplitude = [0.]
    upper_amplitude = [5. * np.pi]
    # Will use this value to divide the total width.
    lower_square_width = [1.]
    upper_square_width = [10.]  # arbitrary
    # Sigma bounds.
    lower_sigma = [1.]
    upper_sigma = [2.]  # arbitrary
    # Packed up bounds.
    # [ (amplitude_0, width_0, sigma_0), ..., (amplitude_n, width_n, sigma_n) ]
    gaussian_lower_bounds = (lower_amplitude + lower_square_width +
                             lower_sigma) * qubit_count
    gaussian_upper_bounds = (upper_amplitude + upper_square_width +
                             upper_sigma) * qubit_count
    amplitude_bounds = list(zip(gaussian_lower_bounds, gaussian_upper_bounds))

    # print(amplitude_bounds)

    # Phase bounds.
    lower_phase = [0.] * (qubit_count * chunks)
    upper_phase = [np.pi] * (qubit_count * chunks)
    phase_bounds = list(zip(lower_phase, upper_phase))

    # Detuning bounds.
    lower_detuning = [0.] * (qubit_count * chunks)  # MHz.
    upper_detuning = [16.33] * (qubit_count * chunks)  # MHz.
    detuning_bounds = list(zip(lower_detuning, upper_detuning))

    bounds = (amplitude_bounds + phase_bounds + detuning_bounds)

    # Just using random parameter values for our initial Gaussians.
    initial_amplitudes = np.random.uniform(low=lower_amplitude,
                                           high=upper_amplitude,
                                           size=(qubit_count,))
    initial_square_widths = np.random.uniform(low=lower_square_width,
                                              high=upper_square_width,
                                              size=(qubit_count,))
    initial_sigmas = np.random.uniform(low=lower_sigma,
                                       high=upper_sigma,
                                       size=(qubit_count,))
    initial_gaussian_parameters = np.column_stack(
        (initial_amplitudes, initial_square_widths, initial_sigmas)).flatten()
    # Just using random numbers to start with on our signals.
    initial_phases = np.random.uniform(low=lower_phase,
                                       high=upper_phase,
                                       size=(qubit_count * chunks,))
    initial_detunings = np.random.uniform(low=lower_detuning,
                                          high=upper_detuning,
                                          size=(qubit_count * chunks,))

    initial_controls = []
    initial_controls = np.concatenate(
        (initial_gaussian_parameters, initial_phases, initial_detunings))

    # optimization_function(initial_controls)

    optimized_result = optimize.minimize(optimization_function,
                                         initial_controls,
                                         args=(want_gate),
                                         bounds=bounds,
                                        #  method="SLSQP")
                                        #  method="trust-constr")
                                         method="Nelder-Mead")

    # optimized_result = optimize.dual_annealing(func=optimization_function,
    #                                            x0=initial_controls,
    #                                            bounds=bounds)
    return optimized_result


################################################################################################################

qubit_count = 3

######################################## X-180 rotation of system ##############################################

# Now let's optimize for an X-rotation
x_180_rotation = np.fliplr(np.eye(2**qubit_count))
x_180_result = run_optimization(x_180_rotation)
x_180_signal = x_180_result.x
print("final X-180 cost = ", x_180_result.fun, "\n")

########################################## Hadamard operation on system ##########################################

# single_hadamard_operation = (1 / np.sqrt(2)) * np.array([[1., 1.], [1., -1.]])
# hadamard_operation = broadcast_single_qubit_operation(single_hadamard_operation,
#                                                       qubit_count)
# hadamard_operation_result = run_optimization(hadamard_operation)
# hadamard_operation_signal = hadamard_operation_result.x
# print("final Hadamard cost = ", hadamard_operation_result.fun, "\n")