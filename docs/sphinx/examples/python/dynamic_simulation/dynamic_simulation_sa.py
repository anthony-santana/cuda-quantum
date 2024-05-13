import numpy as np
import scipy as sp
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

import itertools
# from multiprocessing import Pool
from multiprocess import Pool

################################################################################################################

# A very simple introduction to controlling the evolution
# of a quantum system through optimized input controls.

# Ok let's make a system Hamiltonian that we will have to solve
# for to achieve our desired unitary gate.

# A static term containing information about our ground
# state frequency. Setting it to 1 for simplicity.
omega_0 = 1.  # arbitrary value in hz.

# Define the parameters of the waveform. It will be discretized
# over a time duration (`T`), with each discrete sample of its
# amplitude lasting for a time duration of (`dt`).
# In real life, something like the `dt` would be determined by
# the slew rate of the physical waveform generator in the lab.
# So we will chop the total time of our waveform up into equally
# sized (`dt`), discrete time chunks.
# You can play around with these values to see how they impact the
# actual evolution of the system.
T = 5.  # total time, T. aritrary value in s.
dt = 0.25  # time duration of each waveform chunk. arbitrary value in s.
chunks = int(T / dt)  # number of time chunks we will solve for


# Time-dependent hamiltonian for system, including control waveform.
# Will just consider a piecewise-constant waveform.
def hamiltonian(amplitude: float, time: float) -> np.ndarray:
    """
    Takes the waveform amplitude at a single time slice and returns
    the Hamiltonian matrix at that time.

    hamiltonian = H_static + H_control
    H_static = [0,0]
                [0, omega_0]
    H_control = [0, waveform_amplitude_xy]
                [waveform_amplitude_xy, 0]

    where:
    amplitude_x = amplitude * np.cos(omega_0 * time)
    amplitude_y = amplitude * np.sin(omega_0 * time)
    """
    H_static = np.array([[0, 0], [0, omega_0]])
    # H(t) = amplitude_x(t) * X + amplitude_y(t) * Y
    H_t = (amplitude * np.cos(omega_0 * time)) * np.array([[0, 1], [1, 0]]) + (
        amplitude * np.sin(omega_0 * time)) * np.array([[0, -1j], [1j, 0]])
    return H_static + H_t


################################################################################################################


def unitary_evolution(waveform: np.ndarray, chunks: int) -> np.ndarray:
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

    # Begin the calculation at time T and work backwards
    for time_step in reversed(range(chunks)):
        # Multiply our unitary by the Hamiltonian at that time step,
        # given the evovled waveform.
        time = dt * time_step
        U_slice = sp.linalg.expm(-1j * dt *
                                 hamiltonian(waveform[time_step], time))
        U_t = np.matmul(U_t, U_slice)

    print(f"U(T) =\n {np.matrix(np.round(np.abs(U_t), decimals=5))}")
    return U_t


def unitary_step(time_step, hamiltonian):
    time = dt * time_step
    U_slice = sp.linalg.expm(-1j * dt * hamiltonian(waveform[time_step], time))
    return U_slice


def parallel_unitary_evolution(waveform: np.ndarray, chunks: int) -> np.ndarray:
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
    hamiltonian_repeat = itertools.repeat(hamiltonian, chunks)
    _args = zip(time_steps, hamiltonian_repeat)
    unitary_matrices = pool.starmap(unitary_step, _args)
    for unitary in unitary_matrices:
        U_t = np.matmul(U_t, unitary)
    return U_t

    # Close the process pool
    pool.close()
    pool.join()

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
    waveform = parameters

    # Get the evolved unitary matrix.
    got_gate = unitary_evolution(waveform, chunks)

    # Calculate the fidelity and return it as a cost (1 - fidelity)
    # so that our optimizer can minimize the function.
    cost = 1. - calculate_gate_fidelity(want_gate, got_gate)
    return cost


def run_optimization(unitary_gate: np.ndarray):
    """
    Closed loop optimization of the waveform:
    Will optimize the waveform to produce the provided `unitary_gate`.
    """
    # Semi-arbitrary bounds on the amplitude of the waveform.
    lower = [-1.] * chunks
    upper = [1.] * chunks
    bounds = optimize.Bounds(lb=lower, ub=upper)
    global want_gate
    want_gate = unitary_gate
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
        no_local_search=True,
        maxiter=100)
    return optimized_result


################################################################################################################

# System setup:
# Begin in |0> state
psi_initial = np.array([0, 1])

# Would like to evolve to the |1> state
psi_want = np.array([1, 0])

# To achieve this, the unitary matrix must be an X-gate:
want_gate = np.array([[0, 1], [1, 0]])

# Just as a sanity check that we get the desired state by
# applying that unitary
# |psi_test> = U * |psi_0>
psi_test = np.dot(want_gate, psi_initial)
print(psi_test)
assert (np.allclose(psi_want, psi_test))

# Let's run the optimization for our X-gate:
result = run_optimization(want_gate)

# Now let's take our final parameters (the waveform) and extract the optimized
# unitary.
x_waveform = result.x
got_unitary = unitary_evolution(x_waveform, chunks)
# Let's evolve the initial state (|0>) and check how close it is to our desired state (|1>).
psi_got = np.abs(np.dot(got_unitary, psi_initial))

################################################################################################################

# Let's try to optimize another gate!
hadamard_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
hadamard_result = run_optimization(hadamard_gate)
hadamard_waveform = hadamard_result.x
got_hadamard = unitary_evolution(hadamard_waveform, chunks)
# Let's evolve the initial state (|0>) and check how close it is to our desired state:
# (1/root(2) * [|0> + |1>]).
psi_want_hadamard = (1 / np.sqrt(2)) * np.array([1, 1])
psi_got_hadamard = np.abs(np.dot(got_hadamard, psi_initial))

################################################################################################################

# Now let's optimize for a randomly generated Unitary matrix.

random_gate = stats.unitary_group.rvs(2)
random_result = run_optimization(random_gate)
random_waveform = random_result.x
got_random = unitary_evolution(random_waveform, chunks)
# Let's evolve the initial state (|0>) and check how close it is to our desired state
psi_want_random = np.abs(np.dot(np.conj(random_gate), psi_initial))
psi_got_random = np.abs(np.dot(got_random, psi_initial))

################################################################################################################

print(f"\n\n\nOptimized State (X-gate): {np.round(psi_got, decimals=5)}")
print(f"Desired State (X-gate) {psi_want}")
# And the state fidelity: F = |<psi_want | psi_got>| ** 2
print("Optimized Fidelity (X-gate): ",
      calculate_state_fidelity(np.abs(psi_want), np.abs(psi_got)))

print(
    f"\n\n\nOptimized State (Hadamard): {np.round(psi_got_hadamard, decimals=5)}"
)
print(f"Desired State (Hadamard) {psi_want_hadamard}")
print("Optimized Fidelity: ",
      calculate_state_fidelity(psi_want_hadamard, psi_got_hadamard))

print(
    f"\n\n\nOptimized State (Random Unitary): {np.round(psi_got_random, decimals=5)}"
)
print(f"Desired State (Random) {psi_want_random}")
print("Optimized Fidelity: ",
      calculate_state_fidelity(psi_want_random, psi_got_random))

################################################################################################################

# Make pretty pictures of the waveforms

time_values = np.linspace(0, T, chunks)

fig, axs = plt.subplots(3)
axs[0].set_title("Optimized X Waveform")
axs[0].step(time_values, x_waveform)
axs[1].set_title("Optimized Hadamard Waveform")
axs[1].step(time_values, hadamard_waveform)
axs[1].set_ylabel("Waveform Amplitude")
axs[2].set_title("Optimized Random Waveform")
axs[2].step(time_values, random_waveform)
axs[2].set_xlabel("Time\n(N_time_chunks * dt = T)")
fig.tight_layout(pad=1.0)

plt.savefig("out.png", bbox_inches="tight", pad_inches=0.5)
