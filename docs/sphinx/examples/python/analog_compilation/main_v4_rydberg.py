import cudaq
from utils import *

import random
import numpy as np

from scipy import optimize

# FIXME:
# Most important issue is that I'm not actually recompiling
# each time I generate new coefficients. I'm actually
# starting everything from scratch again with Qutip each time.
# But if I figure out a nice way of writing a `__recompile__`
# function, we should notice additional performance improvement.

# TODO
# Optimize to the unitary instead of to some state evolved from |0>

global global_time_series
global_time_series = np.linspace(0.0, 10.0, 20)


@cudaq.analog_kernel
def kernel(qubit_count: int, signals: list[np.ndarray], phases: list[float],
           detunings: list[float]):
    # Constant Hamiltonian terms.
    # FIXME: Just using the same fixed distance for each atom in
    # a chain right now.
    interaction_energy_c6 = 862690 * 2 * np.pi  # MHz * micrometer^6
    atom_distance = 4  # micrometer
    V_ij = interaction_energy_c6 / (atom_distance**6)

    # Time-dependent Hamiltonian terms.
    # Apply the operators to each individual qubit. This will
    # send a signal along with phase/detuning down to each qubit.
    for qubit in range(qubit_count):
        signal = signals[qubit]
        phase = phases[qubit]
        detuning = detunings[qubit]

        ((signal / 2.) * np.exp(1j * phase)) * 0.5 * (X(qubit) -
                                                      (1j * Y(qubit)))
        ((signal / 2.) * np.exp(-1j * phase)) * 0.5 * (X(qubit) -
                                                       (1j * Y(qubit)))
        (-1 * detuning) * 0.5 * (I(qubit) - Z(qubit))

    # Interaction terms.
    for j in range(qubit_count):
        for i in range(j - 1):
            # FIXME:
            # Need to rework the entire `cudaq.operator` implementation
            # to be able to handle arithmetic between operators that are
            # on different qubits.
            (V_ij * 0.5) * (I(i) - Z(i)) * ()


def objective_function(x, *args):
    # TODO: is it possible to get gradient information back
    #       from qutip to pass back off to an optimizer???
    pass

    # expectation = cudaq.observe(kernel,
    #                             time_steps=global_time_series,
    #                             qubit_count=args[0],
    #                             verbose=False)
    # print("cost = ", expectation)
    # # TODO: Use a different metric than the expectation for the
    # # cost, such that I can optimize to produce desired arbitrary
    # # unitary operations.
    # return expectation


# Optimizer configuration.
qubit_count = 2
bounds = [(0.0, 1.0) for _ in range(len(global_time_series) * qubit_count)]
initial_control_signal = np.random.uniform(low=0.0,
                                           high=1.0,
                                           size=(len(global_time_series) *
                                                 qubit_count,))

optimized_result = optimize.dual_annealing(func=objective_function,
                                           x0=initial_control_signal,
                                           args=(qubit_count,),
                                           accept=1.25,
                                           visit=1.25,
                                           no_local_search=True,
                                           bounds=bounds)
