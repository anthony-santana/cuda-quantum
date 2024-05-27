# This is my first attempt at optimizing signals for a basic
# time-dependent, single-qubit spin hamiltonian within an analog CUDA-Q
# kernel, by simulating it with Qutip's `mesolve`.

# The goal is to find the signals that minimize the energy expectation
# of system with respect to Z.

import cudaq
from utils import *

import random
import numpy as np

from scipy import optimize

import matplotlib.pyplot as plt

# FIXME:
# Most important issue is that I'm not actually recompiling
# each time I generate new coefficients. I'm actually
# starting everything from scratch again with Qutip each time.
# But if I figure out a nice way of writing a `__recompile__`
# function, we should notice additional performance improvement.

global global_time_series
global_time_series = np.linspace(0.0, 10.0, 100)


@cudaq.analog_kernel
def kernel(waveform_x: np.ndarray, waveform_y: np.ndarray):
    # Build up the constant portion of the Hamiltonian:
    arbitrary_frequency = 1.0
    (arbitrary_frequency / 2.) * (I(0) - Z(0))

    # Now let's build up the control portion of the
    # Hamiltonian:
    waveform_x * X(0)
    waveform_y * Y(0)


global costs
costs = []


def objective_function(x):
    # TODO: is it possible to get gradient information back
    #       from qutip to pass back off to an optimizer???
    waveform_x, waveform_y = np.split(np.asarray(x), 2)
    # Returns the evolutino propagator at the final time, T.
    unitary = cudaq.sample(kernel,
                           time_steps=global_time_series,
                           waveform_x=waveform_x,
                           waveform_y=waveform_y,
                           verbose=False)
    # Hard-coding our desired gate as an X-gate.
    want_gate = np.fliplr(np.eye(2**kernel.qubit_count))
    fidelity = cudaq.operator.gate_fidelity(want_gate, unitary)
    print(f"fidelity = {fidelity}")
    print(f"cost = {1.0 - fidelity}\n")

    costs.append(float('%.5f' % (1.0 - fidelity)))

    # Make a pretty plot.
    plt.clf()
    plt.plot(costs)
    plt.savefig("out.png")

    return 1.0 - fidelity


# Optimizer configuration.
waveform_count = 2
bounds = [(0.0, 1.0) for _ in range(len(global_time_series) * waveform_count)]
initial_control_signal = np.random.uniform(low=0.0,
                                           high=1.0,
                                           size=(len(global_time_series) *
                                                 waveform_count,))

optimized_result = optimize.dual_annealing(
    func=objective_function,
    x0=initial_control_signal,
    # accept=-1e4,  #1.05,  # -1e4 # min of range
    visit=2.9,  #1.05,  # range \in (0,3)
    # The temperature should start at a value
    # that's roughly the order of the difference
    # between local minima cost values.
    initial_temp=0.35,  # FIXME: Try next: 0.01
    restart_temp_ratio=0.99999,
    #    no_local_search=True,
    bounds=bounds)

# optimized_result = optimize.minimize(
#     objective_function,
#     initial_control_signal,
#     bounds=bounds,
#     #  method="SLSQP")
#     #  method="trust-constr")
#     method="Nelder-Mead")

# NOTE:
# You could imagine a sequential waveform that
# takes place on one of the qubits looking like this:
#   `2 * waveform_x * X(0)`
# This would be a new operation on the X axis of qubit 0,
# that would have to be scheduled for after the first
# waveform. The total `time_steps` passed to observe
# would have to account for this as well.

# 5/27, 10:15 AM
# - Trying with initial_temp=`0.35`. If that doesn't look great,
#   go back and then try changing accept to -5
