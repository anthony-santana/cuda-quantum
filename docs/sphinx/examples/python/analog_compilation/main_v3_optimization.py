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
    hamiltonian = cudaq.Hamiltonian(qubit_count = 1)

    # Build up the constant portion of the Hamiltonian:
    arbitrary_frequency = 1.0
    hamiltonian += (arbitrary_frequency / 2.) * (I(0) - Z(0))

    # Now let's build up the control portion of the
    # Hamiltonian:
    hamiltonian += waveform_x * X(0)
    hamiltonian += waveform_y * Y(0)


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
    visit=1.05,  # 2.9,  # range \in (1,3]
    # The temperature should start at a value
    # that's roughly the order of the difference
    # between local minima cost values.
    initial_temp=0.35,  # FIXME: Try next: 0.01
    restart_temp_ratio=0.99999,
    #    no_local_search=True,
    bounds=bounds)
