# This presents an example of defining a time-independent
# spin hamiltonian within an analog kernel, then "lowering"
# from the CUDA-Q representation to Qutip data-types.
# The qutip `SESolver` is then used as the target simulator
# for the new analog `cudaq::observe` overload.

import cudaq
from utils import *

import numpy as np


@cudaq.analog_kernel
def kernel(waveform: np.ndarray, waveform_function: callable,
           constant_coefficients: list[float]):

    constant_coefficients[0] * X(0)
    constant_coefficients[1] * Y(1)
    constant_coefficients[2] * Z(2)

    # Some control term that is `waveform[t] * X(qubit)` where the `waveform`
    # is an array of signal amplitude values.
    waveform * X(0)
    waveform * Y(1)
    # Some control term that is `waveform[t] * Y(qubit)` where the `waveform`
    # is a function that is called for each t, and returns the signal amplitude
    # at that time.
    waveform_function * Y(0)
    waveform_function * X(1)

    # The idea is that the user just expresses which operator the entire
    # signal will act upon, then we will handle generating them as time-
    # dependent operator coefficients for them.


# Both test waveform formats just return constant values of 1.0
waveform = np.ones(20)
waveform_function = lambda t: 1.0
constants = [1.0, 2.0, 3.0]

result = cudaq.observe(kernel,
                       time_steps=np.linspace(0.0, 2.0, 20),
                       waveform=waveform,
                       waveform_function=waveform_function,
                       constant_coefficients=constants,
                       verbose=True)
