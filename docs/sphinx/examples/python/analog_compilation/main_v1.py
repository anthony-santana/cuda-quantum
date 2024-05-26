# This is my first attempt at generating a time-dependent,
# single-qubit spin hamiltonian within an analog CUDA-Q
# kernel, then simulating it with Qutip's `mesolve`.

import cudaq
from utils import *

import random
import numpy as np


def custom_envelope_function_x(time: float):
    arbitrary_frequency = 1.0
    return random.randint(0, 1) * np.cos(arbitrary_frequency * 2 * np.pi * time)


def custom_envelope_function_y(time: float):
    arbitrary_frequency = 1.0
    return random.randint(0, 1) * np.sin(arbitrary_frequency * 2 * np.pi * time)


@cudaq.analog_kernel
def kernel(waveform_x: callable, waveform_y: callable):
    # Build up the constant portion of the Hamiltonian:
    arbitrary_frequency = 1.0
    (arbitrary_frequency / 2.) * (X(0) - I(0))

    # Now let's build up the control portion of the
    # Hamiltonian:
    waveform_x * X(0)
    waveform_y * Y(0)

result = cudaq.observe(kernel,
                       time_steps=np.linspace(0.0, 2.0, 20),
                       waveform_x=custom_envelope_function_x,
                       waveform_y=custom_envelope_function_y,
                       verbose=True)
print(result)
