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

    hamiltonian = cudaq.Hamiltonian(2)

    # Build up the constant portion of the Hamiltonian:
    arbitrary_frequency = 1.0
    hamiltonian += (arbitrary_frequency / 2.) * (I(0) - Z(0))

    # Now let's build up the control portion of the
    # Hamiltonian:
    hamiltonian += waveform_x * X(0)
    hamiltonian += waveform_y * Y(0)

    hamiltonian += waveform_x_ * X(0)
    hamiltonian += waveform_y_ * Y(0)

    # NOTE:
    # You could imagine another sequential waveform that
    # takes place on one of those qubits looking like this:
    #   `waveform_x_1 * X(0)`
    # This would be a new operation on the X axis of qubit 0,
    # that would have to be scheduled for after the first
    # waveform. The total `time_steps` passed to observe
    # would have to account for this as well.


result = cudaq.observe(kernel,
                       time_steps=np.linspace(0.0, 2.0, 20),
                       waveform_x=custom_envelope_function_x,
                       waveform_y=custom_envelope_function_y,
                       verbose=False)
print(result)
