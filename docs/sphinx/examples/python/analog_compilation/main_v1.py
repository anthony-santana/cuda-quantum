import cudaq
from utils import *

import numpy as np


def custom_envelope_function(time: float):
    return random.randint(0, 1)


@cudaq.analog_kernel
def kernel(waveform: callable):
    # Build up the constant portion of the Hamiltonian:
    arbitrary_frequency = 1.0
    (arbitrary_frequency / 2.) * (X(0) * I(0))
    # (arbitrary_frequency / 2.) * (I(0) - Z(0))


print(np.asarray((0.5 * (cudaq.spin.x(0))).to_matrix()))

result = cudaq.observe(kernel,
                       waveform=custom_envelope_function,
                       time_steps=np.linspace(0.0, 2.0, 20),
                       verbose=True)
print(result)
