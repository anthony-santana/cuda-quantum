import cudaq

import numpy as np
from utils import *

@cudaq.analog_kernel
def kernel(waveform: np.ndarray, constant_coefficients: list[float]):
    constant_coefficients[0] * X(0)
    constant_coefficients[1] * Y(1)
    constant_coefficients[2] * Z(2)


result = cudaq.observe(kernel,
                       waveform=None,
                       constant_coefficients=[1.0, 2.0, 3.0],
                       debug=True)