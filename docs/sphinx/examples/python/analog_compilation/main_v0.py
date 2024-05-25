import cudaq
from utils import *

import numpy as np

@cudaq.analog_kernel
def kernel(waveform: np.ndarray, constant_coefficients: list[float]):
    constant_coefficients[0] * X(0)
    constant_coefficients[1] * Y(1)
    constant_coefficients[2] * Z(2)


"""
TODO: 

1.  Figure out how to overload the arrays * the spin terms to represent
    a function that is applied at various time steps for a QobjEvo.

    For this prototype, it will likely require having some way to pass around
    the functional coefficient with the cudaq::spin_op until we need to convert
    it to a QobjEvo.

"""



result = cudaq.observe(kernel,
                       waveform=None,
                       constant_coefficients=[1.0, 2.0, 3.0],
                       debug=True)