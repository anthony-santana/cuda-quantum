# This is a reproduction of the example found in the Qutip
# documentation for time-dependent equation solvers.
# https://qutip.readthedocs.io/en/master/guide/dynamics/dynamics-master.html

# This confirms that we reproduce the same results for this
# very basic problem.

import cudaq
from utils import *

import numpy as np


@cudaq.analog_kernel
def kernel():
    hamiltonian = cudaq.Hamiltonian(qubit_count=1)
    hamiltonian += (2 * np.pi * 0.1) * X(0)


result = cudaq.observe(kernel,
                       time_steps=np.linspace(0.0, 10.0, 20),
                       verbose=True)
print(result)
