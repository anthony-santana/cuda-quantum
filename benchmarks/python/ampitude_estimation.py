import cudaq

import math
import numpy as np

cudaq.set_random_seed(0)
np.random.seed(0)


def generate_a(secret_integer, counting_qubit_count):
    theta = secret_integer * np.pi / (2**counting_qubit_count)
    return np.sin(theta)**2


@cudaq.kernel
def amplitude_estimation(state_qubit_count: int, counting_qubit_count: int,
                         a_value: float):
    # Calculate total qubit count.
    qubit_count = state_qubit_count + 1 + counting_qubit_count

    # Allocate quantum memory.
    state_register = cudaq.qvector(state_qubit_count + 1)
    counting_register = cudaq.qvector(counting_qubit_count + 1)

    # Construct the A operator.

    # Start by creating our state strings.
    # FIXME: Further MLIR support could make this string concatenation
    # easier.
    psi_zero = [0 for _ in range(state_qubit_count)]
    psi_one = [1 for _ in range(state_qubit_count)]

    # FIXME: Further MLIR support could reduce this to a call to
    # `2 * np.arcsin()``
    theta = 2. * math.arcsin(math.sqrt(a_value))
    
    for qubit in state_register:
        # Take `|0_{qubit}>` to `sqrt(1-a) |0> + sqrt(a) |1>`
        ry(theta, qubit)
        # Take ``
        x(qubit)
        if (psi_zero[qubit] == 1):
            x.ctrl()


secret_integer = 4
state_qubit_count = 1
counting_qubit_count = 1

a_value = generate_a(secret_integer, counting_qubit_count)
print(a_value)

result = cudaq.sample(amplitude_estimation, state_qubit_count,
                      counting_qubit_count, a_value)
print(result)
