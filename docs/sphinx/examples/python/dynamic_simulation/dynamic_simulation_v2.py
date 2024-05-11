import cudaq

import random
import numpy as np

from new_utility_functions import *

################################### TODO #######################################

# 1. Write out an optimization loop for optimizing these pulses -- this will confront
#    any issues with creating new unitary's each step.
# 2. Multiple controls on a single qubit
# 3. 2-qubit Hamiltonians/Unitaries --> produce a GHZ-state on one of these
# 4. Grab a realistic small Hamiltonian from a provider and test it out.
# 5. Think about how to make things like the time variable amenable to them
#    being an optimized parameter as well. Total time, number of time chunks, etc.

########################### Hamiltonian Definition #############################

# As of right now, I'm just building up the Hamiltonian as a lambda function. But
# you could imagine just extending the cudaq.SpinOperator to be able to handle
# parameterized terms, time variables, etc. Then you could build up a bigger
# SpinOperator the way you typically would. I chose the lambda over a typical
# function as to make it closer to the arithmetic-like way of building up spin ops.

# The only major difference is that, in it's current form, I have the Hamiltonian
# lambda set up to return the Hamiltonian snapshot as a numpy matrix.

# A static term containing information about our ground
# state frequency. Setting it to 1 for simplicity.
omega_0 = 1.0  # arbitrary value in hz.

# `cudaq.Time`
time_variable = cudaq.Time()
time_variable.max_time = 5.  # total time, T. arbitrary value in s.
time_variable.resolution = 0.25  # time duration for each chunk of time evolution. arbitrary value in s.

# `cudaq.ControlSignal`
control_signal = cudaq.ControlSignal(time=time_variable)
# Could also use `control_signal.set_sample_function` to provide a function instead
# of an array of sample values.
control_signal.set_sample_values(
    np.random.rand(len(time_variable.time_series())))

# `H_constant = [[0,0], [0, omega_0]]`
H_constant = (omega_0 / 2) * (cudaq.spin.i(0) - cudaq.spin.z(0))
# `H_control = waveform_amplitude_x * X + waveform_amplitude_y * Y``
#       where `waveform_amplitude_x = amplitude * np.cos(omega_0 * time)`
#       and   `waveform_amplitude_y = amplitude * np.sin(omega_0 * time)`
Hamiltonian = lambda t: np.asarray((H_constant + (control_signal(t) * np.cos(
    omega_0 * time_variable(t))) * cudaq.spin.x(0) + (control_signal(
        t) * np.sin(omega_0 * time_variable(t))) * cudaq.spin.y(0)).to_matrix())

################################################################################

# We get the synthesized unitarys back as a dict with the keys
# being the registered unitary name and value being their matrix.
unitary_operations = cudaq.synthesize_unitary(Hamiltonian, time_variable)

# Allocate a qubit to a kernel that we will apply
# the time evolution operators to.
kernel = cudaq.make_kernel()
qubit = kernel.qalloc()

# Loop through and apply the registered unitary operations
# to the kernel.
for unitary_operation in unitary_operations.keys():
    evaluation_string = "kernel." + unitary_operation + "(qubit)"
    eval(evaluation_string)

final_state = cudaq.get_state(kernel)
print(final_state)

################################################################################
