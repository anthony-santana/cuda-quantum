import qutip
import random
import numpy as np

# Define the discrete time steps for the numerical integration.
time_steps_for_integration = np.linspace(0.0, 10.0, 20)

# Basic time-dependent problem:
hamiltonian_operator = qutip.sigmax()

# Now we define a set of time-dependent coefficients in a function
# that will be invoked at each time step.
def hamiltonian_coefficients(time):
    return random.randint(0,1)

# We pass the operator with its coefficients as the full operator
# to Qutip.
"""
    H(t) = coefficients(t) * X(0)
"""
hamiltonian = [[hamiltonian_operator, hamiltonian_coefficients]]

# Initial state and expectation operators.
expectation_operator = qutip.sigmaz()
initial_state = qutip.basis(2, 0)

# Delegate the program to the Qutip master equation solver.
result = qutip.sesolve(H=hamiltonian,
                        psi0=initial_state,
                        tlist=time_steps_for_integration,
                        e_ops=expectation_operator)

# Print the final expectation value of the system:
print(f"<Z> = {result.expect[0][-1]}\n")