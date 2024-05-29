import qutip
import numpy as np

# Define the discrete time steps for the numerical integration.
time_steps_for_integration = np.linspace(0.0, 10.0, 20)

# Basic time-dependent problem:
hamiltonian_term_1 = qutip.sigmax()
hamiltonian_term_2 = qutip.sigmaz()

# Now we define a set of time-dependent coefficients that
# we will apply to the hamiltonian.
# NOTE: How there is one coefficient per integration time step.
term_1_coefficients = np.linspace(0.0, 5.0, 20)
term_2_coefficients = np.linspace(0.0, 10.0, 20)

# We pass the operator with its coefficients as the full operator
# to Qutip.
"""
    H(t) = coefficients(t) * X(0) + coefficients(t) * Z(0)
"""
hamiltonian = [
  [hamiltonian_term_1, term_1_coefficients],
  [hamiltonian_term_2, term_2_coefficients]
]

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