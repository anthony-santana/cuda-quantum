import qutip
import numpy as np

# Basic time-independent problem:
# We're going to define the Hamiltonian of the actual system
# that we're trying to simulate the dynamics of.
hamiltonian = (2 * np.pi * 0.1) * qutip.sigmax()

# Define the initial state for the system ==> |0>
initial_state = qutip.basis(2, 0)

# The operator we will take the expectation value with respect to:
expectation_operator = qutip.sigmaz()

# Define the discrete time steps for the numerical integration.
time_steps_for_integration = np.linspace(0.0, 10.0, 20)

# Delegate the program to the Qutip master equation solver,
# which will solve the following differential equation:
"""
   dU(t)
  ------ = -i * H(t) * U(t = 0)
    dt

    solution:

  U(t) = exp( integral_{0,T}(H(t)) )

    where U(t) will be evaluated at the provided discrete time steps:

  U(t = 0), U(dt), U(2 * dt),
"""
result = qutip.sesolve(H=hamiltonian,
                        psi0=initial_state,
                        tlist=time_steps_for_integration,
                        e_ops=expectation_operator)

# Print the final expectation value of the system:
print(f"<Z> = {result.expect[0][-1]}\n")
