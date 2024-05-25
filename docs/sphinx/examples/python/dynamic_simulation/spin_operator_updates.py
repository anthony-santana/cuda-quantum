import cudaq
from cudaq import spin

# Spin op features we should support to enable more general Hamiltonian definitions:
"""
Suggestions for adjustments to current CUDA-Q language-level infrastructure.

1. Move `cudaq::spin_op` to living under the umbrella of `cudaq::operator::spin_op`,
   while still maintaining the ability to call it as a `cudaq::spin_op`.
   We can reroute things to maintain the current includes/imports in user code,
   but this will allow all generalized operators to live under the same umbrella.
   This wouldn't require any language level changes to the spin op and, ideally,
   could be done in a way that doesn't break anything.

    Example:
    ```
    # Still works:
    from cudaq import spin
    # Also works:
    from cudaq import operator
    from cudaq.operator import spin
    ```

2. Work with the cuDynamics classes for calculating expectation values in similar way
   we currently do.

    ```
    operator = cudaq.spin ... + ... 
    parameters = [waveforms, phases, ...]
    # Note that we don't need a kernel in this case.
    expectation = cudaq.observe(operator, parameters)
    ```
  
   The observe call delegates to the `cuDynamics` backend, specifically the `cudmExpectation_t`
   class for calculating expectation values of an operator given the set of parameters.

   Supporting this in this way may involve a slight adjustment to how we currently define
   the `observe` function, but in my opinion, it's the most obvious choice to overload for
   executing these operator based simulations.



TODO:
Unitary creation in CUDA-Q to access time evolver from cuDynamics



New operator classes:

1. Bra-ket algebra. Examples:
      |0_{i}> <1_{i}| ==>  `cudaq::operator::ket::zero(qubit_i) * cudaq::bra::one(qubit_i)`

2. Creation/Annihilation Operators:
      cudaq::operator::creation(n)
      cudaq::operator::annihilation(n)




```

Pseudo-Quake

Note: this would ideally occur at the point in which we've flattened the
      user kernel to just plain old gates with no other logic. Likely at
      the point in which we'd currently have our QIR.

// Starts a global clock with a given number of total discrete
// time steps for the program. These may no be linearly spaced
// time steps.

global_clock(number_of_total_time_steps_in_program)

{ 
  start=global_clock[0],
  stop=(global_clock[0] + signal.size()),
  address=qubits[0],
  signal= /* some cc.stdvec<i64> of samples` */, 
  phase= /* some cc.stdvec<i64> of phase values. same length as signal samples */,
  
  ... other hardware dependent parameters that are provided as samples. Ex, 
      laser detuning ...

  operation_name='x' // optional
}

{ 
  start=global_clock[1],
  stop=(global_clock[1] + signal.size()),
  address=qubits[1],
  signal= /* some cc.stdvec<i64> of samples` */, 
  phase= /* some cc.stdvec<i64> of phase values. same length as signal samples */,
  operation_name='h' // optional
}

.
.
.


```

"""
