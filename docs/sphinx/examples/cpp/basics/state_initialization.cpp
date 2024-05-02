// Compile and run with:
// ```
// nvq++ state_initialization.cpp -o out.x && ./out.x
// ```

#include <cudaq.h>
#include <vector>

// This example demonstrates a very simple introduction to the
// allocation of qubits in a user-provided initial state.

__qpu__ void kernel(std::vector<double> initial_state) {
  // Allocate a set of qubits in the provided `initial_state`.
  cudaq::qvector q(initial_state);

  // Can now operate on the qvector as usual:
  // Rotate state of the front qubit 180 degrees along X.
  x(q.front());
  // Rotate state of the back qubit 180 degrees along Y.
  y(q.back());
  // Put qubits into superposition state.
  h(q);

  // Measure.
  mz(q);
}

int main() {
  // Allocating 2 qubits in the `11` state.
  std::vector<double> initial_state = {0., 0., 0., 1.};

  // Our kernel will start with 2 qubits in `11`, then
  // rotate each qubit back to `0` before applying a
  // Hadamard gate.
  auto result = cudaq::sample(kernel, initial_state);

  // Example expected result: { 00:250 10:250 01:250 11:250 }
  result.dump();
}