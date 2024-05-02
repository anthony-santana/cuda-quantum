#include <chrono>
#include <cudaq.h>
#include <iostream>

// If you have a NVIDIA GPU you can use this example to see
// that the GPU-accelerated backends can easily handle a
// larger number of qubits compared the CPU-only backend.

// Depending on the available memory on your GPU, you can
// set the number of qubits to around 30 qubits, and set the
// `--target=nvidia` from the command line.

// Note: Without setting the target to the `nvidia` backend,
// there will be a noticeable decrease in simulation performance.
// This is because the CPU-only backend has difficulty handling
// 30+ qubit simulations.

std::vector<int> random_bitstring(int qubit_count) {
  std::vector<int> vector_of_bits;
  for (auto i = 0; i < qubit_count; i++) {
    // Populate our vector of bits with random binary
    // values (base 2).
    vector_of_bits.push_back(rand() % 2);
  }
  return vector_of_bits;
}

__qpu__ void oracle(cudaq::qview<> qvector, cudaq::qview<> auxillary_qubit,
                    std::vector<int> &hidden_bitstring) {
  for (auto i = 0; i < hidden_bitstring.size(); i++) {
    if (hidden_bitstring[i] == 1)
      // Apply a `cx` gate with the current qubit as
      // the control and the auxillary qubit as the target.
      x<cudaq::ctrl>(qvector[i], auxillary_qubit[0]);
  }
}

__qpu__ void bernstein_vazirani(std::vector<double> &initial_state,
                                std::vector<double> &auxillary_state,
                                std::vector<int> &hidden_bitstring) {
  // Allocate the specified number of qubits - this
  // corresponds to the length of the hidden bitstring.
  // and an extra auxillary qubit.
  cudaq::qvector qvector(initial_state);
  cudaq::qvector auxillary_qubit(auxillary_state);

  // Query the oracle.
  oracle(qvector, auxillary_qubit, hidden_bitstring);

  // Apply another set of Hadamards to the qubits.
  h(qvector);

  // Apply measurement gates to just the `qubits`
  // (excludes the auxillary qubit).
  mz(qvector);
}

int main() {
  auto iterations = 500;
  auto qubit_counts = {
      3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20}; //,21,22,23,24,25,26,27,28,29,30};

  for (auto qubit_count : qubit_counts) {
    std::vector<double> times;

    // For now, including this calculation in the timing loop itself:

    // // Initial state to allocate the system in (Hadamard state).
    // auto value = 1. / sqrt(pow(2, qubit_count));
    // std::vector<double> initialState(pow(2, qubit_count), value);
    // // Initial state for the auxillary qubit (Hadamard + Z).
    // std::vector<double> auxillaryState = {1. / sqrt(2), -1. / sqrt(2)};

    for (auto iteration = 0; iteration < iterations; iteration++) {
      auto start = std::chrono::high_resolution_clock::now();

      // Initial state to allocate the system in (Hadamard state).
      auto value = 1. / sqrt(pow(2, qubit_count));
      std::vector<double> initialState(pow(2, qubit_count), value);
      // Initial state for the auxillary qubit (Hadamard + Z).
      std::vector<double> auxillaryState = {1. / sqrt(2), -1. / sqrt(2)};

      // Generate a bitstring to encode and recover with our algorithm.
      auto hidden_bitstring = random_bitstring(qubit_count);

      auto result = cudaq::sample(bernstein_vazirani, initialState,
                                  auxillaryState, hidden_bitstring);

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

      // result.dump();
      times.push_back(duration.count());
    }
    double average_time =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average time for " << std::to_string(qubit_count)
              << " qubits: " << std::to_string(average_time)
              << " microseconds.\n";
  }
  return 0;
}
