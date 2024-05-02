#include <chrono>
#include <cudaq.h>
#include <iostream>


__qpu__ void ghz(std::vector<double> &initial_state)  {
  cudaq::qvector q(initial_state);
  mz(q);
}

int main() {
  auto iterations = 1000;
  auto qubit_counts = {
      3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  std:: cout << "Benchmarking direct state method, GHZ: \n";
  for (auto qubit_count : qubit_counts) {
    std::vector<double> times;
    for (auto iteration = 0; iteration < iterations; iteration++) {
      // Create the vector for the GHZ state.
      std::vector<double> initial_state(pow(2, qubit_count), 0.0);
      initial_state.front() = 1 / sqrt(2);
      initial_state.back() = 1 / sqrt(2);

      // Time execution of passing the pre-made state vector to
      // the kernel and execution of the kernel.
      auto start = std::chrono::high_resolution_clock::now();
      
      auto result = cudaq::sample(ghz, initial_state);

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      times.push_back(duration.count());
      // result.dump();
    }
    double average_time =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "Average time for " << std::to_string(qubit_count)
              << " qubits: " << std::to_string(average_time)
              << " microseconds.\n";
  }
  std::cout << "\n";
}