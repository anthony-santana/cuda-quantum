import cudaq

import time
import numpy as np


def generate_bitlist(input_size, secret_int):
    s = ('{0:0' + str(input_size) + 'b}').format(secret_int)
    bitlist = list(s)
    # Have to convert the data type from strings to ints.
    return [int(bit) for bit in bitlist]


@cudaq.kernel
def oracle(register: cudaq.qview, ancilla_qubit: cudaq.qubit,
           hidden_bits: list[int]):
    for index, bit in enumerate(hidden_bits):
        if bit == 1:
            # apply a `cx` gate with the current qubit as
            # the control and the auxillary qubit as the target.
            x.ctrl(register[index], ancilla_qubit)


# TODO: Replicate qiskit version 2 of BV, that uses mid-circuit measurements.


@cudaq.kernel
def bernstein_vazirani(hidden_bits: list[int]):
    # Allocate the specified number of qubits - this
    # corresponds to the length of the hidden bitstring.
    qr = cudaq.qvector(len(hidden_bits))
    # Allocate an extra ancilla qubit.
    ancilla_qubit = cudaq.qubit()

    # Put the ancilla in |1> state.
    x(ancilla_qubit)

    # Start with Hadamard on all qubits, including ancilla.
    h(qr)
    h(ancilla_qubit)

    # Generate the oracle.
    oracle(qr, ancilla_qubit, hidden_bits)

    # Hadamard on all qubits, including the ancilla.
    h(qr)
    h(ancilla_qubit)

    # Apply measurement gates to just the data qubits, `qr`
    # (excludes the auxillary qubit).
    mz(qr)


# def analyze_and_print_result():
#     pass


def run(min_qubits=2,
        max_qubits=25,
        max_circuits=3,
        input_value=None,
        shots_count=100):
    # Execute benchmark program N times for multiple circuit sizes.
    # TODO: Accumulate metrics asynchronously as circuits complete.
    for qubit_count in range(min_qubits, max_qubits):
        input_size = qubit_count

        # Determine number of circuits to execute for this group.
        circuit_count = min(2**(input_size), max_circuits)

        print(
            f"\n\n************\nExecuting [{circuit_count}] circuits with num_qubits = {qubit_count}"
        )

        # Determine range of secret strings to loop over.
        if 2**(input_size) <= max_circuits:
            s_range = list(range(circuit_count))
        else:
            s_range = np.random.choice(2**(input_size), circuit_count, False)

        # Loop over limited number of secret strings for this.
        for secret_int in s_range:

            # If user specifies input_value, use it instead.
            if input_value is not None:
                secret_int = input_value

            # Create the kernel for the given qubit size and secret string.
            secret_string = generate_bitlist(input_size, secret_int)
            print(secret_string)
            # Execute the circuit on the given target (default: `nvidia`) and
            # print the execution time.
            start = time.time()
            result = cudaq.sample(bernstein_vazirani,
                                  secret_string,
                                  shots_count=shots_count)
            stop = time.time()
            print(
                f"Kernel creation, compilation, and execution time: {stop-start}s."
            )
            print(f"encoded bitstring = {secret_string}")
            print(f"measured state = {result.most_probable()}")
            print(
                f"Were we successful? {''.join([str(i) for i in secret_string]) == result.most_probable()}"
            )
            print("\n")


# Run the entire benchmarking loop, printing the results to terminal.
run()

# Print a small example of a Bernstein Vazirani circuit for visualization.
print(
    f"Six-qubit Berstein Vazirani circuit:\n{cudaq.draw(bernstein_vazirani, [0,0,0,1,1,1])}"
)

# TODO: Use the metrics module to run the benchmarks and post-process.
