import cudaq
# from cudaq import h, x, t, swap, ry, r1, mz, spin
import numpy as np 

# Simple kernel sampling
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
cudaq.sample(bell).dump()

# def bell_():
#     kernel = cudaq.make_kernel()
#     q = kernel.qalloc(2)
#     kernel.h(q[0])
#     kernel.cx(q[0], q[1])
#     return kernel

# # Simple kernel sampling, control flow
# @cudaq.kernel
# def ghz(size):
#     q = cudaq.qvector(size)
#     h(q[0])
#     for i in range(size-1):
#         x.ctrl(q[i], q[i+1])

# cudaq.sample(ghz, 10).dump()

# # Simple kernel sampling, mid-circ measurement
# @cudaq.kernel
# def midCirc():
#     q = cudaq.qvector(2)
#     h(q[0])
#     i = mz(q[0], "c0")
#     if i:
#         x(q[1])
#     mz(q)

# cudaq.sample(midCirc).dump()

# # Can perform spin_op observation
# @cudaq.kernel
# def ansatz(angle):
#     q = cudaq.qvector(2)
#     x(q[0])
#     ry(angle, q[1])
#     x.ctrl(q[1], q[0])

# hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
#     0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

# result = cudaq.observe(ansatz, hamiltonian, .59)
# print(result.expectation_z())

# # More complexity!

# @cudaq.kernel 
# def iqft(qubits):
#     N = qubits.size()
#     for i in range(N//2):
#         swap(qubits[i], qubits[N-i-1])
    
#     for i in range(N-1):
#         h(qubits[i])
#         j = i + 1
#         for y in range(i, -1, -1):
#             r1.ctrl(-np.pi / 2**(j-y), qubits[j], qubits[y])
    
#     h(qubits[N-1])

# @cudaq.kernel 
# def tGate(qubit):
#     t(qubit)

# @cudaq.kernel 
# def xGate(qubit):
#     x(qubit)

# @cudaq.kernel
# def qpe(nC, nQ, statePrep, oracle):
#     q = cudaq.qvector(nC+nQ)
#     countingQubits = q.front(nC)
#     stateRegister = q.back()
#     statePrep(stateRegister)
#     h(countingQubits)
#     for i in range(nC):
#         for j in range(2**i):
#             cudaq.control(oracle, [countingQubits[i]], stateRegister)
#     iqft(countingQubits)
#     mz(countingQubits)

# cudaq.sample(qpe, 3, 1, xGate, tGate).dump()