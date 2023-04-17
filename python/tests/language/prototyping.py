import cudaq
from cudaq import h, x, ry, mz, spin

# Simple kernel sampling
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
cudaq.sample(bell).dump()

# Simple kernel sampling, control flow
@cudaq.kernel
def ghz(size):
    q = cudaq.qvector(size)
    h(q[0])
    for i in range(size-1):
        x.ctrl(q[i], q[i+1])

cudaq.sample(ghz, 10).dump()

# Simple kernel sampling, mid-circ measurement
@cudaq.kernel
def midCirc():
    q = cudaq.qvector(2)
    h(q[0])
    i = mz(q[0], "c0")
    if i:
        x(q[1])
    mz(q)

cudaq.sample(midCirc).dump()

# Can perform spin_op observation
@cudaq.kernel
def ansatz(angle):
    q = cudaq.qvector(2)
    x(q[0])
    ry(angle, q[1])
    x.ctrl(q[1], q[0])

hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

result = cudaq.observe(ansatz, hamiltonian, .59)
print(result.expectation_z())