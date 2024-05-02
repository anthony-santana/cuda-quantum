import cudaq
import numpy as np

cudaq.set_target('qpp-cpu')

@cudaq.kernel
def kernel():
  qvector = cudaq.qvector(np.array([1.,0.]))

  mz(qvector)


result = cudaq.sample(kernel)
print(result)