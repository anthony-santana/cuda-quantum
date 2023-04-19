import module_
# import numpy as np

@module_.ModuleRunner
def foo(length: int, width: float):
    """A function that gets called from within a class of our module."""
    x = np.zeros(length)
    print(x)
    print("leaving the function\n")
    return x

length = 5
width = 1.
foo(length, width)

# module_.ModuleRunner(foo)()

