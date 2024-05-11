import cudaq
from cudaq import spin

import time
import random
import numpy as np
import scipy as sp


################################################################################
############################## New cudaq data-types ############################
class Variable:
    """
    A type that will be used to serve as dummy variables in parameterized
    Spin Operator expressions.
    """

    def __init__(self):
        pass

    # The Variable when called without a provided value will just return 1.0 .
    # When a value is given, this will become the actual value of the `Variable`.
    def __call__(self, value=1.0):
        return value


cudaq.Variable = Variable


class Time:
    """

    Alternative name ideas: `Clock`, `GlobalClock`

    A type that will be used to serve as variable for the time in
    parameterized Spin Operator expressions.

    `start_time` : 0.0 default start time for time evolution.
    `stop_time`  : the stop time for time evolution. 
                   This value will in the units of time used in the programmers
                   Hamiltonian.
    `resolution` : Otherwise known as the `dt`. This value determines how finely
                   discretized we would like the time evolution to be.
    """

    def __init__(self):
        """ Default values given are slightly magical. """
        self.start_time = 0.0
        self.max_time = 1.0
        self.resolution = 0.25
        self.sample_times = None

    def time_series(self) -> np.ndarray:
        """ 
        Return the discrete time step values, t, over a given
        duration: [start_time, max_time].
        """
        self.sample_times = np.arange(start=self.start_time,
                                      stop=self.max_time,
                                      step=self.resolution)
        return self.sample_times

    def to_index(self, time_value: float) -> int:
        """ 
        A horribly inneficient function implementation. 
        Finds the index in the `sample_times` of the value equal
        to the user-provided `time_value`.
        Raises an index error if that time value wasn't found.
        """
        for index, time in np.ndenumerate(self.sample_times):
            if (time_value == time):
                return index
        raise IndexError

    # The Variable when called without a provided value will just return 1.0 .
    # When a value is given, this will become the actual value of the `Time`.
    # This is used to make it inherently compatible with the arithmetic operators
    # of the `cudaq.SpinOperator` class.
    def __call__(self, value=1.0) -> float:
        return value


cudaq.Time = Time


class HardwareControl:
    pass


cudaq.HardwareControl = HardwareControl


class ControlSignal(cudaq.HardwareControl):
    """
    The `cudaq.ControlSignal` type would be a new function interface that you could
    imagine being extended to existing as a function in the MLIR. 
    
    This would be a black-box function that takes a
    time value, `t`, as input, and returns the amplitude of the control signal at that
    time slice.

    This is potentially a narrow minded first pass of developing a `ControlSignal` type, 
    and it may be better generalized with external input. For example, maybe this makes
    sense for representing a microwave waveform, but not for tuning parameters of a laser.
    """

    def __init__(self, time: cudaq.Time):
        # A user can either provide all of the samples upfront, or provide
        # a blackbox function that returns the sample value at a time-step.
        self.time: cudaq.Time = time

        # We will have one or the other:
        self.full_sample_mode = False
        self.sample_function_mode = False
        self.sample_values: np.ndarray = None
        self.sample_function: function = None

    def set_sample_values(self, sample_values):
        self.sample_values = sample_values
        self.full_sample_mode = True

    def set_sample_function(self, user_function):
        """
        All this requires (not enforced) is that the `user_function` has a `__call__`
        overload that accepts just a time value and returns a float value.
        """
        self.sample_function = user_function
        self.sample_function_mode = True

    def __call__(self, time_value=None) -> float:
        """
        I know the nested returns in this are atrocious (sorry Eric).
        """
        if not (self.full_sample_mode or self.sample_function_mode):
            print(
                "An array of sample values or sample function must be provided."
            )
            raise ValueError

        if time_value is not None:
            if self.full_sample_mode:
                # have to convert the time value to an index in this case
                return self.sample_values[self.time.to_index(time_value)]
            else:
                return self.sample_function(time_value)
        # Need this for it to be compatible with spin operator arithmetic,
        # if we don't yet have sample values.
        return 0.0

    def sample_times(self):
        return self.time.time_series()


cudaq.ControlSignal = ControlSignal


class ControlPhase(HardwareControl):
    """ """

    def __init__(self, time: cudaq.Time):
        self.time: cudaq.Time = time

    def set_phase_shift_times(self):
        pass

    def set_phase_shift_values(self):
        pass


################################################################################


def synthesize_unitary(hamiltonian, time_variable):
    """
    This would be a great function to offload to a GPU, as it
    involves repeated exponentiation of a matrix.

    You could also parallelize these calculations very nicely
    as each time slice of the unitary is indepdent of the others.
    """
    initial_name = "custom_operation_"
    index = 0
    # The reverse is necessary for time ordering.
    for time_value in reversed(time_variable.time_series()):
        operation_name = initial_name + str(index)
        unitary_matrix = sp.linalg.expm(-1j * time_variable.resolution *
                                        hamiltonian(time_value))
        cudaq.register_operation(unitary_matrix, operation_name=operation_name)
        index += 1
    return cudaq.globalRegisteredUnitaries.copy()


cudaq.synthesize_unitary = synthesize_unitary

################################################################################

# Used for testing the alternative feature of passing a function for the envelope


def custom_envelope_function(time: float):
    """
    A custom user-provided function.

    We will just return a random value of 0 or 1 for now.

    This will get passed off to the ControlSignal function so it knows where
    to grab the control signal amplitude from.
    """
    return random.randint(0, 1)
