import numpy as __np__
import tensorflow as __tf__


class Tensor:
    """Tensor product state simulator with support for recording the history of the simulator's state.

    Args:
        backend (numpy): Import any numpy-like library and specify that library with this parameter to achieve a higher degree of interoperability. CuPy can be imported and specified here to speed up simulations on systems with Nvidia graphics cards installed. (default: numpy)
        dtype (type): Data type to use for tensors, must be compatible with specified backend. (default complex)
        device (str): String specifying a device to use when multiplying matrices. (default None)
        entrypoint (list): If specified, simulator will start out with the specified value as its state. (default None)
        trace (bool): If set to True, the simulator's state will be saved in memory prior to each quantum gate's execution. (default True)"""

    def __init__(self,
                 backend=__np__,
                 dtype=complex,
                 device=None,
                 entrypoint=None,
                 trace=True):
        self.backend = backend
        self.dtype = dtype
        self.device = device
        self.trace = trace

        self.history = []

        tensor = self.backend.array

        self.state = tensor(entrypoint if entrypoint is not None else [1, 0],
                            dtype=self.dtype)

    def astype(self,
               dtype=float,
               value='amplitudes'):
        """Return amplitudes or probabilities forcefully cast to the specified Data Type.

        .. warning::
            Using this function to forcefully cast a complex dtype to a float dtype discards its imaginary part.
            Numpy will throw a ComplexWarning if you convert from complex to float.

        Args:
            dtype(type): Data Type to convert (default float)
            value(str): Returns probabilities if set to 'probabilities', and amplitudes if set to 'amplitudes'. (default 'amplitudes')
        Returns:
            numpy.array: Array forcefully cast to specified dtype.
        """

        # Determine which array to copy based on the 'value' keyword parameter.
        array_to_copy = self.amplitudes() if value == 'amplitudes' else self.probabilities()

        # Forcefully convert array's Data Type.
        return self.backend.array(
               [item.astype(dtype) # Convert each variable to the specified Data Type.
                for item in array_to_copy], # Iterate over the array we want to copy.
               dtype=dtype) # Ensure the array's Data Type gets specified with what we just cast it to.

    def amplitudes(self):
        """Returns the current state of the simulator as amplitudes.

        Returns:
            numpy.ndarray: Current state of the simulator, as amplitudes."""
        return self.state

    def probabilities(self):
        """Return the state of the simulator as probabilities.
        This function works by squaring the state of the simulator stored as amplitudes.

        Returns:
            numpy.ndarray: Current state of the simulator, as probabilities"""
        abs = self.backend.abs
        return abs(self.state) ** 2

    def history(self):
        """If the simulator has tracing enabled, this function will return the history of the simulator's state.

        Returns:
            list(numpy.ndarray): Timeline of the simulator's state prior to applying each gate."""

        if self.history:
            return self.history

    def measure(self,
                return_as_type=int):
        """Takes one measurement of the state of the simulator to a classical binary value.

        Args:
            return_as_type(type): Return value measured from qubit(s) as a specified type. (default int)
        Returns:
            return_as_type: Value measured from qubit(s).
        """

        probabilities_as_float = self.astype(float,
                                             'probabilities')

        weighted_pseudorandom_choice = self.backend.random.choice(
            list(
                range(
                    len(
                        probabilities_as_float))),
            p=probabilities_as_float)

        return return_as_type(
            self.backend.binary_repr(
                weighted_pseudorandom_choice))

    def batch_measure(self,
                      shots=10,
                      dtype=int):
        """Take a specified number of measurements, appending each measurement to a list and returning the resulting list.

        Args:
            shots(int): A positive integer describing the total number of measurements to take. (default 10)
            dtype(type): Data Type to use when storing a measurement. (default int)
        Returns:
            list: A list containing measurements cast as the specified Data Type."""

        return [self.measure(dtype) # Take a measurement each iteration and append it to the list we return
                for shot in range(shots)] # Iterate over the number of shots we need to take represented as a range object

    def step(self,
             tensor=None):
        """Alter the simulator's state by applying a quantum logic gate specified as a tensor.

        Args:
            tensor (list): If specified, the value of the simulator will be multiplied by the specified tensor."""
        product = self.backend.matmul

        if tensor is not None:

            if self.trace is True:
                self.history.append(self.state)

            if self.device is None:
                self.state = product(self.state,
                                     tensor)

            if self.device is not None:
                with __tf__.device(self.device):
                    self.state = product(self.state,
                                         tensor)

    def batch_step(self,
                   tensors=None):
        """Alter the simulator's state by taking the tensor product of the simulator's state and the specified batch of tensors.

        Args:
            tensors(numpy.ndarray): List of tensors to multiply by the simulator's state. (default None)"""

        if tensors is not None:
            for i in tensors:
                self.step(tensor=i)
