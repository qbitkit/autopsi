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
        return self.state ** 2

    def history(self):
        """If the simulator has tracing enabled, this function will return the history of the simulator's state.

        Returns:
            list(numpy.ndarray): Timeline of the simulator's state prior to applying each gate."""

        if self.history:
            return self.history

    def batch(self,
              tensors=None):
        """Alter the simulator's state by taking the tensor product of the simulator's state and the specified batch of tensors.

        Args:
            tensors(numpy.ndarray): List of tensors to multiply by the simulator's state. (default None)"""

        if tensors is not None:
            for i in tensors:
                self.step(tensor=i)

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


    def count_qubits(self):
        """Counts the number of qubits being simulated.

        Returns:
            int: Number of qubits being simulated."""

        # Initialize variable 'num_probabilities' to track the number of probabilities.
        num_probabilities = 0
        # Iterate over the array of probabilities, adding 1 to 'num_probabilities' at each iteration.
        for item in self.probabilities():
            num_probabilities += 1
        # Return the number of qubits by dividing the number of probabilities by 2.
        return self.backend.divide(num_probabilities, 2)

    def measure(self,
                return_as_type=int):
        """Takes one measurement of the state of the simulator to a classical binary value.

        Args:
            return_as_type(type): Return value measured from qubit(s) as a specified type. (default int)
        Returns:
            return_as_type: Value measured from qubit(s).
        """

        # Initialize variable 'num_probabilities' to track the number of probabilities.
        num_probabilities = 0
        # Iterate over the array of probabilities, adding 1 to 'num_probabilities' at each iteration.
        for item in self.probabilities():
            num_probabilities += 1

        # Calculate the number of qubits by dividing the number of probabilities by 2.
        num_qubits = self.backend.divide(num_probabilities, 2)

        probabilities_as_float = self.astype(float,
                                             'probabilities')

        weighted_pseudorandom_choice = self.backend.random.choice(
            list(
                range(
                    len(
                        probabilities_as_float))),
            p=probabilities_as_float)

        return self.backend.binary_repr(
            weighted_pseudorandom_choice)
