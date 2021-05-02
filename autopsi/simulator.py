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
            list(numpy.ndarray): List of tensors to multiply by the simulator's state. (default None)"""
        if tensors is not None:
            for i in tensors:
                self.step(tensor=i)
