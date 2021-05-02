import numpy as __np__


class __GateTensor__:
    """Internal class that's subclassed to make each quantum gate.

    Args:
        backend (module): Numpy-like Python library to use when compiling a gate to a tensor. (default numpy)
        dtype (type): Data type to use for gates specified as matrices."""

    def __init__(self,
                 backend=__np__,
                 dtype=complex
                 ):
        self.backend = backend
        self.dtype = dtype

class h(__GateTensor__):
    """Hadamard gate"""
    def tensor(self):
        """Returns gate as a tensor."""

        div = self.backend.div
        sqrt = self.backend.sqrt
        arry = self.backend.array

        return div(arry([1, 1, 1, -1],
                        dtype=self.dtype),
                   sqrt(2)).reshape(2, 2)

