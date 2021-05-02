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