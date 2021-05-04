import numpy as __np__


class __GateTensor__:
    """Internal class that's subclassed to make each quantum gate.

    Args:
        backend (module): Numpy-like Python library to use when compiling a gate to a tensor. (default numpy)
        dtype (type): Data type to use for gates specified as matrices.
        angle (float): Angle parameter for gates that support rotation. (default None)"""

    def __init__(self,
                 backend=__np__,
                 dtype=complex,
                 angle=None
                 ):
        self.backend = backend
        self.dtype = dtype
        self.angle = angle

class h(__GateTensor__):
    """Hadamard gate"""
    def tensor(self):
        """Returns gate as a tensor."""

        div = self.backend.divide
        sqrt = self.backend.sqrt
        arry = self.backend.array

        return div(arry([1, 1, 1, -1],
                        dtype=self.dtype),
                   sqrt(2)).reshape(2, 2)

class ry(__GateTensor__):
    """RY gate. Rotates qubit spin over the Y axis."""
    def tensor(self):
        """Returns gate as a tensor."""

        div = self.backend.divide
        sin = self.backend.sin
        cos = self.backend.cos
        neg = self.backend.negative
        arry = self.backend.array

        angle = self.angle

        return arry([
            cos(
                div(
                    angle,
                    2
                )
            ),
            neg(
                sin(
                    div(
                        angle,
                        2
                    )
                )
            ),
            sin(
                div(
                    angle,
                    2
                )
            ),
            cos(
                div(
                    angle,
                    2
                )
            )])