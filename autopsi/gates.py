import numpy as __np__


class __GateTensor__:
    """Internal class that's subclassed to make each quantum gate.

    Args:
        backend (module): Numpy-like Python library to use when compiling a gate to a tensor. (default numpy)
        dtype (type): Data type to use for gates specified as matrices.
        angle (float): Angle parameter for gates that support rotation. (default None)
        lmda (float): Lambda (:math:`\lambda`) value to specify for a supported gate's :math:`\lambda` parameter. (default None)
        phi (float): Phi (:math:`\phi`) value to specify for a supported gate's :math:`\lambda` parameter. (default None)"""

    def __init__(self,
                 backend=__np__,
                 dtype=complex,
                 angle=None,
                 lmda=None,
                 phi=None
                 ):
        self.backend = backend
        self.dtype = dtype
        self.angle = angle
        self.lmda = lmda
        self.phi = phi

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

class u1(__GateTensor__):
    """U1 Gate

    :math:`u1(\lambda)=\begin{bmatrix}1 & 0\\0 & e^{i\lambda} \end{bmatrix}`"""

    def tensor(self):
        """Returns gate as a tensor.

       Returns:
                numpy.array: Gate represented as a tensor."""

        # Alias the backend's array(), exp() and multiply() functions
        array = self.backend.array
        exp = self.backend.exp
        multiply = self.backend.multiply

        # Grab the lambda value
        lmda = self.lmda


        # Get Euler's constant and convert it to the user-specified dtype
        eulers_constant = self.dtype(
            self.backend.e)

        # Get the imaginary number i and convert it to the user-specified dtype
        imag_number = self.dtype(
            0.+1.j
        )

        # Multiply i by the user-specified lambda, giving us the exponent we need to raise Euler's constant to
        ilmda = multiply(
            imag_number,
            lmda
        )

        # Raise Euler's constant to the power of i multiplied by the user-specified lambda value
        eilmda = eulers_constant ** ilmda


        # Return generated tensor
        return array([
            1., 0.,
            0., eilmda
        ])

    class u2(__GateTensor__):
        """U2 Gate.

        :math:`u2(\phi,\lambda)=\frac{1}{\sqrt{2}}\begin{bmatrix}1 & -e^{i\lambda}\\e^{i\phi} & e^{i(\phi+\lambda)}\end{bmatrix}`
        """

        def tensor(self):
            """Returns gate as a tensor.

            Returns:
                numpy.array: Gate represented as a tensor."""

            # Alias the backend's array(), exp(), multiply() and negative() and add() functions
            array = self.backend.array
            exp = self.backend.exp
            multiply = self.backend.multiply
            neg = self.backend.negative
            add = self.backend.add

            # Get user-specified parameters
            lmda = self.lmda
            phi = self.phi

            # Alias Euler's constant according to the backend
            e = self.backend.e

            # Alias the imaginary number i
            i = 0+1j

            # Create a list to store things we are going to put in our array
            array_elements = []

            # Add the first element, just the number 1
            array_elements.append(1)

            # Add the second element by calculating its value
            array_elements.append(
                neg(
                    exp(
                        e,
                        multiply(
                            i,
                            lmda))))

            # Add the third element by calculating its value
            array_elements.append(
                exp(
                    e,
                    multiply(
                        i,
                        phi)))

            # Add the fourth and final element by calculating its value
            array_elements.append(
                exp(
                    e,
                    multiply(
                        i,
                        add(
                            phi,
                            lmda))))

            # Return the generated tensor
            return array(
                array_elements, # Use Generated Array Elements
                dtype=self.dtype) # Ensure dtype of the array matches the user-specified Data Type
