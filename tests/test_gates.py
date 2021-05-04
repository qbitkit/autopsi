from unittest import TestCase as __tc__
from autopsi import gates as __gates__
import numpy as __np__


class TestInternalFunctions(__tc__):
    def test___GateTensor__(self):
        testing_message = 'test'
        testing_gatetensor = __gates__.__GateTensor__(backend=testing_message)
        self.assertEqual(testing_message,
                         testing_gatetensor.backend)

class TestSingleQubitGates(__tc__):
    def test_h(self):
        h_list = [[(0.7071067811865475+0j), (0.7071067811865475+0j)],
                  [(0.7071067811865475+0j), (-0.7071067811865475+0j)]]
        self.assertEqual(
            h_list,
            __gates__.h().tensor().tolist()
        )
    def test_ry(self):
        ry1_list = [0.8775825618903728, -0.479425538604203,
                    0.479425538604203, 0.8775825618903728]
        self.assertEqual(
            ry1_list,
            __gates__.ry(
                angle=1
            ).tensor().tolist()
        )