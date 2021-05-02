from unittest import TestCase as __tc__
from autopsi import gates as __gates__


class TestInternalFunctions(__tc__):
    def test___GateTensor__(self):
        testing_message = 'test'
        testing_gatetensor = __gates__.__GateTensor__(backend=testing_message)
        self.assertEqual(testing_message,
                         testing_gatetensor.backend)