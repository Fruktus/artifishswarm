from unittest import TestCase
from unittest.mock import patch, Mock
import numpy as np

from artifishswarm import AFSA


class TestAFSA(TestCase):
    def setUp(self):
        self.afsa = AFSA(
            func=lambda x: -x ** 2,
            dimensions=2,
            population=1,
            max_iterations=1,
            vision=0.6,
            crowding_factor=0.98,
            step=0.6,
            search_retries=3
        )

    @patch('artifishswarm.afsa.random')
    def test_swimming(self, random):
        random.random = Mock(return_value=0.2)

        self.afsa.fish = [np.array(0.1)]
        self.afsa.swimming(0)
        self.assertEqual(0.1 + 0.6 * 0.2, self.afsa.fish[0])
