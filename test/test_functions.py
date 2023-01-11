from unittest import TestCase

import numpy as np

from artifishswarm.functions import rastrigin, rosenbrock


class TestFunctions(TestCase):
    def test_rastrigin(self):
        self.assertEqual(0.0, rastrigin([0]))
        self.assertEqual(0.0, rastrigin([0, 0]))
        self.assertEqual(0.0, rastrigin([0, 0, 0]))

        self.assertEqual(1.0, rastrigin([1]))
        self.assertEqual(4.0, rastrigin([2]))
        self.assertEqual(2.0, rastrigin([1, 1]))
        self.assertEqual(8.0, rastrigin([2, 2]))

        self.assertEqual([0.0, 1.0], rastrigin([np.array([0.0, 1.0])]).tolist())

    def test_rosenbrock(self):
        self.assertEqual(0.0, rosenbrock([0]))
        self.assertEqual(1.0, rosenbrock([0, 0]))
        self.assertEqual(2.0, rosenbrock([0, 0, 0]))
