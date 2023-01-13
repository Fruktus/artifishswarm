from unittest import TestCase

import numpy as np

from artifishswarm.functions import ackley, rastrigin, rosenbrock, beale, bukin6, \
    levi13, himmelblau, eggholder


class TestFunctions(TestCase):
    def test_ackley(self):
        self.assertAlmostEqual(0.0, ackley([0]))
        self.assertAlmostEqual(0.0, ackley([0, 0]))
        self.assertAlmostEqual(0.0, ackley([0, 0, 0]))

        self.assertNotAlmostEqual(0.0, ackley([1]))
        self.assertNotEqual(0.0, ackley([1]))

    def test_rastrigin(self):
        self.assertAlmostEqual(0.0, rastrigin([0]))
        self.assertAlmostEqual(0.0, rastrigin([0, 0]))
        self.assertAlmostEqual(0.0, rastrigin([0, 0, 0]))

        self.assertAlmostEqual(1.0, rastrigin([1]))
        self.assertAlmostEqual(4.0, rastrigin([2]))
        self.assertAlmostEqual(2.0, rastrigin([1, 1]))
        self.assertAlmostEqual(8.0, rastrigin([2, 2]))

        np.testing.assert_almost_equal(
            [0.0, 1.0],
            rastrigin(
                [np.array([0.0, 1.0])]
            ).tolist())

    def test_rosenbrock(self):
        self.assertAlmostEqual(0.0, rosenbrock([0]))
        self.assertAlmostEqual(1.0, rosenbrock([0, 0]))
        self.assertAlmostEqual(2.0, rosenbrock([0, 0, 0]))

        self.assertAlmostEqual(401.0, rosenbrock([2, 2]))

        np.testing.assert_almost_equal(
            [1.0, 401.0],
            rosenbrock([
                np.array([0, 2]),
                np.array([0, 2])
            ]).tolist())

    def test_beale(self):
        self.assertAlmostEqual(0.0, beale([3, 0.5]))

        np.testing.assert_almost_equal(
            [0.0, 0.0],
            beale([
                np.array([3, 3]),
                np.array([0.5, 0.5])
            ]).tolist())

    def test_bukin6(self):
        self.assertAlmostEqual(0.0, bukin6([-10, 1]))

        np.testing.assert_almost_equal(
            [0.0, 0.0],
            bukin6([
                np.array([-10, -10]),
                np.array([1, 1])
            ]).tolist())

    def test_levi13(self):
        self.assertAlmostEqual(0.0, levi13([1, 1]))

        np.testing.assert_almost_equal(
            [0.0, 0.0],
            levi13([
                np.array([1, 1]),
                np.array([1, 1])
            ]).tolist())

    def test_himmelblau(self):
        self.assertAlmostEqual(0.0, himmelblau([3, 2]))

        np.testing.assert_almost_equal(
            [0.0, 0.0],
            himmelblau([
                np.array([3, 3.584428]),
                np.array([2, -1.848126])
            ]).tolist())

    def test_eggholder(self):
        self.assertAlmostEqual(-959.6407, eggholder([512, 404.2319]), delta=0.00005)

        np.testing.assert_almost_equal(
            [-959.6407, -959.6407],
            eggholder([
                np.array([512, 512]),
                np.array([404.2319, 404.2319])
            ]).tolist(),
            decimal=4
        )
