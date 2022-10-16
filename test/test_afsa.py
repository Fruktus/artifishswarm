from unittest import TestCase
from unittest.mock import Mock
import numpy as np

from artifishswarm import AFSA


class TestAFSA(TestCase):
    def setUp(self):
        self.afsa = AFSA(
            func=lambda x: 2*x,
            dimensions=2,
            population=1,
            max_iterations=1,
            vision=0.6,
            crowding_factor=0.98,
            step=0.6,
            search_retries=3
        )

    def test_swimming(self, ):
        self.afsa.rng = Mock()
        self.afsa.rng.uniform = Mock(return_value=0.2)

        self.afsa.fish = [np.array(0.1)]
        self.afsa.swim(0)
        self.assertEqual(0.1 + 0.6 * 0.2, self.afsa.fish[0])

    def test_find_nearby_fish_in_vision(self):
        self.afsa.fish = np.array([
            np.array(0.2),
            np.array(0.1),
            np.array(0.0),
            np.array(0.1),
            np.array(0.2),
            np.array(0.3)])
        self.afsa.vision = 0.3

        fishes_in_range = self.afsa.find_nearby_fish_in_vision(2)
        np.testing.assert_array_equal(np.array([0, 1, 3, 4]), fishes_in_range)

    def test_find_nearby_fish_in_vision_multidim(self):
        self.afsa.fish = np.array([
            np.array([0.2, 0.2]),
            np.array([0.1, 0.1]),
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
            np.array([0.3, 0.3])
        ])
        self.afsa.vision = 0.3
        self.afsa.dimensions = 3
        
        fishes_in_range = self.afsa.find_nearby_fish_in_vision(2)
        np.testing.assert_array_equal(np.array([0, 1, 3, 4]), fishes_in_range)

    def test_search(self):
        self.afsa.rng = Mock()
        self.afsa.rng.uniform = Mock(return_value=0.2)
        self.afsa.make_step = Mock()
        self.afsa.swim = Mock()
        self.afsa.fish = np.array([0.0])
        self.afsa.vision = 0.3
        self.afsa.func = lambda x: 2*x

        self.afsa.search(0)

        self.afsa.make_step.assert_called_with(0, 0.06)
        self.afsa.swim.assert_not_called()

    def test_swarm_better(self):
        self.afsa.make_step = Mock()
        self.afsa.search = Mock()
        self.afsa.fish = np.array([
            np.array([0.0]),
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.4])
        ])
        self.afsa.vision = 0.4
        self.afsa.func = lambda x: 2 * x
        self.afsa.population = len(self.afsa.fish)

        self.afsa.swarm(0)

        self.afsa.make_step.assert_called_once()
        self.assertEqual(0, self.afsa.make_step.call_args[0][0])
        np.testing.assert_almost_equal(self.afsa.make_step.call_args[0][1], 0.15)
        self.afsa.search.assert_not_called()

    def test_swarm_worse(self):
        self.afsa.make_step = Mock()
        self.afsa.search = Mock()
        self.afsa.fish = np.array([
            np.array([0.3]),
            np.array([0.1]),
            np.array([0.2]),
            np.array([0.4])
        ])
        self.afsa.vision = 0.4
        self.afsa.func = lambda x: 2 * x
        self.afsa.population = len(self.afsa.fish)

        self.afsa.swarm(0)

        self.afsa.make_step.assert_not_called()
        self.afsa.search.assert_called_with(0)

    def test_follow(self):
        self.afsa.make_step = Mock()
        self.afsa.search = Mock()
        self.afsa.fish = np.array([
            np.array([0.1]),
            np.array([0.1]),
            np.array([0.1]),
            np.array([0.3])
        ])
        self.afsa.vision = 0.4
        self.afsa.func = lambda x: 2 * x
        self.afsa.population = len(self.afsa.fish)

        self.afsa.follow(0)

        self.afsa.make_step.assert_called_once_with(0, 0.3)
        self.afsa.search.assert_not_called()
