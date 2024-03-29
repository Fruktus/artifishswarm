import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import spatial
from tqdm import tqdm


class AFSA:
    """
    Conventions:
    x - coordinates, can be any dimension
    y - the value that optimised function self.func takes for given x
    """

    def __init__(self, *, func, dimensions: int, population: int,
                 max_iterations: int, vision: float,
                 crowding_factor: float, step: float,
                 search_retries: int, optimize_towards = 'max',
                 high: int = 1,
                 low: int = 0, save_history: bool = False, rand_seed=None,
                 leap_eps: float = 0, leap_scale: float = 10, verbose=False):
        """
        Initialization of artificial fish swarm algorithm.

        :param func: the function to be optimized
        :param dimensions: the dimensionality of the domain, i.e. dim(x)
        :param population: total number of artifish
        :param max_iterations: how many iterations can be made at most while searching for optimum
        :param vision: the distance that the fish examines during optimisation
        :param crowding_factor: factor determining whether the area is overcrowded
            (compared with amount of fish/population)
        :param step: step length factor, used to calculate the distance of the fish movement
        :param search_retries: how many times will the fish search for the food during search behavior
        :param optimize_towards: whether to look for maximum or minimum
        :param high: the upper limit for fish location initialization
        :param low: the lower limit for fish location initialization
        :param save_history: if specified, every iteration's state will be stored and available later through
            get_history()
        :param rand_seed: seed for the random number generator
        "param verbose: if enabled, will provide additional time elapsed output
        """
        if not callable(func):
            raise TypeError('func is not callable')

        self.rng = np.random.RandomState(rand_seed)

        def func_safe(x):
            if not hasattr(x, 'shape'):
                raise TypeError(f'x is not an array {x}')
            if len(x.shape) != 1:
                raise ValueError(f'x has more than 1 axis {x}')
            if x.shape != (self.dimensions,):
                raise ValueError(f'x has wrong number of dimensions {x}')

            y = func(x)

            if not isinstance(y, float):
                raise TypeError(f'func returned a non-float value {y} for {x}')
            return y

        self.func = func_safe
        self.dimensions = dimensions
        self.population = population
        self.max_iterations = max_iterations
        self.vision = vision
        self.crowding_factor = crowding_factor
        self.step = step
        self.search_retries = search_retries
        self.history = pd.DataFrame()
        self.save_history = save_history
        self.leap_eps = leap_eps
        self.leap_scale = leap_scale
        self.comparator = lambda a, b: a > b if optimize_towards == 'max' else a < b
        self.best_x = None
        self.best_y = float('-inf') if self.comparator(float('inf'), float('-inf')) else float('inf')
        self.verbose = verbose

        self.fish = self.rng.uniform(
            low=low, high=high, size=population * dimensions).reshape((population, dimensions))

    def prey(self, fish_idx: int):
        """
        Searching Behavior - For fish in position Xi examine the surrounding area within vision.
        Assume that Yi, Yj are values of the optimised function for Xi, Xj.
        If the Yj > Yi, then move towards Xj.

        :param fish_idx: the index of fish which performs the search behavior
        :return: None
        """
        for _ in range(self.search_retries):
            target_x = self.fish[fish_idx] + \
                self.vision * self.rng.uniform(-1, 1)
            # originally the random is between [0,1) but it may be worth exploring
            # how it behaves when allowed the range [-1,1]

            target_y = self.func(target_x)
            if self.comparator(target_y, self.func(self.fish[fish_idx])):
                return target_x, target_y

        return None, None

    def swarm(self, fish_idx: int):
        """
        Swarming Behavior - the fish examines its surroundings for the area with other fishes and more food
        (higher Y value).
        Assume that Xc is the center of nearby fishes (nf), whose distance dij < vision.
        If Yc > Yi and nf/n < delta_af (n - total population, delta_af - crowding factor<0, 1>), then move towards this
        direction.
        Otherwise, do the search behavior.

        :param fish_idx: the index of fish which performs the swarm behavior
        :return:
        """

        fish_in_vision_idx = self.find_nearby_fish_in_vision(fish_idx)
        fish_in_vision_x = self.fish[fish_in_vision_idx]

        if len(fish_in_vision_x) > 0:
            x_center = np.mean(fish_in_vision_x, axis=0)
            y_center = self.func(x_center)

            is_center_better = self.comparator(y_center, self.func(self.fish[fish_idx]))
            is_overcrowded = len(fish_in_vision_idx) / \
                self.population > self.crowding_factor

            if is_center_better and not is_overcrowded:
                return x_center, y_center

        return None, None

    def follow(self, fish_idx: int):
        """
        Following Behavior - the fish examines its surroundings for fishes (dij < vision).
        Then, it examines their Yj for best value.
        If the best Yj > Yi, then move towards that fish.
        Otherwise, do the search behavior.

        :param fish_idx: the index of fish which performs the follow behavior
        :return:
        """
        own_y = self.func(self.fish[fish_idx])

        fish_in_vision_idx = self.find_nearby_fish_in_vision(fish_idx)
        fish_in_vision_x = self.fish[fish_in_vision_idx]
        fish_in_vision_y = [self.func(x) for x in fish_in_vision_x]
        is_overcrowded = len(fish_in_vision_idx) / \
            self.population > self.crowding_factor

        best_y = own_y
        best_idx = fish_idx
        for food_idx in range(len(fish_in_vision_y)):
            if self.comparator(fish_in_vision_y[food_idx], best_y):
                best_y = fish_in_vision_y[food_idx]
                best_idx = food_idx

        if best_idx != fish_idx and not is_overcrowded:
            return self.fish[best_idx], best_y
        else:
            return None, None

    def leap(self):
        """
        Helps if the fish get stuck in local extremum.
        Select a fish randomly and make a leap in a random direction.
        """
        selected_fish_idx = self.rng.randint(0, self.population)
        leap_distances = self.rng.normal(0, self.leap_scale, self.dimensions)
        self.fish[selected_fish_idx] += leap_distances

    def move(self, fish_idx: int):
        """
        Swimming Behavior - swim in the randomly chosen direction.

        :param fish_idx: the index of fish which performs swim behavior
        :return:
        """
        self.fish[fish_idx] += self.vision * \
            self.rng.uniform(-1, 1, self.dimensions)

    def find_nearby_fish_in_vision(self, fish_idx) -> ArrayLike:
        """
        Returns an array of fishies that the fish described by fish_idx can see

        :param fish_idx: the fish in context of which to perform the search
        :return: numpy array of nearby fish indexes, which can contain zero or more entries
        """
        fish_distances = spatial.distance.cdist(
            np.array([self.fish[fish_idx]]),
            self.fish)
        fish_distances = fish_distances.reshape(-1, 1)
        fish_distances[fish_idx] = self.vision

        return np.where(fish_distances < self.vision)[0]

    def make_step(self, fish_idx: int, visual_x: ArrayLike):
        """
        Moves the fish towards the distance (in place), takes the modifiers into account.

        :param fish_idx: index of the fish to move
        :param visual_x: the visual x position, the fish moves towards this point
        :return:
        """
        dest_y = self.func(visual_x)
        if self.comparator(dest_y, self.best_y):
            self.best_x = visual_x
            self.best_y = dest_y

        current_x = self.fish[fish_idx]

        diff_x = visual_x - current_x
        direction = diff_x / np.linalg.norm(diff_x)
        destination_x = current_x + direction * \
            self.step * self.rng.uniform(0, 1)

        self.fish[fish_idx] = destination_x
        return dest_y

    def iteration(self):
        """
        Runs a single iteration of the simulation

        :return:
        """
        iter_best_y = float('-inf') if self.comparator(float('inf'), float('-inf')) else float('inf')
        for fish_idx in range(self.population):
            x_s, y_s = self.swarm(fish_idx)
            x_f, y_f = self.follow(fish_idx)
            x_p, y_p = self.prey(fish_idx)

            if y_s and self.comparator(y_s, self.func(self.fish[fish_idx])):
                y = self.make_step(fish_idx=fish_idx, visual_x=x_s)
                iter_best_y = iter_best_y if self.comparator(iter_best_y, y) else y
                continue

            if y_f and self.comparator(y_f, self.func(self.fish[fish_idx])):
                y = self.make_step(fish_idx=fish_idx, visual_x=x_f)
                iter_best_y = iter_best_y if self.comparator(iter_best_y, y) else y
                continue

            if y_p and self.comparator(y_p, self.func(self.fish[fish_idx])):
                y = self.make_step(fish_idx=fish_idx, visual_x=x_p)
                iter_best_y = iter_best_y if self.comparator(iter_best_y, y) else y
                continue

            self.move(fish_idx)
        return iter_best_y

    def get_history(self) -> pd.DataFrame:
        """Returns saved fish state history over iterations if save_history was set to true"""
        return self.history

    @property
    def result(self):
        return self.best_x, self.best_y

    def run(self) -> None:
        """
        Executes the algorithm

        :return: returns the best (highest) value found by the fish swarm
        """

        self.history = pd.DataFrame()
        self.best_x = None
        self.best_y = float('-inf') if self.comparator(float('inf'), float('-inf')) else float('inf')

        last_iter_best_y = float('-inf') if self.comparator(float('inf'), float('-inf')) else float('inf')
        for epoch in tqdm(range(self.max_iterations), disable=not self.verbose):
            if self.save_history:
                df = pd.DataFrame(self.fish)
                df['epoch'] = epoch
                self.history = pd.concat([self.history, df], ignore_index=True)

            iter_best_y = self.iteration()

            if abs(last_iter_best_y - iter_best_y) < self.leap_eps:
                self.leap()

            last_iter_best_y = iter_best_y
