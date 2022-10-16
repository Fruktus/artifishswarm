import numpy as np
from numpy.typing import ArrayLike
from scipy import spatial


class AFSA:
    """
    Conventions:
    x - coordinates, can be any dimension
    y - the value that optimised function self.func takes for given x
    """

    def __init__(self, func, dimensions: int, population: int, max_iterations: int, vision: float, crowding_factor: float, step: float,
                 search_retries: int, rand_seed=None):
        """
        Initialization of artificial fish swarm algorithm.

        :param func: The optimisation function
        :param dimensions: the dimensionality of the function, dim(x) = dimensions-1, dim(y) = 1
        :param population: total number of artifish
        :param max_iterations: how many iterations can be made at most while searching for optimum
        :param vision: the distance that the fish examines during optimisation
        :param crowding_factor: factor determining whether the area is overcrowded
            (compared with amount of fish/population)
        :param step: step length factor, used to calculate the distance of the fish movement
        :param search_retries: how many times will the fish search for the food during search behavior
        :param rand_seed: seed for the random number generator
        """
        self.rng = np.random.RandomState(rand_seed)

        self.func = func
        self.dimensions = dimensions
        self.population = population
        self.max_iterations = max_iterations
        self.vision = vision
        self.crowding_factor = crowding_factor
        self.step = step
        self.search_retries = search_retries

        self.fish = self.rng.rand(self.population)
        self.food = np.array([self.func(fish) for fish in self.fish])

    def search(self, fish_idx: int):
        """
        Searching Behavior - For fish in position Xi examine the surrounding area within vision.
        Assume that Yi, Yj are values of the optimised function for Xi, Xj.
        If the Yj > Yi, then move towards Xj.

        :param fish_idx: the index of fish which performs the search behavior
        :return: None
        """
        for _ in range(self.search_retries):
            target_x = self.fish[fish_idx] + self.vision * self.rng.uniform()
            # originally the random is between <0,1) but it may be worth exploring
            # how it behaves when allowed the range <-1,1>

            if self.func(target_x) > self.func(fish_idx):
                self.make_step(fish_idx, target_x)
                return

        self.swim(fish_idx)

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
        fish_in_vision_x = np.take(self.fish, fish_in_vision_idx)
        x_center = np.mean(fish_in_vision_x)
        y_center = self.func(x_center)

        is_center_better = y_center > self.func(self.fish[fish_idx])
        is_overcrowded = len(fish_in_vision_idx)/self.population > self.crowding_factor

        if is_center_better and not is_overcrowded:
            self.make_step(fish_idx, x_center)
        else:
            self.search(fish_idx)

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
        fish_in_vision_x = np.take(self.fish, fish_in_vision_idx)
        fish_in_vision_y = [self.func(x) for x in fish_in_vision_x]
        is_overcrowded = len(fish_in_vision_idx)/self.population > self.crowding_factor

        best = own_y
        best_idx = fish_idx
        for food_idx in range(len(fish_in_vision_y)):
            if fish_in_vision_y[food_idx] > best:
                best = fish_in_vision_y[food_idx]
                best_idx = food_idx

        if best_idx != fish_idx and not is_overcrowded:
            self.make_step(fish_idx, fish_in_vision_x[best_idx])
        else:
            self.search(fish_idx)

    def leap(self, fish_idx: int):
        """
        TODO
        Helps if the fish get stuck in local extremum.
        Select few fishes randomly and make them move towards other fishes.

        :param fish_idx: the index of fish which performs the leap behavior
        :return:
        """
        pass

    def swim(self, fish_idx: int):
        """
        Swimming Behavior - swim in the randomly chosen direction.

        :param fish_idx: the index of fish which performs swim behavior
        :return:
        """

        self.fish[fish_idx] += self.vision * self.rng.uniform()

    def find_nearby_fish_in_vision(self, fish_idx) -> ArrayLike:
        """
        Returns an array of fishies that the fish described by fish_idx can see

        :param fish_idx: the fish in context of which to perform the search
        :return: numpy array of nearby fish indexes, which can contain zero or more entries
        """
        fish_distances = spatial.distance.cdist(self.fish[fish_idx].reshape((-1, self.dimensions-1)),
                                                self.fish.reshape(-1, self.dimensions-1))
        fish_distances = fish_distances.reshape(-1, 1)
        fish_distances[fish_idx] = self.vision

        return np.where(fish_distances < self.vision)[0]

    def make_step(self, fish_idx: int, destination_x: ArrayLike):
        """
        Moves the fish towards the distance (in place), takes the modifiers into account.

        :param fish_idx: index of the fish to move
        :param destination_x: the destination x position
        :return:
        """
        current_x = self.fish[fish_idx]

        new_x = self.fish[fish_idx] + ((destination_x - current_x)/np.linalg.norm(destination_x - current_x)) \
            * self.step * self.rng.uniform()

        self.fish[fish_idx] = new_x

    def iteration(self):
        """
        Runs a single iteration of the simulation

        :return:
        """
        pass

    def run(self):
        """
        Executes the algorithm

        :return: TODO describe properly - returns solution
        """
        for epoch in range(self.max_iterations):
            for fish_idx in range(self.population):
                self.swarm(fish_idx)
                self.follow(fish_idx)

        return self.fish  # TODO verify whether this is the proper way to return solution
