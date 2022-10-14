class AFSA:
    def __init__(self, func, population: int, vision: float, crowding_factor: float, step: float):
        """
        Initialization of artificial fish swarm algorithm.

        :param func: The optimisation function
        :param population: total number of artifish
        :param vision: the distance that the fish examines during optimisation
        :param crowding_factor: factor determining whether the area is overcrowded
            (compared with amount of fish/population)
        :param step: step length factor, used to calculate the distance of the fish movement
        """
        pass

    def searching(self, retries: int):
        """
        Searching Behavior - For fish in position Xi examine the surrounding area within vision.
        Assume that Yi, Yj are values of the optimised function for Xi, Xj.
        If the Yj > Yi, then move towards Xj.

        :param retries: Defines how many times the fish will attempt searching for food before moving to default
            'swimming' behavior
        :return: None
        """
        pass

    def swarming(self, asd):
        """
        Swarming Behavior - the fish examines its surroundings for the area with other fishes and more food
        (higher Y value).
        Assume that Xc is the center of nearby fishes (nf), whose distance dij < vision.
        If Yc > Yi and nf/n < delta_af (n - total population, delta_af - crowding factor<0, 1>), then move towards this
        direction.
        Otherwise, do the search behavior.

        :param asd:
        :return:
        """
        pass

    def following(self):
        """
        Following Behavior - the fish examines its surroundings for fishes (dij < vision).
        Then, it examines their Yj for best value.
        If the best Yj > Yi, then move towards that fish.
        Otherwise, do the search behavior.

        :return:
        """
        pass

    def leaping(self):
        """
        TODO
        Helps if the fish get stuck in local extremum.
        Select few fishes randomly and make them move towards other fishes.

        :return:
        """
        pass

    def swimming(self):
        """
        Swimming Behavior - swim in the randomly chosen direction.

        :return:
        """
        pass