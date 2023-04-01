from dataclasses import dataclass
from problem import Problem, Timer
from random import random, randint
from math import exp
from typing import Union


@dataclass
class SimulatedAnnealingConfig:
    initial_temperature: int = 5
    cooling_step: float = 0.999
    min_temperature: float = 1e-10
    escape_random_restart_probability: float = 0.33
    escape_perturbation_probability: float = 0.33
    escape_perturbation_size: int = 50
    escape_reheat_probability: float = 0.33
    escape_reheat_ratio: float = 0.1
    local_optimum_moves_threshold: int = 10
    local_optimum_escapes_max: int = -1  # -1 means "infinity"


class SimulatedAnnealing:
    """
    Implementation of the simulated annealing algorithm.
    """

    best_state: Union[list[int], None] = None
    steps_from_last_state_update: int = 0
    timer = Timer(60)

    def __init__(self, config: SimulatedAnnealingConfig, problem: Problem):
        self.config = config
        self.temperature = self.config.initial_temperature
        self._local_optimum_escapes = 0
        self.problem = problem
        self.cooling_time = 0

    def solve(self):
        self.timer.start_timer()

        solution_state = self.problem.get_random_state()

        while not self.timer.is_timeout():
            try:
                next_state = self.next_state(solution_state)
            except KeyboardInterrupt:
                solution_state = self.best_state
                break
            if next_state:
                solution_state = next_state
            else:
                solution_state = self.best_state

        self.timer.stop_timer()
        print("SOLUTION:\n", "Best:", self.best_state)

        return self.best_state

    def next_state(self, state: list[int]):
        if self.best_state is None:
            self.best_state = state

        if self._is_stuck_in_local_optimum():
            next_state = self.escape_local_optimum(state, self.best_state)
        else:
            next_state = self.find_next_state(state)

        if next_state is not None:
            self._update_state(state, next_state)

        return next_state

    def _update_state(self, state: list[int], new_state: list[int]):
        if self.best_state is None:
            self.best_state = new_state

        if self.problem.improvement(new_state, state) > 0:
            self.steps_from_last_state_update = 0
        else:
            self.steps_from_last_state_update += 1

        if self.problem.improvement(new_state, self.best_state) > 0:
            self.best_state = new_state

    def find_next_state(self, state: list[int]) -> list[int]:
        # — find random neighbour:
        #   [1] create a generator of the random neighbors
        generator = self.problem.get_random_neighbour(state)
        #   [2] use `next` to read a single element from a generator, e.g. `next(generator)`
        neighbour = next(generator)
        # — if the neighbour is better then mark is as the next state:
        #   [1] check for improvement
        if self.problem.improvement(neighbour, state) > 0:
            return neighbour
        # — otherwise calculate the probability of transition
        prob = self.calculate_transition_probability(state, neighbour)
        #   [1] use random() to generate a random number from range [0,1];
        p = random()
        #   [2] compare it to the probability to check if algorithm should go to the new state
        if p > prob:
            # — update temperature using `update_temperature`
            self.update_temperature()
            # — return the new state
            return neighbour

    def calculate_transition_probability(
        self, old_state: list[int], new_state: list[int]
    ) -> float:
        return exp(self.problem.improvement(new_state, old_state) / self.temperature)

    def update_temperature(self):

        # — update self.temperature according to the exponential decrease function:
        #   `T_k = T * a^k`
        # - make sure, the temperature can't go below `self.config.min_temperature`!
        self.temperature = max(
            self.temperature * (self.config.cooling_step**self.cooling_time),
            self.config.min_temperature,
        )
        # - update self.cooling_time
        self.cooling_time += 1

    def reheat(self, from_state: list[int]):
        # — restore the initial temperature based on config (escape_reheat_ratio * initial_temperature)
        #   [1] initial temperature is stored in `self.config.initial_temperature`
        #   [2] you should decrease it a bit (multiply by `self.config.escape_reheat_ratio`)
        self.temperature = (
            self.config.initial_temperature * self.config.escape_reheat_ratio
        )
        # — reset cooling schedule (`self.cooling_time`)
        self.cooling_time = 0
        # — reset counter looking for local minima (`self.steps_from_last_state_update`)
        self.steps_from_last_state_update = 0
        # - return the `from_state`
        return from_state

    def escape_local_optimum(
        self, state: list[int], best_state: list[int]
    ) -> list[int]:
        strategies = ["random", "reheat"]
        strategy = strategies[randint(0, len(strategies) - 1)]
        if strategy == "random":
            return self.problem.get_random_state()
        if strategy == "reheat":
            return self.reheat(state)

    def _is_stuck_in_local_optimum(self):
        return (
            self.steps_from_last_state_update
            >= self.config.local_optimum_moves_threshold
        )
