from collections.abc import Generator
from copy import deepcopy
from time import time
import pandas as pd
import pandas.core.frame as pd_frame
import pandas.core.series
from random import shuffle
from abc import ABC, abstractmethod
from typing import TypeVar

series_type = TypeVar("pandas.core.series.Series")



class Problem(ABC):

    def __init__(self, computers: pd_frame.DataFrame, employees: pd_frame.DataFrame):
        self.employees = employees
        self.computers = computers
        self.parameters = computers.columns[1:]

    @abstractmethod
    def get_value_by_needs(self, value: int, employee_type: str) -> int:
        pass

    def get_random_state(self):
        state = [c for c in range(self.computers.shape[0])]
        shuffle(state)
        return state[:3]

    def calculate_computer_value_for_employee(self, computer: series_type, employee: series_type) -> float:
        result = sum(
            [employee[col] * self.get_value_by_needs(computer[col], employee["utilization_bin"]) for col in
             self.parameters[:-1]])
        result += employee[self.parameters[-1]] * computer[self.parameters[-1]]
        return result

    def get_best_from_three_for_employee(self, computers_indexes: list[int], employee: series_type) -> float:
        return max([self.calculate_computer_value_for_employee(self.computers.iloc[c], employee)
                    for c in computers_indexes])

    def calculate_state_cost(self, state: list[int]) -> float:
        cost = 0
        for _, e in self.employees.iterrows():
            cost += self.get_best_from_three_for_employee(state, e)
        return cost

    def improvement(self, new_state: list[int], old_state: list[int]) -> float:
        return self.calculate_state_cost(new_state) - self.calculate_state_cost(old_state)

    def get_random_neighbour(self, state: list[int]) -> Generator:
        neighbour_states = [(i, j) for i in range(len(state))
                            for j in [x for x in range(self.computers.shape[0]) if x not in state]]

        shuffle(neighbour_states)

        for i, j in neighbour_states:
            new_state = deepcopy(state)
            new_state[i] = j
            yield new_state

    def get_all_states(self):
        result = []
        for i in range(self.computers.shape[0]):
            result += [[k, j, i] for j in range(i) for k in range(j)]

        return result


class ProblemMax(Problem, ABC):
    max_values = {"high": 10, "medium": 7, "low": 3}

    def get_value_by_needs(self, value: int, employee_type: str) -> int:
        return min(value, self.max_values.get(employee_type))


class ProblemScale(Problem, ABC):
    scale_values = {"high": 1, "medium": 3 / 2, "low": 3 / 1}

    def get_value_by_needs(self, value: int, employee_type: str) -> int:
        return min(value * self.scale_values.get(employee_type), 10)




class Timer:
    def __init__(self, time_limit: float):
        self._time_limit = time_limit
        self._terminated = False

    def start_timer(self):
        self.start_time = time()

    def wall_time(self) -> float:
        return time() - self.start_time

    def is_timeout(self):
        return self.wall_time() > self._time_limit

    def stop_timer(self):
        self.total_time = self.wall_time()




# import pandas as pd
#
# utilization = pd.read_csv("https://raw.githubusercontent.com/shubhamkalra27/dsep-2020/main/datasets/util_b_emp.csv")
# survey = pd.read_csv("https://raw.githubusercontent.com/shubhamkalra27/dsep-2020/main/datasets/survey_emp.csv")
# computers = pd.read_csv("https://raw.githubusercontent.com/shubhamkalra27/dsep-2020/main/datasets/vendor_options.csv")
#
# employees = utilization.merge(survey, left_on="employee_id", right_on="employee_id")
# problem = ProblemMax(computers, employees)
# print(len(problem.get_all_states()))


