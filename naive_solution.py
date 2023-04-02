from problem import Problem


class Naive:
    def __init__(self, problem: Problem):
        self.problem = problem

    def solve(self):
        states = self.problem.get_all_states()
        best_state = states[0]

        for state in states[1:]:
            if self.problem.improvement(state, best_state) > 0:
                best_state = state

        print("SOLUTION:\n", "Best:", best_state)
        return best_state
