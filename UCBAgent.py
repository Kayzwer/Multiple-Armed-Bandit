import numpy as np


class UCBAgent:
    def __init__(self, c: float, arms: int, step_length: int) -> None:
        self.pulls_count = np.zeros(arms, dtype=np.int64)
        self.actions_value = np.zeros(arms, dtype=np.float64)
        self.avg_reward_history = np.zeros(step_length, dtype=np.float64)
        self.average_reward = 0.0
        self.total_pull_count = 0
        self.c = c

        self._UCB_cache = np.zeros_like(self.actions_value, dtype=np.float64)

    def __str__(self) -> str:
        output = ""
        for i, action_value in enumerate(self.actions_value):
            output += f"Slot: {i}, Action Value: {action_value}\n"
        output += f"Average reward: {self.average_reward}"
        return output

    def get_action(self, cur_step: int) -> int:
        for i, action_value_pull_count in enumerate(zip(self.actions_value,
                                                        self.pulls_count)):
            action_value, pull_count = action_value_pull_count
            self._UCB_cache[i] = action_value + self.c * \
                np.sqrt(np.log(cur_step) / pull_count) if \
                pull_count > 0 else np.inf
        max_actions = []
        max_UCB = max(self._UCB_cache)
        for i, value in enumerate(self._UCB_cache):
            if value == max_UCB:
                max_actions.append(i)
        action = int(np.random.choice(max_actions, 1))
        self.pulls_count[action] += 1
        self.total_pull_count += 1
        return action

    def update_action_value(self, idx: int, reward: float) -> None:
        self.actions_value[idx] = self.actions_value[idx] + (reward  -
            self.actions_value[idx]) / self.pulls_count[idx]
        self.average_reward = self.average_reward + (reward -
            self.average_reward) / self.total_pull_count
        self.avg_reward_history[self.total_pull_count - 1] = self.average_reward
