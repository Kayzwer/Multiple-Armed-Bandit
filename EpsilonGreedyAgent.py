import numpy as np
from typing import List


class EpsilonGreedyAgent:
    def __init__(self, epsilon: float, init_value: float, arms: int,
                 step_length: int) -> None:
        self.pulls_count = np.zeros(arms, dtype=np.int64)
        self.actions_value = np.full(arms, init_value, dtype=np.float32)
        self.avg_reward_history = np.zeros(step_length, dtype=np.float64)
        self.epsilon = epsilon
        self.average_reward = 0.0
        self.total_pull_count = 0

    def __str__(self) -> str:
        output = ""
        for i, action_value in enumerate(self.actions_value):
            output += f"Slot: {i}, Action Value: {action_value}\n"
        output += f"Average reward: {self.average_reward}"
        return output

    def get_action(self) -> int:
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            actions = self.get_highest_value_actions()
            action = np.random.choice(actions)
        else:
            action = np.random.randint(0, 9)
        self.pulls_count[action] += 1
        self.total_pull_count += 1
        return action

    def get_highest_value_actions(self) -> List[int]:
        output = []
        max_ = max(self.actions_value)
        for i, action_value in enumerate(self.actions_value):
            if max_ == action_value:
                output.append(i)
        return output

    def update_action_value(self, idx: int, reward: float) -> None:
        self.actions_value[idx] = self.actions_value[idx] + (reward  -
            self.actions_value[idx]) / self.pulls_count[idx]
        self.average_reward = self.average_reward + (reward -
            self.average_reward) / self.total_pull_count
        self.avg_reward_history[self.total_pull_count - 1] = self.average_reward
