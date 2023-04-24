from CategoricalDistribution import CategoricalDistribution
import numpy as np


class GradientAgent:
    def __init__(self, alpha: float, arms: int, step_length: int) -> None:
        self.preferences = np.zeros(arms, dtype=np.float32)
        self.avg_reward_history = np.zeros(step_length, dtype=np.float64)
        self.average_reward = 0.0
        self.total_pull_count = 0
        self.alpha = alpha
        self.action_dist = CategoricalDistribution(self.softmax(
            self.preferences))

    def get_action(self) -> int:
        self.total_pull_count += 1
        return self.action_dist.sample()

    def update(self, action: int, reward: float) -> None:
        self.average_reward = self.average_reward + (reward -
            self.average_reward) / self.total_pull_count
        self.avg_reward_history[self.total_pull_count - 1] = self.average_reward
        diff = reward - self.average_reward
        for i, preference in enumerate(self.preferences):
            if i == action:
                self.preferences[i] = preference + self.alpha * diff * \
                    (1. - self.action_dist.probs[i])
            else:
                self.preferences[i] = preference - self.alpha * diff * \
                    self.action_dist.probs[i]
        self.action_dist.reset_probs(self.softmax(self.preferences))

    def __str__(self) -> str:
        output = ""
        for i, prob in enumerate(self.action_dist.probs):
            output += f"Slot: {i}, Prob: {prob}\n"
        output += f"Average reward: {self.average_reward}"
        return output

    
    @staticmethod
    def softmax(preferences: np.ndarray) -> np.ndarray:
        e_preferences = np.exp(preferences)
        return e_preferences / np.sum(e_preferences)
