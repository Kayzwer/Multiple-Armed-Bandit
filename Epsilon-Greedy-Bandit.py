import random
from typing import List


class Bandit:
    def __init__(self, arms: int) -> None:
        self.slots = {}
        for i in range(arms):
            self.slots[i] = (random.randint(-3, 3), 1)

    def __str__(self) -> str:
        output = ""
        for i, slot in enumerate(self.slots.items()):
            output += f"Slot: {i}, Mean: {slot[0]}, Std: {slot[1]}\n"
        return output

    def pull(self, idx: int) -> float:
        mean, std = self.slots[idx]
        return random.gauss(mean, std)


class Agent:
    def __init__(self, epsilon: float, arms: int, step_length: int) -> None:
        assert 0 <= epsilon <= 1
        self.pulls_count = [0 for _ in range(arms)]
        self.actions_value = [0. for _ in range(arms)]
        self.avg_reward_history = [0. for _ in range(step_length)]
        self.epsilon = epsilon
        self.average_reward = 0.0
        self.total_pull_count = 0

    def __str__(self) -> str:
        output = ""
        for i, action_value in enumerate(self.actions_value):
            output += f"Slot: {i}, Action Value: {action_value}\n"
        output += f"Average reward: {self.average_reward}"
        return output

    def get_actions(self) -> int:
        if random.uniform(0, 1) < 1 - self.epsilon:
            actions = self.get_highest_value_actions()
            action = random.choice(actions)
        else:
            action = random.randint(0, 9)
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


if __name__ == "__main__":
    step_length = 1000
    arms = 10
    agent1_avg_reward = 0.0
    agent2_avg_reward = 0.0
    agent3_avg_reward = 0.0
    agent4_avg_reward = 0.0
    for i in range(2000):
        bandit = Bandit(arms)
        agent1 = Agent(0.1, arms, step_length)
        agent2 = Agent(0.01, arms, step_length)
        agent3 = Agent(0.0, arms, step_length)
        agent4 = Agent(0.2, arms, step_length)
        for _ in range(step_length):
            action1 = agent1.get_actions()
            action2 = agent2.get_actions()
            action3 = agent3.get_actions()
            action4 = agent4.get_actions()
            reward1 = bandit.pull(action1)
            reward2 = bandit.pull(action2)
            reward3 = bandit.pull(action3)
            reward4 = bandit.pull(action4)
            agent1.update_action_value(action1, reward1)
            agent2.update_action_value(action2, reward2)
            agent3.update_action_value(action3, reward3)
            agent4.update_action_value(action4, reward4)
        agent1_avg_reward = agent1_avg_reward + (agent1.average_reward -
            agent1_avg_reward) / (i + 1)
        agent2_avg_reward = agent2_avg_reward + (agent2.average_reward -
            agent2_avg_reward) / (i + 1)
        agent3_avg_reward = agent3_avg_reward + (agent3.average_reward -
            agent3_avg_reward) / (i + 1)
        agent4_avg_reward = agent4_avg_reward + (agent4.average_reward -
            agent4_avg_reward) / (i + 1)
    print(f"Agent1 (0.1): {agent1_avg_reward}, Agent2 (0.01): {agent2_avg_reward}"
          f", Agent3 (0.0): {agent3_avg_reward}, Agent4 (0.2): {agent4_avg_reward}")
