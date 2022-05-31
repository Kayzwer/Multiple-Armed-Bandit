import numpy as np


class Bandit:
    def __init__(
        self,
        n_bandit: int,
        min_exp: float,
        max_exp: float,
        min_var: float,
        max_var: float
    ) -> None:
        self.n_bandit = n_bandit
        self.bandit_params = np.zeros(n_bandit, dtype=np.ndarray)
        for i in range(n_bandit):
            self.bandit_params[i] = np.array(
                [np.random.uniform(min_exp, max_exp),
                 np.random.uniform(min_var, max_var)])

    def pull(self, index: int) -> float:
        assert 0 <= index < self.n_bandit
        selected_param = self.bandit_params[index]
        return np.random.normal(selected_param[0], selected_param[1])

    def __str__(self) -> str:
        output_str = f"Number of bandit: {self.n_bandit}\n"
        for i, data in enumerate(self.bandit_params):
            output_str += f"Bandit {i}: {str(data)}\n"
        return output_str


class Policy:
    def __init__(
        self,
        n_action: int,
        learning_rate: float
    ) -> None:
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.h = np.zeros(n_action, dtype=np.float32)
        self.action_taken_count = [0 for _ in range(n_action)]
        self.exp_reward = 0.0
        self.num_of_trial = 0
        self.total_reward = 0.0

    def get_action_prob(self, index: int) -> float:
        assert 0 <= index < self.n_action
        temp = np.exp(self.h - self.h.max())
        return temp[index] / np.sum(temp)

    def choose_action(self) -> int:
        temp = np.exp(self.h - self.h.max())
        deno = np.sum(temp)
        actions_prob = np.zeros(self.n_action, dtype=np.float32)
        for i, h in enumerate(self.h):
            actions_prob[i] = round(np.exp(h) / deno, 3)
        actions_prob /= actions_prob.sum()
        return np.random.choice(self.n_action, p=actions_prob)

    def update(self, action: int, reward: float) -> None:
        assert 0 <= action < self.n_action
        self.action_taken_count[action] += 1
        self.total_reward += reward
        self.num_of_trial += 1
        self.exp_reward = (reward - self.exp_reward) / self.num_of_trial + \
            self.exp_reward
        for i in range(self.n_action):
            if i == action:
                self.h[i] += (self.learning_rate * (reward - self.exp_reward) *
                              (1 - self.get_action_prob(action)))
            else:
                self.h[i] -= (self.learning_rate * (reward - self.exp_reward) *
                              self.get_action_prob(action))


if __name__ == "__main__":
    env = Bandit(10, 0, 10, 0, 10)
    agent = Policy(10, 0.1)

    for _ in range(1000):
        action = agent.choose_action()
        reward = env.pull(action)
        agent.update(action, reward)

    print(env)
    print(agent.action_taken_count)
    print(agent.exp_reward)
    print(agent.total_reward)
