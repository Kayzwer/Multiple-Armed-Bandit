import numpy as np
from Bandit import Bandit
from UCBAgent import UCBAgent
import matplotlib.pyplot as plt


if __name__ == "__main__":
    step_length = 1000
    arms = 10
    c_params = np.arange(0., 2.1, 0.1)
    bandit = Bandit(arms)
    agents_avg_reward = np.zeros(len(c_params), dtype=np.float32)
    for episode in range(2000):
        ucb_agents = np.array([UCBAgent(c, arms, 1000) for c in c_params], dtype=UCBAgent)
        for i in range(step_length):
            for agent in ucb_agents:
                action = agent.get_action(i + 1)
                reward = bandit.pull(action)
                agent.update_action_value(action, reward)
        for i, agent in enumerate(ucb_agents):
            agents_avg_reward[i] = agents_avg_reward[i] + (agent.average_reward \
                - agents_avg_reward[i]) / (episode + 1)
        bandit.reset(arms)
        print(f"Episode: {episode + 1} done")
    plt.figure()
    plt.plot(c_params, agents_avg_reward)
    plt.xlabel("c")
    plt.ylabel("Avg Reward")
    plt.savefig("c_vs_reward.png")
