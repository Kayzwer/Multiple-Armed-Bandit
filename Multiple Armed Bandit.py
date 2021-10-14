import numpy as np

class Bandit:
    """
    A class to represent a multiarm bandit.
    
    Attributes
    ----------
    num_of_arm : int
        the number of arms the bandit can have.

    reward_list : list[list[float]]
        the list that stores the historical reward of each arm.
    
    arms_cumulative_reward : list[int]
        the cumulative reward for each arm.

    average_reward : list[float]
        the list that calculates the mean of each arm.
    
    total_reward : float
        the sum of all reward.

    spin_count : list[int]
        the number of spins done for each arm.
    
    distribution : list[list]
        the mean and std for each arm.
        [mean, std]
    
    Methods
    -------
    spin(n:int)
        generate a number from normal distribution with mean and std generated from normal distribution and value between 0 to 10.
    
    reset()
        reset the bandit state and keep the number of arms.
    """
    def __init__(self, n:int) -> None:
        """
        Parameters
        ----------
        n : int
            the number of arm
        """
        self.num_of_arm = n
        self.reward_list = [[] for _ in range(n)]
        self.arms_cumulative_reward = [0 for _ in range(n)]
        self.average_reward = [0 for _ in range(n)]
        self.total_reward = 0
        self.spin_count = [0 for _ in range(n)]
        self.__distribution = []
        for _ in range(n):
            self.__distribution.append((np.random.rand(2) * 10))
    
    def spin(self, n:int) -> None:
        """
        If n is invalid arm, raise error.

        Parameters
        ----------
        n : int
            the arm n to be spin
        """
        if n >= self.num_of_arm:
            raise Exception("Please select valid arm.")
        else:
            mean, std = self.__distribution[n]
            reward:float = np.random.normal(mean, std)
            self.arms_cumulative_reward[n] += reward
            self.reward_list[n].append(reward)
            self.spin_count[n] += 1
            self.average_reward[n] = np.mean(self.reward_list[n])
            self.total_reward = np.sum(self.arms_cumulative_reward)
    
    def reset(self) -> None:
        """Reset the bandit, but keep the number of arms."""
        self.__init__(self.num_of_arm)


def print_bandit_stats(bandit:Bandit):
    """Print the stats for the bandit."""
    for i in range(len(bandit.average_reward)):
        print(f"Arm {i} have {bandit.average_reward[i]:.3f} average reward and {bandit.arms_cumulative_reward[i]:.3f} total reward with {bandit.spin_count[i]} times spined.")
    print(f"The total reward is {bandit.total_reward:.3f}.")

def random_policy(n:int, iteration:int) -> Bandit:
    """Generate a bandit with n arms and spin randomly.
    
    Parameters
    ----------
    n : int
        the number of arms.
    
    iteration : int
        number of times to spin.
    """
    bandit:Bandit = Bandit(n)
    for _ in range(1000):
        bandit.spin(np.random.randint(0, n))
    print_bandit_stats(bandit)
    return bandit

def greedy_policy(n:int, iteration:int, explore_times:int) -> Bandit:
    """Generate a bandit with n arms and spin greedily.
    
    Parameters
    ----------
    n : int
        the number of arms.
    
    iteration : int
        number of times to spin.

    explore_times : int
        number of times to explore each arm.
    """
    spin_left:int = iteration
    bandit:Bandit = Bandit(n)
    for _ in range(explore_times):
        for j in range(n):
            bandit.spin(j)
            spin_left -= 1
    for _ in range(spin_left):
        bandit.spin(np.argmax(bandit.average_reward))
    print_bandit_stats(bandit)
    return bandit

def epsilon_greedy_policy(n:int, iteration:int, epsilon:float) -> Bandit:
    """Generate a bandit with n arms and spin greedily with probability epsilon.
    
    Parameters
    ----------
    n : int
        the number of arms.
    
    iteration : int
        number of times to spin.

    explore_times : int
        number of times to explore each arm.
    
    epsilon : float
        probability to take explore action.
    """
    bandit:Bandit = Bandit(n)
    for _ in range(iteration):
        if np.random.random() < epsilon:
            bandit.spin(np.random.randint(0, n))
        else:
            bandit.spin(np.argmax(bandit.average_reward))
    print_bandit_stats(bandit)
    return bandit

def optimistic_greedy_policy(n:int, iteration:int, init_estimate: float = 10) -> Bandit:
    """Generate a bandit with n arms with some init_estimate and spin greedily.

    Parameters
    ----------
    n : int
        the number of arms.
    
    iteration : int
        number of times to spin.

    init_estimate : float
        initial value for all arms' estimation.
    """
    bandit:Bandit = Bandit(n)
    bandit.average_reward = [init_estimate for _ in range(n)]
    for _ in range(1000):
        bandit.spin(np.argmax(bandit.average_reward))
    print_bandit_stats(bandit)
    return bandit


if __name__ == "__main__":
    pass
