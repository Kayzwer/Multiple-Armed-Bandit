import numpy as np


class CategoricalDistribution:
    def __init__(self, probs: np.ndarray)-> None:
        self.probs = probs
        self.cmf = self.get_cmf()

    def sample(self) -> int:
        rand = np.random.uniform(0., 1.)
        for i, category_prob in enumerate(self.cmf):
            if rand < category_prob:
                return i
        return -1

    def get_cmf(self) -> np.ndarray:
        cdf = np.zeros_like(self.probs, dtype=np.ndarray)
        running_sum = 0.0
        for i, prob in enumerate(self.probs):
            running_sum += prob
            cdf[i] = running_sum
        return cdf
    
    def reset_probs(self, probs: np.ndarray) -> None:
        self.__init__(probs)
