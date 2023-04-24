import numpy as np


class Bandit:
    def __init__(self, arms: int) -> None:
        self.slots = {}
        for i in range(arms):
            self.slots[i] = (np.random.uniform(-3, 3), 1)

    def __str__(self) -> str:
        output = ""
        for i, slot in enumerate(self.slots.items()):
            output += f"Slot: {i}, Mean: {slot[0]}, Std: {slot[1]}\n"
        return output

    def pull(self, idx: int) -> float:
        mean, std = self.slots[idx]
        return np.random.normal(mean, std)

    def reset(self, arms: int) -> None:
        for i in range(arms):
            self.slots[i] = (np.random.uniform(-3, 3), 1)
