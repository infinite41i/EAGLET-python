import numpy as np

class Population:
    def __init__(self, pop_size: int, label_count: int) -> None:
        self.pop_size = pop_size
        self.individuals = np.zeros((pop_size,label_count), np.byte)

