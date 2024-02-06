import numpy as np

class Simulation():

    def __init__(self, mean, variance) -> None:
        self.mean = mean
        self.variance = variance

    def sample(self, n):
        # Take square root of variance because np.random.normal wants the standard deviation, not variance
        return np.random.normal(self.mean, np.sqrt(self.variance), n)