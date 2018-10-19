import numpy as np
class Sigmoid:
    def sigmoid(self, t):
        s = 1 / (1 + np.power(np.e, (-t)))
        return s
