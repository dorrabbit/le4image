import numpy as np
class Relu:
    def relu(t):
        t = np.where(t > 0, t, 0)
        return t
