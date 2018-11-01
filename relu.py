import numpy as np
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, t):
        self.mask = t <= 0
        out = t.copy()
        out[self.mask] = 0
        #t = np.where(t > 0, t, 0)
        return out

    def backward(self, en_midrslt):
        en_midrslt[self.mask] = 0
        return en_midrslt
