import numpy as np
class Sigmoid:
    def __init__(self):
        self.s = None
        
    def forward(self, t):
        self.s = 1 / (1 + np.exp(-t))
        return self.s

    def backward(self, en_midrslt):
        sig_dash = (1 - self.s) * self.s
        return en_midrslt * sig_dash
