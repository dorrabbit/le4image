import numpy as np
class Sigmoid:
    def forward(self, t):
        s = 1 / (1 + np.exp(-t))
        return s

    def backward(self, en_midrslt):
        pre_sig = (1 - en_midrslt) * en_midrslt
        return pre_sig
