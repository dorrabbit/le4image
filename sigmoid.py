import math
class Sigmoid:
    def sigmoid(self, t):
        s = 1 / (1 + math.e ** (-t))
        return s
