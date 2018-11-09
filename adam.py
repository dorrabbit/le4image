import numpy as np
class Adam:
    def __init__(self, shape):
        self.t = 0
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.alpha = 0.001
        self.beta_one = 0.9
        self.beta_two = 0.999
        self.epsilon = 10e-8

    def update_amount(self, en_w):
        self.t = self.t + 1
        self.m = self.beta_one * self.m + (1 - self.beta_one) * en_w
        self.v = self.beta_two * self.v + (1 - self.beta_two) * (en_w ** 2)
        m_norm = self.m / (1 - self.beta_one ** self.t)
        v_norm = self.v / (1 - self.beta_two ** self.t)
        amount = self.alpha * m_norm / (np.sqrt(v_norm) + self.epsilon)

        return amount
