import numpy as np

class Diff:
    def diff(self, en_y, w, x):
        #in _two
        #en_y.shape = [10,100]
        #w.shape = [10,55]
        #x(midrslt).shape = [55,100]

        #in _one
        #en_y(pre_sig).shape = [55,100]
        #w.shape = [55,784]
        #x.shape = [784,100]
        
        en_x = np.dot(w.T, en_y)
        #en_x.shape = [55,100]
        en_w = np.dot(en_y, x.T)
        #en_w.shape = [10,55]
        en_b = np.sum(en_y, axis=1)
        #en_b.shape = [10,1]

        return (en_x, en_w, en_b)
