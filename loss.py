import numpy as np
class Loss:
    def loss(self, outrslt, y_onehot):
        #outrslt.shape=[100,10]
        #y_onehot.shape=[100,10]

        crsen = np.diag(0 - np.dot(y_onehot, np.log(outrslt).T))
        #print(crsen.shape)

        crsen_ave = sum(crsen) / 100

        return crsen_ave
