from numpy.random import *
import numpy as np
from sigmoid import Sigmoid
import math

class Middle:
    def middle(self, xlist, prenum, nownum, w, b, batchnum):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, 1))

        xlist = xlist.reshape(prenum,batchnum)
        summid = np.dot(w, xlist) + b

        #print(summid.shape)
        
        sigclass = Sigmoid()
        sig = sigclass.sigmoid(summid)
        return sig
