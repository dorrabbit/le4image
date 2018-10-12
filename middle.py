from numpy.random import *
import numpy as np
from sigmoid import Sigmoid
import math

class Middle:
    def middle(self, xlist, prenum, nownum, seednum):
        #random
        seed(seednum)
        wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        seed(seednum)
        bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, 100))
        summid = np.dot(wran, xlist) + bran

        #print(summid)
        
        sigclass = Sigmoid()
        sig = sigclass.sigmoid(summid)
        return sig
