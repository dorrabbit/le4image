from numpy.random import *
from sigmoid import Sigmoid
import math

class Middle:
    def middle(self, xlist, prenum, nownum, seednum):
        #random
        seed(seednum)
        wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        seed(seednum)
        bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = nownum)
        summid = wran.dot(xlist) + bran

        print(summid)
        
        sigclass = Sigmoid()
        sig = sigclass.sigmoid(summid)
        return sig
