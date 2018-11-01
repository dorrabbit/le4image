import numpy
from middle import Middle
from outnode import Outnode
from loss import Loss

class Three_nn:
    def three_nn(self, xlist, ylist, w_one, b_one, w_two, b_two, batchnum, itbool, srbool, srclass):
        #xlist.shape=[784,100]
        #ylist.shape=[100,10] #one-hot
        
        midclass = Middle()
        midrslt = midclass.middle(xlist, 784, 55, w_one, b_one, batchnum, itbool, srbool, srclass)
        #print("midrslt:")
        #print(midrslt.shape)
        #midrslt.shape=[55,100]
        
        outclass = Outnode()
        outrslt = outclass.outnode(midrslt, 55, w_two, b_two, batchnum)
        #print("outrslt:")
        #print(outrslt.shape)
        #outrslt.shape=[10,100]
        
        lossclass = Loss()
        loss_ave = lossclass.loss(outrslt.T, ylist)
        
        return (midrslt, outrslt, loss_ave)
