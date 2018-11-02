import numpy
from middle import Middle
from outnode import Outnode
from loss import Loss

class Three_nn:
    def three_nn(self, xlist, ylist, w_one, b_one, w_two, b_two, \
                 batchnum, itbool, srclass, dropclass, normclass_mid, normclass_out):
        #xlist.shape=[784,100]
        #ylist.shape=[100,10] #one-hot
        
        (midrslt, running_mean_mid, running_var_mid) = \
                Middle.middle(xlist, 784, 55, w_one, b_one, batchnum, itbool, srclass, normclass_mid)
        #print("midrslt:")
        #print(midrslt.shape)
        #midrslt.shape=[55,100]

        dropclass.forward(midrslt, itbool)
        
        outclass = Outnode()
        (outrslt, running_mean_out, running_var_out) = \
                outclass.outnode(midrslt, 55, w_two, b_two, batchnum, itbool, normclass_out)
        #print("outrslt:")
        #print(outrslt.shape)
        #outrslt.shape=[10,100]
        
        lossclass = Loss()
        loss_ave = lossclass.loss(outrslt.T, ylist)
        
        return (midrslt, outrslt, loss_ave, running_mean_mid, running_var_mid, running_mean_out, running_var_out)
