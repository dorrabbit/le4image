class Three_nn:
    def three_nn(self, xlist, ylist, w_one, b_one, w_two, b_two):
        #xlist.shape=[784,100]
        #ylist.shape=[100,10] #one-hot
        
        import numpy
        from middle import Middle
        midclass = Middle()
        midrslt = midclass.middle(xlist, 784, 55, w_one, b_one)
        #print("midrslt:")
        #print(midrslt.shape)
        #midrslt.shape=[55,100]
        
        from outnode import Outnode
        outclass = Outnode()
        outrslt = outclass.outnode(midrslt, 55, w_two, b_two)
        #print("outrslt:")
        #print(outrslt.shape)
        #outrslt.shape=[10,100]
        
        from loss import Loss
        lossclass = Loss()
        loss_ave = lossclass.loss(outrslt.T, ylist)
        
        return (midrslt, outrslt, loss_ave)
