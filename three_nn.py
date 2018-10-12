'''
from hyojun_in import Hyojun_in
hyoinclass = Hyojun_in()
innum = hyoinclass.hyojun_in()
'''

from inputer import Inputer
inclass = Inputer()
dlist = inclass.inputer()[0]
#print(dlist)
anslist = inclass.inputer()[1]
#print(ylist)

import numpy
from middle import Middle
midclass = Middle()
rannumlist=numpy.random.choice(dlist.shape[0], 100, replace=False)
xlist = (dlist[rannumlist] / 255).T
#xlist.shape=[784,100]
              
midrslt = midclass.middle(xlist, 784, 55, 1)
#print("midrslt:")
#print(midrslt.shape)
#midrslt.shape=[55,100]

from outnode import Outnode
outclass = Outnode()
outrslt = outclass.outnode(midrslt, 55, 1)
#print("outrslt:")
#print(outrslt.shape)
#outrslt.shape=[100,10]

ylist = anslist[rannumlist]
#ylist.shape=[100,]

#rslt = outrslt.argmax(axis=1)
#print(rslt, "is written.")

from loss import Loss
lossclass = Loss()
loss_ave = lossclass.loss(outrslt, ylist)
print("loss_ave is ", loss_ave)
