import numpy
'''
from hyojun_in import Hyojun_in
hyoinclass = Hyojun_in()
innum = hyoinclass.hyojun_in()
'''
#print("start") #measure time

from inputer import Inputer
inclass = Inputer()
dlist = inclass.inputer()[0]
#print(dlist)
anslist = inclass.inputer()[1]
#print(ylist)
trainsize = dlist.shape[0] #N

#print("start three_nn") #measure time
from three_nn import Three_nn
threeclass = Three_nn()

midnum = 55 #middle_node_of_number

load_yn = input("use saved weight? y/n >")
if load_yn == "y":
    w_one = numpy.load('wb_learn.npy')[0]
    w_two = numpy.load('wb_learn.npy')[1]
    b_one = numpy.load('wb_learn.npy')[2]
    b_two = numpy.load('wb_learn.npy')[3]
else:
    #initialize
    from numpy.random import *
    w_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 784))
    w_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, midnum))
    b_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 1))
    b_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, 1))
    
eponum = 100

for i in range(int(eponum * (trainsize / 100))): #i <- N/B * 100 or so
    rannumlist = numpy.random.choice(trainsize, 100, replace=False)
    xlist = (dlist[rannumlist] / 255).T
    #xlist.shape=[784,100]
    ylist = anslist[rannumlist]
    #ylist.shape=[100,]
    #one-hot-vector-ize
    y_onehot = numpy.identity(10)[ylist]
    #y_onehot.shape=[100,10]

    (midrslt, outrslt, loss_ave) = \
            threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two)
    print(loss_ave)

    from back import Back
    backclass = Back()
    (en_w_one, en_w_two, en_b_one, en_b_two) = \
            backclass.back(xlist, midrslt, outrslt, y_onehot, w_one, w_two)

    en_b_one = en_b_one.reshape(midnum,1)
    en_b_two = en_b_two.reshape(10,1)
    
    #update w and b
    lr = 0.01 #learn rate
    w_one = w_one - lr * en_w_one
    w_two = w_two - lr * en_w_two
    b_one = b_one - lr * en_b_one
    b_two = b_two - lr * en_b_two

wb = numpy.array([w_one, w_two, b_one, b_two])
numpy.save('wb_learn.npy', wb)
