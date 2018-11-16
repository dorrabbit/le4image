import numpy
from inputer import Inputer
inclass = Inputer()
from three_nn import Three_nn
threeclass = Three_nn()
from numpy.random import *
from back import Back
backclass = Back()
from hyojun_in import Hyojun_in
from sigmoid import Sigmoid
from relu import Relu
from batchnorm import Batchnorm
batchmoment = 0.9
from dropout import Dropout
dropclass = Dropout()
import sys
from adam import Adam
from convolutional import Convolutional

####
while True:
    iden_train = input("identification or training? i/t > ")
    if iden_train == "i":
        itbool = True
        break
    elif iden_train == "t":
        itbool = False
        break
    else:
        print("illegal input from keyboard.")
####

####
while True:
    sig_relu = input("sigmoid or relu? s/r > ")
    if sig_relu == "s":
        npyfile = 'wb_learn_s.npy'
        srclass = Sigmoid()
        srclass_c = Sigmoid()
        break
    elif sig_relu == "r":
        npyfile = 'wb_learn_r.npy'
        srclass = Relu()
        srclass_c = Relu()
        break
    else:
        print("illegal input from keyboard.")
####
sys.stdout.write("Now loading...")
sys.stdout.flush()

dlist = inclass.inputer(itbool)[0]
#print(dlist)
anslist = inclass.inputer(itbool)[1]
#print(ylist)
trainsize = dlist.shape[0] #N

midnum = 55 #middle_node_of_number

#!!!!!!!
[w_one, w_two, b_one, b_two, gamma_mid, beta_mid, gamma_out, beta_out,\
 running_mean_mid, running_var_mid, running_mean_out, running_var_out,\
 w_conv, b_conv] = numpy.load(npyfile)

if not itbool:
    while True:
        load_yn = input("\r" + "use saved weight? y/n > ")
        if load_yn == "n":
            #initialize
            w_one = normal(loc = 0, scale = 1/numpy.sqrt(784*20) , size = (midnum, 784*20))
            w_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, midnum))
            b_one = normal(loc = 0, scale = 1/numpy.sqrt(784*20) , size = (midnum, 1))
            b_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, 1))
            
            gamma_mid = numpy.ones((midnum, 1))
            beta_mid = numpy.zeros((midnum, 1))
            gamma_out = numpy.ones((10, 1))
            beta_out = numpy.zeros((10, 1))

            w_conv = normal(loc = 0, scale = 1/numpy.sqrt(9) , size = (20, 9))
            b_conv = normal(loc = 0, scale = 1/numpy.sqrt(9) , size = (20, 1))
            
            running_mean_mid = None
            running_var_mid = None
            running_mean_out = None
            running_var_out = None
            break
        elif load_yn == "y":
            break
        else:
            print("illegal input from keyboard.")
    
    eponum = 10

    #update_variable
    #lr = 0.01 #learn rate
    w_one_adamclass = Adam(w_one.shape)
    b_one_adamclass = Adam(b_one.shape)
    w_two_adamclass = Adam(w_two.shape)
    b_two_adamclass = Adam(b_two.shape)
    gamma_mid_adamclass = Adam(gamma_mid.shape)
    beta_mid_adamclass = Adam(beta_mid.shape)
    gamma_out_adamclass = Adam(gamma_out.shape)
    beta_out_adamclass = Adam(beta_out.shape)
    w_conv_adamclass = Adam(w_conv.shape)
    b_conv_adamclass = Adam(b_conv.shape)
    
    for i in range(int(eponum * (trainsize / 100))): #i <- N/B * 100 or so
        rannumlist = numpy.random.choice(trainsize, 100, replace=False)
        image = (dlist[rannumlist] / 255)#.T
        #xlist.shape=[784,100]
        ylist = anslist[rannumlist]
        #ylist.shape=[100,]
        #one-hot-vector-ize
        y_onehot = numpy.identity(10)[ylist]
        #y_onehot.shape=[100,10]
        
        normclass_mid = Batchnorm(gamma_mid, beta_mid, batchmoment, running_mean_mid, running_var_mid)
        normclass_out = Batchnorm(gamma_out, beta_out, batchmoment, running_mean_out, running_var_out)

        convclass = Convolutional(w_conv, b_conv, srclass_c)
        xlist = convclass.forward(image) #xlist.shape = (20*28*28,100)
        
        (midrslt, outrslt, loss_ave, \
         new_running_mean_mid, new_running_var_mid, new_running_mean_out, new_running_var_out) = \
                threeclass.three_nn(xlist, y_onehot, \
                                    w_one, b_one, w_two, b_two, \
                                    100, itbool, srclass, dropclass, normclass_mid, normclass_out)
        print("(No.", "{0:02d}".format(int(i*100/trainsize)), " epoch) ", sep="", end="")
        print(loss_ave)
        
        (en_x, en_w_one, en_w_two, en_b_one, en_b_two, en_gamma_mid, en_beta_mid, en_gamma_out, en_beta_out) = \
                backclass.back(xlist, midrslt, outrslt, y_onehot, w_one, w_two, \
                               srclass, dropclass, normclass_mid, normclass_out)
        (en_w_conv, en_b_conv) = convclass.backward(en_x)
    
        #update w and b
        w_one = w_one - w_one_adamclass.update_amount(en_w_one)
        b_one = b_one - b_one_adamclass.update_amount(en_b_one)
        w_two = w_two - w_two_adamclass.update_amount(en_w_two)
        b_two = b_two - b_two_adamclass.update_amount(en_b_two)
        gamma_mid = gamma_mid - gamma_mid_adamclass.update_amount(en_gamma_mid)
        beta_mid = beta_mid - beta_mid_adamclass.update_amount(en_beta_mid)
        gamma_out = gamma_out - gamma_out_adamclass.update_amount(en_gamma_out)
        beta_out = beta_out - beta_out_adamclass.update_amount(en_beta_out)
        w_conv = w_conv - w_conv_adamclass.update_amount(en_w_conv)
        b_conv = b_conv - b_conv_adamclass.update_amount(en_b_conv)
        
        running_mean_mid = new_running_mean_mid
        running_var_mid = new_running_var_mid
        running_mean_out = new_running_mean_out
        running_var_out = new_running_var_out
        
    wb = numpy.array([w_one, w_two, b_one, b_two, \
                      gamma_mid, beta_mid, gamma_out, beta_out, \
                      running_mean_mid, running_var_mid, running_mean_out, running_var_out, \
                      w_conv, b_conv])
    numpy.save(npyfile, wb)

else:
    while True:
        one_or_measure = input("\r" + "one-image-identification or accuracy-measurement? 1/% > ")
        if one_or_measure == "1":
            innum = Hyojun_in.hyojun_in()
            batchsize = 1
            break
        elif one_or_measure == "%":
            sys.stdout.write("Now loading...")
            sys.stdout.flush()
            
            innum = numpy.random.choice(trainsize, 10000, replace=True)
            batchsize = 10000
            break
        else:
            print("illegal input from keyboard.")
        
    image = (dlist[innum] / 255).reshape(batchsize, 1, 28, 28)
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]

    normclass_mid = Batchnorm(gamma_mid, beta_mid, batchmoment, running_mean_mid, running_var_mid)
    normclass_out = Batchnorm(gamma_out, beta_out, batchmoment, running_mean_out, running_var_out)

    convclass = Convolutional(w_conv, b_conv, srclass_c)
    xlist = convclass.forward(image)
    
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, \
                                  batchsize, itbool, srclass, dropclass, normclass_mid, normclass_out)[1]
    rslt = numpy.argmax(outrslt, axis=0)
    
    if one_or_measure == "1":
        print(rslt[0], " is written in No.", innum, " image, maybe.", sep="")
        print(ylist, " is written in No.", innum, " image, definitely.", sep="")
        print("my identification is ", rslt[0] == ylist, ".", sep="")
    elif one_or_measure == "%":
        checkanslist = rslt == ylist
        print("\r" + "accuracy is ", numpy.sum(checkanslist) / 100, "%.", sep="")
