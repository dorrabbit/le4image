import numpy

from inputer import Inputer
inclass = Inputer()
dlist = inclass.inputer(True)[0]
#print(dlist)
anslist = inclass.inputer(True)[1]
#print(ylist)
trainsize = dlist.shape[0] #N

sig_relu = input("sigmoid or relu? s/r > ")
####
if sig_relu == "s":
    srbool = True
    npyfile = 'wb_learn_s.npy'
elif sig_relu == "r":
    srbool = False
    npyfile = 'wb_learn_r.npy'
####

from three_nn import Three_nn
threeclass = Three_nn()

midnum = 55 #middle_node_of_number

w_one = numpy.load(npyfile)[0]
w_two = numpy.load(npyfile)[1]
b_one = numpy.load(npyfile)[2]
b_two = numpy.load(npyfile)[3]

loopnum = 10000
truenum = 0
for i in range(loopnum):
    innum = numpy.random.choice(trainsize, replace=True)
    
    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]
    
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, 1, srbool)[1]
    
    rslt = numpy.argmax(outrslt)

    accbool = ylist == rslt

    if accbool:
        truenum = truenum + 1
    #else:
    #    print(innum, ylist, rslt)
        
print("accuracy is", (truenum / loopnum) * 100, "%.")

    
