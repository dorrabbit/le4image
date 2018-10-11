from hyojun_in import Hyojun_in
hyoinclass = Hyojun_in()
innum = hyoinclass.hyojun_in()

from inputer import Inputer
inclass = Inputer()
dlist = inclass.inputer()
#print(dlist[innum])

from middle import Middle
midclass = Middle()
xlist = dlist[innum] / 255
sig = midclass.middle(xlist, 784, 55, 1)
print("midrslt:")
print(sig)

from outnode import Outnode
outclass = Outnode()
soft = outclass.outnode(sig, 55, 1)
print("out:")
print(soft)

import numpy
rslt = numpy.argmax(soft)
print(rslt, "is written in number", innum, "image.")
