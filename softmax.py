import numpy
class Softmax:
    def softmax(self, midrslt):
        maxa = max(midrslt)
        suma = sum(numpy.exp(midrslt - maxa))
        ysec = numpy.exp(midrslt - maxa) / suma
        return ysec
