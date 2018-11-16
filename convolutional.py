import numpy as np
from diff import Diff

class Convolutional:
    def __init__(self, w, b, srclass):
        self.k = 20
        self.w = w
        #w.shape = (20, 3*3*1)
        self.b = b
        #b.shape = (20, 1)
        self.x = None
        #x.shape = (3*3*1, 28*28*100)
        self.srclass = srclass
        
    def forward(self, image):
        #image.shape(100, 1, 28, 28)

        (batchnum, ch, height, width) = image.shape
        #print(batchnum, height, width)
        image = np.pad(image, [(0,0),(0,0),(1,1),(1,1)], 'constant')
        #image.shape = (100, 1, 30, 30)

        x_ = np.zeros((batchnum, ch, 3, 3, height, width))

        for y in range(3):
            y_max = y + height
            for x in range(3):
                x_max = x + width
                x_[:, :, y, x, :, :] = image[:, :, y:y_max, x:x_max]
                
        self.x = x_.transpose(0, 4, 5, 1, 2, 3).reshape(batchnum*height*width, -1).T #.T <- ayashii <- maybe ok
        #self.x.shape(1*3*3, 100*28*28)
        
        y_ = np.dot(self.w, self.x) + self.b
        #y_.shape = (20, 100*28*28)
        y = y_.reshape(20, batchnum, 28*28).transpose(0, 2, 1).reshape(-1, batchnum)
        
        convrslt = self.srclass.forward(y) #reshape <- ayashii <- maybe ok

        return convrslt

    def backward(self, en_y):
        pre_sig = self.srclass.backward(en_y)
        #print(pre_sig.shape)
        (en_x, en_w, en_b) = Diff.diff(pre_sig.reshape(20, -1), self.w, self.x)  #reshape <- ayashii
        return (en_w, en_b)
