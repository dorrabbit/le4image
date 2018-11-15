import numpy as np
from diff import Diff

class convolutional:
    def __init__(self, w, b, srclass):
        self.fsize_x = 3
        self.fsize_y = 3
        self.fsize_z = 1
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
        image = np.pad(image, [(0,0),(0,0),(1,1),(1,1)], 'constant')
        #image.shape = (100, 1, 30, 30)

        x_ = np.zeros((batchnum, ch, 3, 3, height, width))

        for y in range(3):
            y_max = y + height
            for x in range(3):
                x_max = x + width
                x_[:, :, y, x, :, :] = image[:, :, y:y_max, x:x_max]

        self.x = x_.transpose(0, 4, 5, 1, 2, 3).reshape(batchnum*height*width, -1).T #.T <- ayashii
        #self.x.shape(1*3*3, 100*28*28)
        
        y = np.dot(self.w, self.x) + self.b
        #y.shape = (20, 100*28*28)
        
        convrslt = srclass.forward(y)

        return convrslt

    def backward(self, en_y):
        pre_sig = srclass.backward(en_y)
        (en_x, en_w, en_b) = Diff.(pre_sig, self.w, self.x)
        return (en_x, en_w, en_b)
