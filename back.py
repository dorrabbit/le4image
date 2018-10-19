class Back:
    def back(self, xlist, midrslt, outrslt, anslist, w_one, w_two): #and more
        #ylist.shape = [10, 100]
        #anslist.shape = [100, 10]

        #softmax + cross_en back-propagation
        en_ak = (outrslt - anslist.T) / 100
        #en_ak.shape = [10,100]

        from diff import Diff
        diffclass = Diff()
        #out back-propagation
        (en_midrslt, en_w_two, en_b_two) = \
                        diffclass.diff(en_ak, w_two, midrslt)
        
        #sigmoid back-propagation
        pre_sig = (1 - en_midrslt) * en_midrslt
        
        #middle back-propagation
        (en_x, en_w_one, en_b_one) = \
                        diffclass.diff(pre_sig, w_one, xlist)

        return (en_w_one, en_w_two, en_b_one, en_b_two)
