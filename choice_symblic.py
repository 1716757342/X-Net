import numpy as np
from numpy import *
from Symbol_change import *
from Symbolic_computation import *

def mse_per_row(A, B):
    # calculate the difference
    diff = A - B
    # calculate the squared difference
    squared_diff = diff ** 2
    # calculate the mean squared error
    mse = np.mean(squared_diff, axis=1)
    return mse
def choice_symblic(aa,aal,x,x_i,S,n_node):
    if aa.left != None or aa.right != None:
        if aa.left.val[0] != '0' and aa.right.val[0] == '0':
            sy_E2 = sy(aa.left.val[1]*aa.left.val[2] + aa.left.val[3], x_i)
            # sy_E2_s = list(abs(sy_E2 - aa.val[1]))
            sy_E2_s = list(mse_per_row(sy_E2, aa.val[1]))
            index2 = sy_E2_s.index(min(sy_E2_s))
            # print("index", index)
            # S[index1] = 'sin'
            aa = change(aa, S[index2],x_i)
        elif aa.left.val[0] == '0' and aa.right.val[0] != '0':
            sy_E2 = sy(aa.right.val[1]*aa.right.val[2]+aa.right.val[3], x_i)
            # sy_E2_s = list(abs(sy_E2 - aa.val[1]))
            sy_E2_s = list(mse_per_row(sy_E2, aa.val[1]))
            index2 = sy_E2_s.index(min(sy_E2_s))
            # print("index", index)
            # S[index1] = 'sin'
            aa = change(aa, S[index2],x_i)
        # print('aa.left.val',aa.left)
        elif aa.left.val[0] != '0' and aa.right.val[0] != '0':
            # print('aa.left.val',aa.left.val)
            sy_E2 = sy_n(aa.left.val[1]*aa.left.val[2]+aa.left.val[3], aa.right.val[1]*aa.right.val[2] + aa.right.val[3])
            # sy_E2_s = list(abs(sy_E2 - aa.val[1]))
            sy_E2_s = list(mse_per_row(sy_E2, aa.val[1]))

            sy_E = sy(x_i[0], x_i)
            sy_E_s = list(abs(sy_E - aa.val[1]))
            # sy_E_s = list(mse_per_row(sy_E, aa.val[1]))
            sy_E_s = list(np.mean(np.array(sy_E_s), axis=1))

            sy_El = sy(aa.left.val[1]*aa.left.val[2] + aa.left.val[3], x_i)   ## Determining the new unary operator and the left branch conjunction
            sy_El_s = list(mse_per_row(sy_El, aa.val[1]))

            sy_Er = sy(aa.right.val[1]*aa.right.val[2]+aa.right.val[3], x_i)  ## Determining the new unary operator to combine with the right branch
            sy_Er_s = list(mse_per_row(sy_Er , aa.val[1]))

            lr = [min(sy_E2_s),min(sy_E_s),min(sy_El_s),min(sy_Er_s)]
            lr_ind = lr.index(min(lr))
            if lr_ind == 0:
                index2 = sy_E2_s.index(min(sy_E2_s))
            elif lr_ind == 1:
                index2 = sy_E_s.index(min(sy_E_s))
            elif lr_ind == 2:
                index2 = sy_El_s.index(min(sy_El_s))
            elif lr_ind == 3:
                index2 = sy_Er_s.index(min(sy_Er_s))

            aa = change(aa, S[index2],x_i,lr_ind)
    elif aa.left == None and aa.right == None:
        if len(aal) >= n_node:
            return
        # if aa.val[0] == 'x':
        if 'x' in aa.val[0]:
            sy_E2 = sy(x_i[0], x_i)
            sy_E2_s = list(mse_per_row(sy_E2, aa.val[1]))
            # sy_E2_s = list(abs(sy_E2 - aa.val[1]))
            index2 = sy_E2_s.index(min(sy_E2_s))
            # print("index", index)
            # S[index1] = 'sin'
            aa = change(aa, S[index2],x_i)
    return aa
