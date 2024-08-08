import numpy as np
from numpy import *
from Basic_functions import *
# def sy(x,x_1):
#     return np.array(list([x+x_1,x-x_1,mul(x,x_1),div(x,x_1),np.sin(x),np.cos(x),exp_s(x),log_s(x),sqrt_s(x),x_1]))
# def sy(x,x_1): ######
#     return np.array(list([x+x_1[0],x-x_1[0],mul(x,x_1[0]),np.sin(x),np.cos(x),exp_s(x),log_s(x),x_1[0],x_1[1],x_1[2]]))

def sy(x,x_1): ######
    sy_list = list([x+x_1[0],x-x_1[0],mul_all(x,x_1[0]),np.sin(x),np.cos(x)])
    for i in range(len(x_1)):
        sy_list.append(x_1[i])
    return np.array(sy_list)
# def sy(x,x_1): ######
#     sy_list = list([x+x_1[0],x-x_1[0],mul_all(x,x_1[0])])
#     for i in range(len(x_1)):
#         sy_list.append(x_1[i])
#     return np.array(sy_list)
# def sy(x,x_1): ######
#     sy_list = list([x+x_1[0],x-x_1[0],mul_all(x,x_1[0]),np.sin(x),np.cos(x),exp_all(x),sqrt_all(x),log_all(x)])
#     for i in range(len(x_1)):
#         sy_list.append(x_1[i])
#     return np.array(sy_list)
def sy_n(z,m):
    return np.array(list([z+m,z-m,mul_all(z,m),div_all(z,m)]))
