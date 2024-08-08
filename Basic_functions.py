from numpy import *
import numpy as np


def MaxMinN(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

max_val = 10e40
def div(x1,x2):
    # if np.isnan(x1) or np.isnan(x2):
    #     return 1
    # else:
    if abs((x1/(abs(x2)+0.001)))>=1000:
        return 1000 * (x1/(abs(x2)+0.00001))/abs((x1/(abs(x2)+0.00001)))
    else:
        return (x1/(abs(x2)+0.001))

def log_s(x1):
    # if np.isnan(x1) :
    #     return 1
    # else:
    if abs(x1) <= 0.01 :
        return -1
    elif abs(np.log(abs(x1)))>=1000:
        return 1000.0 * (np.log(abs(x1))/abs(np.log(abs(x1))))
    else:
        return np.log(abs(x1))
def sqrt_s(x1):
    # if np.isnan(x1) or np.isnan(x1):
    #     return 1
    # else:
    if np.sqrt(abs(x1))>=1000:
        return 1000.0
    else:
        return np.sqrt(abs(x1))

def exp_s(x):
    # if np.isnan(x):
    #     return 1
    # else:
    if x>=10:
        return np.exp(10)
    else:
        return np.exp(x)
def mul(x1,x2):
    if abs(x1)>1000 and abs(x2)>1000:
        return 10000
    elif abs(x1 * x2) >= 10000:
        return 10000
    else:
        return x1 * x2


def div_all(x1,x2,pwd = 1):
    if pwd == 1:
        aa = x1/(abs(x2)+0.000001)
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa
    if pwd == 0:
        return x1/x2

def log_all(x1,pwd =1):
    if pwd == 1:
        aa = np.log(abs(x1))
        aa[np.argwhere(abs(x1) <= 0.00001)] = -100
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val

        # x1[np.argwhere(x1 <= 0.00001)] = 0.00001
        # aa = np.log(abs(x1))
        # aa[np.argwhere(abs(x1) <= 0.001)] = -100
        # aa[np.argwhere(aa >= max_val)] = max_val
        # aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa

    if pwd == 0:
        return np.log(x1)
def sqrt_all(x1,pwd = 0):

    if pwd == 1:
        aa = np.sqrt(abs(x1))
        aa[np.argwhere(aa >= max_val)] = max_val
        return aa
    if pwd == 0:
        # print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        return sqrt(abs(x1))

def exp_all(x,pwd = 1):
    if pwd == 1:
        aa = np.exp(x)
        aa[np.argwhere(aa >= max_val)] = max_val
        return aa
    if pwd == 0:
        return(np.exp(x))
def mul_all(x1,x2,pwd = 1):
    if pwd == 1:
        aa = x1 * x2
        aa[np.argwhere(aa >= max_val)] = max_val
        aa[np.argwhere(aa <= -max_val)] = -max_val
        return aa
    if pwd == 0:
        return x1 * x2
