import numpy as np
from numpy import *
from torch.autograd import Variable
from numpy import *
from Basic_functions import *

def E_forward(z,m,s):
    global z33
    z33 = z.copy()
    if s == '+':
        z33 = z + m
    if s == '-':
        z33 = z - m
    if s == '*':
        z33 = mul(z,m)
    if s == '/':
        z33 = div(z , m)
    return np.array(z33.copy())

def all_farward(l2,X,d2,b2):
    global stack1 ,v1
    stack1 = []
    for i in range(len(l2)):
        s = l2[-(i + 1)]
        c = round(d2[-(i + 1)])
        b = round(b2[-(i + 1)])
        # if s == 'x':
        if 'x' in s and 'e' not in s:
            stack1.append(X[int(s[1])] * c + b)
        if s == '0':
            mkl = 0
        if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            if s == 'exp':
                v1 = exp_all(stack1.pop())
            if s == 'log':
                v1 = log_all(stack1.pop())
            if s == 'cos':
                v1 = np.cos(stack1.pop())
            if s == 'sin':
                v1 = np.sin(stack1.pop())
            if s == 'sqrt':
                v1 = sqrt_all(stack1.pop())
            stack1.append(v1 * c + b)
        if s in ['+', '-', '*', '/']:
            if s == '+':
                v1 = stack1.pop() + stack1.pop()
            if s == '-':
                v1 = stack1.pop() - stack1.pop()
            if s == '*':
                v1 = mul_all(stack1.pop() , stack1.pop())
            if s == '/':
                v1 = div_all(stack1.pop() , stack1.pop())
            stack1.append(v1 * c + b)
    return stack1[0]

def all_farward_c(l2,X,w):
    global stack1 ,v1,d2,b2
    d2 = w[0:int(0.5*len(w))]
    b2 = w[int(0.5 * len(w)):int(len(w))]
    stack1 = []
    for i in range(len(l2)):
        s = l2[-(i + 1)]
        # c = round(d2[-(i + 1)])
        # b = round(b2[-(i + 1)])
        c = d2[-(i + 1)]
        b = b2[-(i + 1)]
        # if s == 'x':
        if 'x' in s and 'e' not in s:
            stack1.append(X[int(s[1])] * c + b)
        if s == '0':
            mkl = 0 * c + b
        if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            if s == 'exp':
                v1 = exp_all(stack1.pop())
            if s == 'log':
                v1 = log_all(stack1.pop())
            if s == 'cos':
                v1 = np.cos(stack1.pop())
            if s == 'sin':
                v1 = np.sin(stack1.pop())
            if s == 'sqrt':
                v1 = sqrt_all(stack1.pop())
            stack1.append(v1 * c + b)
        if s in ['+', '-', '*', '/']:
            if s == '+':
                v1 = stack1.pop() + stack1.pop()
            if s == '-':
                v1 = stack1.pop() - stack1.pop()
            if s == '*':
                v1 = mul_all(stack1.pop() , stack1.pop())
            if s == '/':
                v1 = div_all(stack1.pop() , stack1.pop())
            stack1.append(v1 * c + b)
    return stack1[0]