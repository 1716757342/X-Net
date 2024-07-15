import copy
import time
from scipy.optimize import minimize
from numpy import *
from copy import deepcopy
from Basic_functions import *
def E_grad(z,z2,g): ## Taking the derivative of the preceding binary operator
    if z == '+':
        return 1 * g
    if z == '-':
        return 1 * g
    if z == '*':
        return mul_all(z2 , g)
    if z == '/':
        return div_all(g,(z2))
def E_grad_2(z,z1,z2,g): ## Taking the derivative of a binary operator symbol
    if z == '+':
        return 1 * g
    if z == '-':
        return -1 * g
    if z == '*':
        return mul_all(z1 , g)
    if z == '/':
        return mul_all(-z1 ,mul_all( div_all(1,(z2 ** 2)) , g))

def grad(z,g,z_i,x1):
    if z == '+':
        return g
    if z == '-':
        return g
    if z == '*':
        return mul_all(g , x1)
    if z == '/':
        return mul_all(g ,div(1,x1))
    if z == 'sin':
        return g * np.cos(z_i)
    if z == 'cos':
        return g * (-np.sin(z_i))
    if z == 'log':
        return div_all(g ,z_i*(z_i/abs(z_i+0.000000001)))
    if z == 'exp':
        return mul_all(g , exp_all(z_i))
    if z == 'sqrt':
        return div_all(g ,(2 * sqrt_all(z_i))) * (z_i/abs(z_i))

def grad_node(s,g,z1,z2,x):
    global z23
    z23 = 0
    if s in ['+','-', '*', '/']:
        z23 = E_grad(s,z2,g) ## With respect to z1, we have to pass in z2
    if s in ['sin', 'cos', 'log', 'exp','sqrt']:
        z23 = grad(s,g,z1,x) ## Derivative with respect to z1, you have to pass in z1
    return z23

def grad_node2(s,g,z1,z2,x): ## x is not just x here
    global z21
    z21 = 0
    if s in ['+', '-', '*', '/']:
        z21 = E_grad_2(s,z1,z2,g) ## Take the derivative of z2

    if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
        z21 = grad(s,g,z2,x) ## Take the derivative of z2
    return z21