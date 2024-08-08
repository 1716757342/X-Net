import numpy as np
from numpy import *

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def change(a1,s,x, id = 10):
    if a1.val[0] in ['+', '-', '*', '/']:
        if s in ['+', '-', '*', '/']:
            # print('a1.val[0]',a1.val[0])
            a1.val[0] = s
        elif s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            sssss = 0
            a1.val[0] = s
            if id == 1:
                a1.left = TreeNode(['x0', x[0], 1.0, 0.])
                a1.right = TreeNode(['0', 0., 0.0, 0.])
            elif id == 2:
                # a1.left = TreeNode(['x', x])
                a1.right = TreeNode(['0', 0., 0., 0.])
            elif id == 3:
                a1.left = TreeNode(['0', 0., 0., 0.])
                # a1.right = TreeNode(['0', 0])
        elif s in ['0']:
            a1.val[0] = s
            a1.left = None
            a1.right = None
        elif 'x' in s and 'e' not in s:
            a1.val[0] = s
            a1.left = None
            a1.right = None
    elif a1.val[0] in ['sin', 'cos', 'log', 'exp', 'sqrt']:
        if s in ['+', '-', '*', '/']:
            a1.val[0] = s
            if a1.right.val[0] == '0':
                a1.right = TreeNode(['x0',x[0],1.0, 0.])
            if a1.left.val[0] == '0':
                a1.left = TreeNode(['x0',x[0],1.0, 0.])
            sssss = 0
        elif s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            a1.val[0] = s
        elif s in ['0']:
            a1.val[0] = s
            a1.left = None
            a1.right = None
        elif 'x' in s and 'e' not in s:
            a1.val[0] = s
            a1.left = None
            a1.right = None
    elif a1.val[0] in ['0']:
        if s in ['+', '-', '*', '/']:
            # print('a1.val[0]',a1.val[0])
            a1.val[0] = s
            a1.right = TreeNode(['x0', x[0], 1.0, 0.])
            a1.left = TreeNode(['x0', x[0], 1.0, 0.])
        elif s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            sssss = 0
            a1.val[0] = s
            a1.left = TreeNode(['x0',x[0],1.0, 0.])
            a1.right = TreeNode(['0',0.,0.0, 0.])
        elif s in ['0']:
            a1.val[0] = s
            a1.left = None
            a1.right = None
        elif 'x' in s and 'e' not in s:
            a1.val[0] = s
            a1.left = None
            a1.right = None
    elif 'x' in a1.val[0]:
        if s in ['+', '-', '*', '/']:
            # print('a1.val[0]',a1.val[0])
            a1.val[0] = s
            a1.right = TreeNode(['x0', x[0], 1.0, 0.])
            #### %%%%%%
            a1.left = TreeNode(['x0', x[0], 1.0, 0.])
        elif s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            sssss = 0
            a1.val[0] = s
            a1.left = TreeNode(['x0',x[0],1.0, 0.])
            a1.right = TreeNode(['0',0.,0.0, 0.])
        elif s in ['0']:
            a1.val[0] = s
            a1.left = None
            a1.right = None
        elif 'x' in s and 'e' not in s:
            a1.val[0] = s
            a1.left = None
            a1.right = None
    return a1
    # a[1].val = S[index1]
