import numpy as np
from numpy import *
from numpy import *
from copy import deepcopy
from Get_gradient import *
from Gradient_processing import *


def bpd(aa, de, x, key, cir,step_size):
    global d_k_l,d_k_r,ssss
    d_k_l = 0
    d_k_r = 0
    # print('aa',aa.val)
    if aa.left == None and aa.right == None:

        return
    elif aa.left.val[0] != '0' or aa.right.val[0] != '0':
        if aa.left.val[0] != '0' and aa.right.val[0] == '0': ## The derivative of only the left node, sin, cos...
            dE_l = grad_node(aa.val[0], de, aa.left.val[1] * aa.left.val[2] + aa.left.val[3],aa.right.val[1] * aa.right.val[2] + aa.right.val[3], x)
            # print('d_k_l',d_k_l,aa.left.val)
            d_k_l = grad_clip(dE_l * aa.left.val[2])
            dd_l = grad_clip(dE_l * aa.left.val[1])
            db_l = grad_clip(dE_l)
            if key%cir == 0:
                aa.left.val[1] = aa.left.val[1] - step_size * grad_clip(d_k_l)

            # if key%cir != 0:
            #     aa.left.val[2] = aa.left.val[2] - step_size_c * grad_clip_s(dd_l)
            #     aa.left.val[3] = aa.left.val[3] - step_size_b * grad_clip_s(db_l)
            #     print('step_size_c * grad_clip_s(dd_l)', step_size_c * grad_clip_s(dd_l))

            # d_k_r = 0
        elif aa.left.val[0] == '0' and aa.right.val[0] != '0':## The derivative with respect to only the right node
            # d_k_l = 0

            dE_r = grad_node(aa.val[0], de, aa.right.val[1] * aa.right.val[2] + aa.right.val[3] ,aa.left.val[1] * aa.left.val[2] + aa.left.val[3], x)
            d_k_r = grad_clip(dE_r * aa.right.val[2])
            dd_r = grad_clip(dE_r * aa.right.val[1])
            db_r = grad_clip(dE_r)
            if key%cir == 0:
                aa.right.val[1] = aa.right.val[1] - step_size * grad_clip(d_k_r)
            # if key%cir != 0:
            #     aa.right.val[2] = aa.right.val[2] - step_size_c * grad_clip_s(dd_r)
            #     aa.right.val[3] = aa.right.val[3] - step_size_b * grad_clip_s(db_r)
            # #     print('step_size_c * grad_clip_s(dd_r)', step_size_c * grad_clip_s(dd_r))

        elif aa.left.val[0] != '0' and aa.right.val[0] != '0': ## Derivatives of binary operators, +-*/
            dE_l = grad_node(aa.val[0], de, aa.left.val[1] * aa.left.val[2] + aa.left.val[3], aa.right.val[1] * aa.right.val[2] + aa.right.val[3], x)
            d_k_l = grad_clip(dE_l * aa.left.val[2])
            dd_l = grad_clip(dE_l * aa.left.val[1])
            db_l = grad_clip(dE_l )
            if key%cir == 0:
                aa.left.val[1] = aa.left.val[1] - step_size * grad_clip(d_k_l)
            # if key%cir != 0:
            #     aa.left.val[2] = aa.left.val[2] - step_size_c * grad_clip_s(dd_l)
            #     aa.left.val[3] = aa.left.val[3] - step_size_b * grad_clip_s(db_l)
            # #     print('step_size_c * grad_clip_s(dd_l)', step_size_c * grad_clip_s(dd_l))

            dE_r = grad_node2(aa.val[0], de, aa.left.val[1] * aa.left.val[2] + aa.left.val[3], aa.right.val[1] * aa.right.val[2] + aa.right.val[3], x)
            d_k_r = grad_clip(dE_r * aa.right.val[2])
            dd_r = grad_clip(dE_r * aa.right.val[1])
            db_r = grad_clip(dE_r)
            if key%cir == 0:
                aa.right.val[1] = aa.right.val[1] - step_size * grad_clip(d_k_r)
            # if key%cir != 0:
            #     aa.right.val[2] = aa.right.val[2] - step_size_c * grad_clip_s(dd_r)
            #     aa.right.val[3] = aa.right.val[3] - step_size_b * grad_clip_s(db_r)
            # #     print('step_size_c * grad_clip_s(dd_r)', step_size_c * grad_clip_s(dd_r))

            # print('d',d_k_l,d_k_r)
    dkl = deepcopy(d_k_l)
    dkr = deepcopy(d_k_r)

    bpd(aa.left, dkl, x, key, cir,step_size)
    bpd(aa.right, dkr,x, key, cir,step_size)
