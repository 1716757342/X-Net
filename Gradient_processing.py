from numpy import *

def grad_clip(x,pwd = 0):
    if pwd == 1:
        if x>0:
            aa =  np.min([x+(1e-40),10e1])
        else:
            aa =  np.max([x-(1e-40),-10e1])
        # if np.abs(aa) <= 0.1:
        #     aa = aa/abs(aa)
    if pwd == 0:
        aa = x
    return aa

def grad_clip_s(x,pwd = 1):
    if pwd == 1:
        if x > 0:
            aa = np.min([x+(1e-40), 0.1])
        else:
            aa =  np.max([x-(1e-40), -0.1])
        # aa = np.tanh(x) * 2
        if np.abs(aa) <= 0.1:
            aa = aa/abs(aa)
    if pwd == 0:
        aa = x
    return aa
