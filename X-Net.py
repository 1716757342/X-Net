import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from Forward_function import *
from Gradient_processing import *
from choice_symblic import *
from Backpropagation import *
import sympy as sp
n_node = 2 ** 4 -1

# global pwd
# pwd = 1

global L,D
L = []
D = []
B = []
def pretravale(treenode):

    if treenode==None:
        return None
    # print('treenode.val',treenode.val)
    L.append(treenode.val[0])
    D.append(treenode.val[2])
    B.append(treenode.val[3])
    pretravale(treenode.left)
    pretravale(treenode.right)
    return L, D, B

def round(x):
    return np.round(x * 10) / 10

def r2(EEe,yy_1):
    return 1 - (np.sum((yy_1 - EEe) ** 2)) / (np.sum((yy_1 - np.mean(EEe)) ** 2))

def translate(lt):
    stack_tr = []
    for i in range(len(lt)):
        s = lt[-(i + 1)]
        if s == 'x':
            stack_tr.append('x')
        if s == '0':
            nnnn = 0
        if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
            if s == 'exp':
                v = 'exp(' + stack_tr.pop() + ')'
            if s == 'log':
                v = 'log(' + stack_tr.pop() + ')'
            if s == 'cos':
                v = 'cos(' + stack_tr.pop() + ')'
            if s == 'sin':
                v = 'sin(' + stack_tr.pop() + ')'
            if s == 'sqrt':
                v = 'sqrt(' + stack_tr.pop() + ')'
            stack_tr.append(v)
        if s in ['+', '-', '*', '/']:
            if s == '+':
                v = '(' + stack_tr.pop() + '+' + stack_tr.pop() + ')'
            if s == '-':
                v = stack_tr.pop() + '-' + stack_tr.pop()
            if s == '*':
                v = '(' + stack_tr.pop() + ') * (' + stack_tr.pop()+ ')'
            if s == '/':
                v = '(' + stack_tr.pop() + ') / (' + stack_tr.pop()+ ')'
            stack_tr.append(v)
    return stack_tr


loss_mean = []
X = []
num = 600
batch = 6
dim = 4
X = np.random.rand(dim,num) * 5
# Variables are defined using loops and dynamic variable name generation
for i in range(dim):
    globals()[f'x_{i}'] = X[i]


# S = ['+','-','*','sin','cos','exp','log','sqrt']
S = ['+','-','*','sin','cos']
S_G = S.copy()
S_n = ['+','-','*','/']
for i in range(len(X)):
    S.append('x'+str(i))
print(S)
# s = 'x1'
# print('int',int(s[1]) + 1)

y_1 = 0.0
y_1 = sin(x_0) + cos(x_0)

threshold = 0.9999
Y = y_1.copy()


#### Adding disturbance noise
# bound = (max(y_1) - min(y_1))/20
# for k in range(len(y_1)):
#     y_1[k] += np.random.uniform(-bound,bound)

def Arity(s):

    if 'x' in s and 'e' not in s:
        return 0
    if s in ['sin', 'cos', 'exp', 'ln', 'sqrt']:
        return 1
    if s in ['+', '-', '*', '/', '^']:
        return 2
def G_expression(coun):
    counter = 1
    Ex = []
    while counter < coun:
        s = S_G[int(np.random.uniform(0,len(S_G)))]
        if s in ['+','-','*']:
            Ex.append(s)
        else:
            Ex.append(s)
            Ex.append('0')
        counter = counter + Arity(s) - 1
    ci = 0
    while counter != 0:
    # for i in range(X.shape[0]):
        ci = np.random.uniform(0, X.shape[0])
        s = 'x'+str(ci)
        Ex.append(s)
        counter = counter + Arity(s) - 1
        # ci = ci + 1
    return Ex
#### Initializes the network structure
l = ['*','*','x0','x0','sin','x0','0']

d = np.ones(len(l))
b = np.zeros(len(l))


iters = 0
ite = 400
ite_s = 100

b_best = b.copy()
d_best = d.copy()
l_best = l.copy()
r_best = -100
loss_best = 100
W_best = np.random.randn(len(l) * 2)

step_size = 0.1
step_size_c = 0.02
step_size_b = 0.02
mm = 0
time_start = time.time()

global key
key = 0
v_C = 0
v_b = 0
W = np.random.randn(len(l) * 2)
for w in range(ite):

    z_1_all = []
    z_2_all = []
    t = 1
    loss_all = []
    l = deepcopy(l_best)
    d = deepcopy(d_best)
    b = deepcopy(b_best)
    for n in range(num):
        cir = 1
        # loss_all = []
        l = deepcopy(l_best)
        d = deepcopy(d_best)
        b = deepcopy(b_best)

        for h in range((cir + 1) * 10 ):
            # print(l)
            # if key % cir != 0:
            iters += 1

            #### Do you need constants? ####

            # if iters % 1000 == 0:
            #     W = np.hstack((d_best,b_best))
            #     print(W.shape)
            #     for bf in range(30):
            #         W = np.random.randn(len(W))
            #         fun = lambda W : np.mean(0.5 * (y_1 - all_farward_c(l_best,X,W))**2)
            #         # res = minimize(fun, W, method="BFGS")
            #         res = minimize(fun, W, method="l-bfgs-b")
            #
            #         res = np.array(res.x)
            #         d_b = np.array(res[0:int(len(W) / 2)])
            #         b_b = np.array(res[int(len(W) / 2):len(W)])
            #         X_b = X.copy()
            #         # X.sort()
            #         E_best_b = all_farward(l_best, X_b, d_b, b_b)
            #         r_E_r_b = r2(E_best_b, Y)
            #         if r_best < r_E_r_b and 1 >= r_E_r_b >= 0:
            #             r_best = r_E_r_b.copy()
            #             l_best = deepcopy(l_best)
            #             d_best = deepcopy(d_b)
            #             b_best = deepcopy(b_b)
            #             W_best = deepcopy(W)
            #             print('R' * 100)
            #             print('l_best',l_best)
            #             print('r_best',r_best)

            a = list(range(len(l)))
            a_v = list(range(len(a)))
            a_v_i = list(range(len(a)))
            random_indices = np.random.choice(X.shape[1], batch, replace=False)
            x = X[:,random_indices].copy()
            a = list(range(len(l)))
            ac = list(range(len(l)))
            stack_t = []
            stack = []
            for i in range(len(l)): #### Computes the preorder traversal
                s = l[-(i + 1)]
                c = d[-(i + 1)]
                bb = b[-(i + 1)]
                # if s == 'x':
                if 'x' in s and 'e' not in s:
                    # print('xxxx',x,s[1])
                    a_v[-(i + 1)] = x[int(s[1])]
                    stack.append(x[int(s[1])] * c + bb)
                    a_v_i[-(i + 1)] = x[int(s[1])] * c + bb

                if s == '0':
                    a_v_i[-(i + 1)] = 0
                    a_v[-(i + 1)] = 0
                if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
                    if s == 'exp':
                        v = exp_all(stack.pop())
                    if s == 'log':
                        v = log_all(stack.pop())
                    if s == 'cos':
                        v = np.cos(stack.pop())
                    if s == 'sin':
                        v = np.sin(stack.pop())
                    if s == 'sqrt':
                        v = sqrt_all(stack.pop())

                    a_v[-(i + 1)] = v
                    stack.append(v * c + bb)
                    a_v_i[-(i + 1)] = v * c + bb

                if s in ['+', '-', '*', '/']:
                    if s == '+':
                        v = stack.pop() + stack.pop()
                    if s == '-':
                        v = stack.pop() - stack.pop()
                    if s == '*':
                        v = mul_all(stack.pop() , stack.pop())
                    if s == '/':
                        v = div_all(stack.pop() , stack.pop())

                    a_v[-(i + 1)] = v
                    stack.append(v * c + bb)
                    a_v_i[-(i + 1)] = v * c + bb

            ###### Building binary trees The third parameter of each tree is a constant. The second value of the node is the value that is not multiplied by a constant
            for i in range(len(l)):
                st = l[-(i + 1)]
                # if st == 'x':
                if 'x' in st and 'e' not in st:
                    a[-(i + 1)] = TreeNode([st ,a_v[-(i + 1)],d[-(i + 1)],b[-(i + 1)]])
                    stack_t.append(a[-(i + 1)])
                if st == '0':
                    a[-(i + 1)] = TreeNode(['0',a_v[-(i + 1)],d[-(i + 1)],b[-(i + 1)]])
                    stack_t.append(a[-(i + 1)])
                if st in ['sin', 'cos', 'log', 'exp', 'sqrt','+', '-', '*', '/']:
                    a[-(i + 1)]  = TreeNode([st,a_v[-(i + 1)],d[-(i + 1)],b[-(i + 1)]])
                    a[-(i + 1)].left = stack_t.pop()
                    a[-(i + 1)].right = stack_t.pop()
                    stack_t.append(a[-(i + 1)])

            E1 = np.array(a[0].val[1])
            E = np.array(a[0].val[1] * d[0] + b[0])

            loss = np.mean((y_1[n] - E) * (y_1[n] - E))/2
            loss_all.append(loss)

            ## Compute the best
            Xa = X.copy()
            E_best = all_farward(l, Xa, d, b)
            r_E = r2(E_best, y_1)
            r_E_r = r2(E_best, Y)
            if r_best < r_E_r and 1>=r_E_r>=0:
                r_best = r_E_r.copy()
                l_best = deepcopy(l)
                d_best = deepcopy(d)
                b_best = deepcopy(b)
                W_best = deepcopy(W)
                print('r_best', r_best)
                print('The best l:', l_best)

            if r_best >= threshold:

                print('Had found the best equ!',r_E_r)
                time_end = time.time()  #### It takes time to finish recording
                time_sum = time_end - time_start
                print('The time:',time_sum)
                break

            if len(loss_all) >= 2:
                step_size = np.tanh(np.exp(- (loss_all[-2] - loss_all[-1]))) / 0.1
            if step_size <= 0.00001 or np.isnan(step_size_c):
                step_size = 0.00001

            if len(loss_all) >= 2:
                step_size_c = np.tanh(np.exp(- (loss_all[-2] - loss_all[-1]))) / 1
            if step_size_c <= 0.00001 or np.isnan(step_size_c):
                step_size_c = 0.00001

            if len(loss_all) >= 2:
                step_size_b = np.tanh(np.exp(- (loss_all[-2] - loss_all[-1]))) / 1
            if step_size_b <= 0.00001 or np.isnan(step_size_b):
                step_size_b = 0.00001


            dE = grad_clip(E - y_1[n])
            # print('dE',dE)
            dE1 = grad_clip(dE * a[0].val[2])
            # print(dE * a[0].val[2])
            # print(dE1)
            dde = dE * a[0].val[1]

            ddb = dE
            # print(ddb)
            # if key%cir==0:
            # if iters % cir != 0:
            if iters % 10000 != 0:
                E1 = E1 - step_size * dE1

            a[0].val[1] = E1
            #### Backpropagate, and update.
            bpd(a[0], dE1, x, key, cir, step_size)
            #### Select the symbol of the primary connection node
            # if key % cir == 0:
            if iters % 10000 != 0:
                for q in range(len(a)):
                    choice_symblic(a[q],a,x[0],x,S,n_node)
            ### Disturbances are introduced to avoid falling into local optima.
            # if key%cir == 0:
            if iters % 10000 != 0:
                if len(loss_all)>=2:
                    # print(loss_all[-2] - loss_all[-1])
                    if (loss_all[-2] - loss_all[-1]) <= 0 or np.isnan((loss_all[-2] - loss_all[-1])):
                        mm += 1
                        if mm % 2 == 0:
                            for j in range(1):
                                fu1 = np.random.choice(list(range(0, len(S))))
                                zhi = np.random.choice(list(range(0,len(a))))
                                a[zhi] = change(a[zhi],S[fu1],x)

                                # fu2 = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                                # fu2 = np.random.choice([0, 1, 2, 3, 4, 5])
                                # for t in range(len(a)):
                                #     if a[t].val[0] == 'x':
                                #         zhi2 = np.random.choice([0,0,0,0,1])
                                #         # print('zhi2:',zhi2)
                                #         if zhi2 == 1 and len(a) <= n_node:
                                #             a[t] = change(a[t],S[fu2],x)
            if r_E_r >= threshold:
                print('Had found the best equ!')
                break
            l = []
            d = []
            b = []
            l,d,b = pretravale(a[0])
            L = []
            D = []
            B = []
        loss_mean.append(np.mean(loss_all))
        if r_best >= threshold:
            print('Had found the best equ!')
            break
    if r_best >= threshold:
        print('Had found the best equ!')
        break
plt.show()
time_end = time.time()
print('Time',time_end-time_start)
l = deepcopy(l_best)
d = deepcopy(d_best)
b = deepcopy(b_best)
stack = []
x = X.copy()
print("The End : ",l)
E = all_farward(l,x,d,b)
print('The last R2 : ',r2(E,Y))
# print('The last one: ',translate(l))









