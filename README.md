# A Novel Paradigm for Neural Computation: X-Net with Learnable Neurons and Adaptable Structure

This repository provides a basic implementation version of X-Net. 'The sparrow may be small but it has all the vital organs'. This repository contains the core components and implementation details of X-Net. I hope you can appreciate the core idea of X-Net through this repository. And then help everyone to carry out further work on the basis of X-Net.
[Paper](https://arxiv.org/abs/2401.01772)&nbsp;&nbsp;&nbsp;

<br>

![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

<br>

## Getting started

Prespecified hyperparameters in X-Net.py are required.

```
num = 600 ## Number of sampling points

batch = 6 ## batch size

dim = 1 ## Variable dimension

S = ['+','-','*','sin','cos'] ## Activation function candidate library

l = ['*','*','x0','x0','sin','x0','0'] #### Initializing the network
ite = 400 ## Train epoch

Finally， given the input X and the output Y

```

## Run code
```
python X-Net.py
```

Output:

```
The best l: ['+', 'cos', 'x0', '0', 'sin', 'x0', '0']
Had found the best equ! 1.0
The time: 0.43471407890319824
Had found the best equ!
Had found the best equ!
Time 0.7416861057281494
The End :  ['+', 'cos', 'x0', '0', 'sin', 'x0', '0']
The last R2 :  1.0
```

## Citation

If you found our work useful, please use the following citation:

```
@article{li2024novel,
  title={A Novel Paradigm for Neural Computation: X-Net with Learnable Neurons and Adaptable Structure},
  author={Li, Yanjie and Li, Weijun and Yu, Lina and Wu, Min and Liu, Jinyi and Li, Wenqiang and Hao, Meilan},
  journal={arXiv preprint arXiv:2401.01772},
  year={2024}
}
```
