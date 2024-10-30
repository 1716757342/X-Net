# A Novel Paradigm for Neural Computation: X-Net with Learnable Neurons and Adaptable Structure

This repository provides a basic implementation version of X-Net. 'The sparrow may be small but it has all the vital organs'. This repository contains the core components and implementation details of X-Net. I hope you can appreciate the core idea of X-Net through this repository. And then help everyone to carry out further work on the basis of X-Net.

[Paper](https://arxiv.org/abs/2401.01772)&nbsp;&nbsp;&nbsp;

## Instruction

Multilayer perception (MLP) has permeated various disciplinary domains, ranging from bioinformatics to financial analytics, where their application has become an indispensable facet of contemporary scientific research endeavors. However, MLP has obvious drawbacks. 1), The type of activation function is single and relatively fixed, which leads to poor `representation ability' of the network, and it is often to solve simple problems with complex networks; 2), the network structure is not adaptive, it is easy to cause network structure redundant or insufficient. In this work, we propose a novel neural network paradigm X-Net promising to replace MLPs. X-Net can dynamically learn activation functions individually based on derivative information during training to improve the network's representational ability for specific tasks. At the same time, X-Net can precisely adjust the network structure at the neuron level to accommodate tasks of varying complexity and reduce computational costs. We show that X-Net outperforms MLPs in terms of representational capability. X-Net can achieve comparable or even better performance than MLP with much smaller parameters on regression and classification tasks. Specifically, in terms of the number of parameters, X-Net is only 3% of MLP on average and only 1.1% under some tasks. We also demonstrate X-Net's ability to perform scientific discovery on data from various disciplines such as energy, environment, and aerospace, where X-Net is shown to help scientists discover new laws of mathematics or physics.
<br>

![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

<br>

![Alt text](https://github.com/1716757342/X-Net/blob/main/X-Net.gif)


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
