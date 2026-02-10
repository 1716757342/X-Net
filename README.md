# 🧠 A Novel Paradigm for Neural Computation: X-Net with Learnable Neurons and Adaptable Structure

This repository provides the official implementation of **X-Net**, a novel neural network paradigm featuring **learnable neuron-wise activation functions** and **adaptive architectures**.

In addition to the source code, we include a dynamic training animation to illustrate how X-Net evolves during optimization.

📄 Paper: https://arxiv.org/abs/2401.01772  

---

![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

<br>

<p align="center">
  <img src="https://github.com/1716757342/X-Net/blob/main/X-Net.gif" width="600">
</p>

---

## 📌 Overview

Multilayer perceptrons (MLPs) are widely used across domains such as bioinformatics and financial analytics, and have become a core tool in contemporary scientific research. However, standard MLPs suffer from two key limitations. First, they typically employ a single, fixed activation function throughout the network, which restricts their representational capacity and often requires unnecessarily deep or wide architectures to solve even relatively simple tasks. Second, the network architecture itself is not adaptive, making it prone to structural redundancy or, conversely, insufficient capacity. In this work, we introduce X-Net, a new neural network paradigm that aims to replace conventional MLPs. X-Net can dynamically learn neuron-wise activation functions from derivative information during training, thereby improving task-specific representational capacity. In addition, X-Net can adapt its architecture at the level of individual neurons, enabling it to match the complexity of the target task while reducing computational cost. We show that X-Net consistently surpasses MLPs in representational power. On both regression and classification benchmarks, X-Net attains comparable or superior predictive performance with far fewer parameters: on average, X-Net uses only 3 $\%$ of the parameters of an MLP. We further demonstrate the potential of X-Net for scientific discovery across a broad range of scientific tasks, including data from energy, environment, aerospace, chaotic dynamical systems, and partial differential equations, where X-Net helps researchers uncover new mathematical or physical relationships.

---

## 🚀 Getting Started

Before running X-Net, please configure the following hyperparameters in `X-Net.py`:

```python
num = 600      # Number of sampling points
batch = 6      # Batch size
dim = 1        # Variable dimension

S = ['+','-','*','sin','cos']   # Candidate activation function library

l = ['*','*','x0','x0','sin','x0','0']  # Initial network structure

ite = 400      # Training epochs
```

# Finally, provide input X and output Y

---

## ▶️ Run

```python X-Net.py```

Example Output

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
## 📖 Citation

If you find this work useful, please cite:

@article{li2024novel,
  title={A Novel Paradigm for Neural Computation: X-Net with Learnable Neurons and Adaptable Structure},
  author={Li, Yanjie and Li, Weijun and Yu, Lina and Wu, Min and Liu, Jinyi and Li, Wenqiang and Hao, Meilan},
  journal={arXiv preprint arXiv:2401.01772},
  year={2024}
}
