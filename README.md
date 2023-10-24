# RenOnet: Renormalized Operator networks for complex social systems

This repository contains an implementation of a multiscale operator learning framework for modelling and forecasting complex social systems. The framework learns multiscale dynamics of a complex system given an initial adjacency matrix $A^{(0)}$ and length $K$ history of the system $\mathbf{s} _t^{0}`\{s_{t-k}^{(0)}, \ k=1,\ldots,K \}`$  at various training times $t_i$. The model learns governing dynamics $\tilde{\mathbf{u}}^{(l)}_{t+\Delta t}(t,\mathbf{y}) = \mathcal{G}(v)(t,\mathbf{y};\zb_{t}^{(l)})$

A brief overview of important modules in this repository are:

train.py - data loading, LR scheduling, graph sampling, and logging of training data.

nn/models/renonet.py - contains a module of the framework shown below, as well as vmapped and serial loss functions for optimizing the loss shown below.

nn/models/models.py - contains modules for the encoder and renormalization networks (GCN, HGCN) and decoder networks (MLP, Transformer, DeepOnet).


[RenONet_implementation.pdf](https://github.com/nngabe/renonet/files/13127626/RenONet_implementation.pdf)

