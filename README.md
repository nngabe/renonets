# RenOnet: Renormalized Operator networks for complex social systems

This repository contains an implementation of a multiscale operator learning framework for modelling and forecasting complex social systems. The framework learns multiscale dynamics of a complex system given an initial adjacency matrix $A^{(0)}$ and length $K$ history of the system. See figure below for details. [[slides](https://www.dropbox.com/scl/fi/2py8doe6gaqjwv9g6pcuw/Multiscale_operator_learning_for_social_dynamics.pdf?rlkey=1ljnspm5zjjvnc9mn66qfcvm6&dl=0)]

A brief overview of important modules in this repository are:

train.py - data loading, LR scheduling, graph sampling, and logging of training data.

nn/models/renonet.py - contains a module of the framework shown below, as well as vmapped and serial loss functions for optimizing the loss shown below.

nn/models/models.py - contains modules for the encoder and renormalization networks (GCN, HGCN) and decoder networks (MLP, Transformer, DeepOnet).


![RenONet_implementation.pdf](https://github.com/nngabe/renonet/files/13127626/RenONet_implementation.pdf)

