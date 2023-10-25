# RenOnet: Multiscale operator learning for complex social systems

This repository contains an implementation of a multiscale operator learning framework for modelling and forecasting complex social systems. The framework learns multiscale dynamics and forecasts the evolution of a complex system given an initial adjacency matrix $A^{(0)}$ and history of the system. See figure below for illustration and full slides for details. [[slides](https://www.dropbox.com/scl/fi/2py8doe6gaqjwv9g6pcuw/Multiscale_operator_learning_for_social_dynamics.pdf?rlkey=1ljnspm5zjjvnc9mn66qfcvm6&dl=0)]

A brief overview of important modules in this repository are:

**train.py**
* Data loading, LR scheduling, graph sampling, and logging of training data.

**nn/models/renonet.py** 
* Contains a module of the framework shown below, as well as vmapped and serial loss functions for optimizing the loss shown below.

**nn/models/models.py** 
- Modules for the encoder and renormalization networks (GCN, HGCN) and decoder networks (MLP, Transformer, DeepOnet).

**lib/graph_utils.py**
- Utilities for sampling, padding, and otherwise manipulating graphs.

**lib/positional_encoding.py**
- Functions for computing positional encoding (node2vec, random walk PE, and laplacian eigenvector PE).

**nn/manifolds/**
- Manifold definitions for hyperbolic layers. Ported from the original pytorch code ([HGCN](https://github.com/HazyResearch/hgcn)) to JAX.
- Includes Euclidean, Poincaré, and Hyperboloid manifolds.

<img width="903" alt="renonet" src="https://github.com/nngabe/renonet/assets/50005216/012602fe-19f1-4ac4-a540-04fde74a3b40">


