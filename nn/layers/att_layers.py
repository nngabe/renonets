"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn

prng_key = jax.random.PRNGKey(0)

class DenseAtt(eqx.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, key=prng_key)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.shape[0]
        i,j = jnp.where(jnp.ones((n,n)))

        x_cat = jnp.concatenate((x[i], x[j]),axis=0)
        att_adj = jax.vmap(self.linear)(x_cat)
        att_adj = jax.nn.sigmoid(att_adj).reshape(n,n)  
        return att_adj

