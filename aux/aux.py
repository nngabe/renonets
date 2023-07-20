import sys
from typing import List, Dict

from nn.models import models

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng_key = jax.random.PRNGKey(0)


class neural_burgers(eqx.Module):
    
    F: eqx.Module
    v: eqx.Module
    x_dim: int

    def __init__(self, args):
        super(neural_burgers, self).__init__()
        self.F = getattr(models, args.decoder)(args)
        self.v = getattr(models, args.decoder)(args)
        self.x_dim = args.enc_dims[-1]


class pooling(eqx.Module):
    
    pools: Dict[int,eqx.Module]
    L: int
    kappa: int
    
    def __init__(self, args):
        super(pooling, self).__init__()
        self.pools = {}
        self.L = args.pool_init
        self.kappa = args.kappa
        for i in range(self.L):
            self.pools[i] = getattr(models, args.pool)(args)

    def __getitem__(self, i):
        return self.pools[i]

    def keys(self):
        return self.pools.keys()
