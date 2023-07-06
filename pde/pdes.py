import sys

from nn.models import models

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng_key = jax.random.PRNGKey(0)

class PDE(eqx.Module):

    def __init__(self, args):
        super(PDE, self).__init__()
        

class neural_burgers(PDE):
    F: eqx.Module
    v: eqx.Module
    x_dim: int

    def __init__(self, args):
        super(neural_burgers, self).__init__(args)
        self.F = getattr(models, args.decoder)(args)
        self.v = getattr(models, args.decoder)(args)
        self.x_dim = args.enc_dims[-1]

