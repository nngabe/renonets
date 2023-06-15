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
    g: eqx.Module
    x_dim: int

    def __init__(self, args):
        super(neural_burgers, self).__init__(args)
        self.F = getattr(models, args.decoder)(args)
        self.g = getattr(models, args.decoder if not args.g else args.g)(args)
        self.x_dim = args.enc_dims[-1]

    def N(self, t):
        return t*3./4. + 2.
    
    @eqx.filter_jit
    def res(self, u, x, grad_t, grad_x):
        u = u.reshape(-1,1)
        x = jnp.concatenate([x,u], axis=1) 
        F = self.F(x)
        g = -jax.nn.relu(self.g(x))

        f_0 = grad_t.reshape(-1,1)
        f_1 = F * (  u * grad_x ).sum(1).reshape(-1,1)
        f_2 = g
        w_0 = 1e+1
        return jnp.array([f_0 + f_1 + f_2, w_0 * jax.nn.leaky_relu(-f_0, 1e-2/w_0)])
