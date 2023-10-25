"""Hyperboloid manifold."""

import jax
import jax.numpy as jnp

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """
    name: str
    eps: float 
    min_norm: float
    max_norm: float

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = 1e-15
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = jnp.sum(x * y, axis=-1) - 2 * x[:, 0] * y[:, 0]
        if keepdim:
            res = res.reshape(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return jnp.sqrt(jnp.clip(dot, a_min=self.eps))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = jnp.clip(-prod / K, a_min=1.0 + self.eps)
        sqdist = K * arcosh(theta) ** 2
        return jnp.clip(sqdist, None, 50.0)

    def proj(self, x, c):
        K = 1. / c
        y = x[:,1:]
        y_sqnorm = jnp.linalg.norm(y, ord=2, axis=1, keepdims=True) ** 2 
        mask = jnp.ones_like(x)
        mask = mask.at[:, 0].set(0)
        vals = jnp.zeros_like(x)
        vals = vals.at[:, 0:1].set(jnp.sqrt(jnp.clip(K + y_sqnorm, a_min=self.eps)))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        ux = jnp.sum(x[:,1:] * u[:,1:], axis=1, keepdims=True)
        mask = jnp.ones_like(u)
        mask = mask.at[:, 0].set(0)
        vals = jnp.zeros_like(u)
        vals.at[:, 0:1].set(ux / jnp.clip(x[:, 0:1], a_min=self.eps))
        return vals + mask * u

    def proj_tan0(self, u, c):
        if jnp.ndim(u)==1: u = u.reshape(1,-1)
        narrowed = u[:, :1]
        vals = jnp.zeros_like(u)
        vals = vals.at[:, :1].set(narrowed)
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = jnp.clip(normu,a_max=self.max_norm)
        theta = normu / sqrtK
        theta = jnp.clip(theta,self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = jnp.clip(self.minkowski_dot(x, y) + K, a_max=-self.eps) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = jnp.clip(normu, a_min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        if jnp.ndim(u)==1: u = u.reshape(1,-1)
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.shape[-1] - 1
        x = u[:, 1:]#.reshape(-1, d)
        x_norm = jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = jnp.clip(x_norm, a_min=self.min_norm)
        theta = x_norm / sqrtK
        res = jnp.ones_like(u)
        res = res.at[:, 0:1].set( sqrtK * cosh(theta) )
        res = res.at[:, 1:].set( sqrtK * sinh(theta) * x / x_norm )
        return self.proj(res, c)

    def logmap0(self, x, c):
        if jnp.ndim(x)==1: x = x.reshape(1,-1)
        K = 1. / c
        sqrtK = K ** 0.5
        y = x[:,1:]
        y_norm = jnp.linalg.norm(y, ord=2, axis=1, keepdims=True)
        y_norm = jnp.clip(y_norm, a_min=self.min_norm)
        res = jnp.zeros_like(x)
        theta = jnp.clip(x[:, 0:1] / sqrtK, a_min=1.0 + self.eps)
        res = res.at[:, 1:].set(sqrtK * arcosh(theta) * y / y_norm)
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.T
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = jnp.clip(self.sqdist(x, y, c), a_min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x[:,:1]
        y = x[:,1:]
        y_norm = jnp.clip(jnp.linalg.norm(y, ord=2, axis=1, keepdims=True), a_min=self.min_norm)
        y_normalized = y / y_norm
        v = jnp.ones_like(x)
        v = v.at[:, 0:1].set( - y_norm )
        v = v.at[:, 1:].set( (sqrtK - x0) * y_normalized )
        alpha = jnp.sum(y_normalized * u[:, 1:], axis=1, keepdims=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        return sqrtK * x[:,1:] / (x[:, 0:1] + sqrtK)

