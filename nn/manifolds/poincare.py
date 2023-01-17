"""Poincare ball manifold."""

import jax
import jax.numpy as jnp

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """
    name: str
    min_norm: float
    eps: dict

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-7
        self.eps = 1e-5

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * jnp.linalg.norm(self.mobius_add(-p1, p2, c, dim=-1), axis=-1, ord=2, keepdims=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = jnp.sum(x*x, axis=-1, keepdims=True)
        return 2 / jnp.clip( 1. - c * x_sqnorm , a_min=self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p**2
        return dp

    def proj(self, x, c):
        norm = jnp.clip( jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True), self.min_norm)
        maxnorm = (1 - self.eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return jnp.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = jnp.clip(jnp.linalg.norm(u, axis=-1, ord=2, keepdims=True), a_min=self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = jnp.clip(jnp.linalg.norm(sub,axis=-1, ord=2, keepdims=True), self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = jnp.clip(jnp.linalg.norm(u, axis=-1, ord=2, keepdims=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = jnp.clip(jnp.linalg.norm(p, axis=-1, ord=2, keepdims=True), a_min=self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = (x*x).sum(axis=dim, keepdims=True)
        y2 = (y*y).sum(axis=dim, keepdims=True)
        xy = (x * y).sum(axis=dim, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / jnp.clip(denom,self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = jnp.clip(jnp.linalg.norm(x, axis=-1, keepdims=True, ord=2), a_min=self.min_norm)
        mx = x @ m.T
        mx_norm = jnp.clip(jnp.linalg.norm(mx, axis=-1, keepdims=True, ord=2), a_min=self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdims=True, dtype=jnp.uint8)
        res_0 = jnp.zeros(1, dtype=res_c.dtype)
        res = jnp.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        init_fn = jax.nn.initializers.glorot_uniform()
        w = init_fn(prng_key,w.shape)*irange
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = (u * u).sum(axis=dim, keepdims=True)
        v2 = (v * v).sum(axis=dim, keepdims=True)
        uv = (u * v).sum(axis=dim, keepdims=True)
        uw = (u * w).sum(axis=dim, keepdims=True)
        vw = (v * w).sum(axis=dim, keepdims=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / jnp.clip(d, a_min=self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(axis=-1, keepdims=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / jnp.clip(lambda_x, a_min=self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = jnp.linalg.norm(x, p=2, axis=1, keepdims=True) ** 2
        return sqrtK * jnp.concatenate([K + sqnorm, 2 * sqrtK * x], axis=1) / (K - sqnorm)

