"""Euclidean manifold."""

from manifolds.base import Manifold
import jax

prng_key = jax.random.PRNGKey(0)

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """
    name: str
    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        p = p/jnp.linalg.norm(a, ord=2, axis=1, keepdims=True)
        return p

    def sqdist(self, p1, p2, c):
        return jnp.square(p1 - p2).sum(axis=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = jax.numpy.einsum('ij,kj -> ik', x, m)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        init_fn = jax.nn.initializers.glorot_uniform()
        w = init_fn(prng_key,w.shape)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(axis=-1, keepdims=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
