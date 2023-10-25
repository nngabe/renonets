"""Math utils functions."""

import jax.numpy as jnp

def cosh(x, clamp=15):
    return jnp.cosh(x.clip(-clamp, clamp))


def sinh(x, clamp=15):
    return jnp.sinh(x.clip(-clamp, clamp))


def tanh(x, clamp=15):
    return jnp.tanh(x.clip(-clamp, clamp))


def arcosh(x):
    clamp = 1.
    return jnp.arccosh(x.clip(clamp))

def arsinh(x):
    return jnp.arcsinh(x)

def artanh(x):
    clamp = 1. - 1e-7
    return jnp.arctanh(x.clip(clamp,clamp))
