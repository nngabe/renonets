"""Math utils functions."""

import jax.numpy as jnp

def cosh(x, clamp=15):
    return jnp.cosh(x.clip(-clamp, clamp))


def sinh(x, clamp=15):
    return jnp.sinh(x.clip(-clamp, clamp))


def tanh(x, clamp=15):
    return jnp.tanh(x.clip(-clamp, clamp))


def arcosh(x):
    return jnp.arccosh(x)

def arsinh(x):
    return jnp.arcsinh(x)

def artanh(x):
    return jnp.arctanh(x)
