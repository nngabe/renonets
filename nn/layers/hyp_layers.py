from typing import Any, Optional, Sequence, Tuple, Union, Callable
import math

import manifolds

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import jax.tree_util as tree

import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

prng_key = jax.random.PRNGKey(0)
act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu}


class DenseAtt(eqx.Module):
    
    mlp: eqx.nn.MLP
    heads: int
    
    def __init__(self, in_features, heads=1, alpha=0.2):
        super(DenseAtt, self).__init__()
        act = lambda x: jax.nn.leaky_relu(x,alpha)
        self.mlp = eqx.nn.MLP( 2 * in_features, heads, width_size=64, depth=2, activation=act, final_activation=act, key = prng_key)
        self.heads = heads

    def __call__(self, x):
        n = x.shape[0]
        i,j = jnp.where(jnp.ones((n,n)))

        x_cat = jnp.concatenate((x[i], x[j]),axis=1)
        attn = jax.vmap(self.mlp)(x_cat).reshape(self.heads,n,n)
        attn = jax.nn.softmax(attn)
        return att_adj


class HypLinear(eqx.Module):

    dropout: eqx.nn.Dropout
    manifold: manifolds.base.Manifold
    c: float
    bias: jax.numpy.ndarray
    weight: jax.numpy.ndarray

    def __init__(self, key, manifold, in_features, out_features, c, p, use_bias):
        super(HypLinear, self).__init__()
        self.dropout = dropout(p) 
        self.manifold = manifold
        self.c = c
        lin = eqx.nn.Linear(in_features, out_features, key=key)
        self.bias = lin.bias
        self.weight = lin.weight 
        self.reset_parameters()

    def reset_parameters(self):
        init_weights = jax.nn.initializers.glorot_uniform()
        self.weight = init_weights(prng_key,self.weight.shape)

    def __call__(self, x, key=prng_key):
        drop_weight = self.dropout(self.weight, key=prng_key)  
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        #if self.use_bias:
        bias = self.manifold.proj_tan0(self.bias.reshape(1, -1), self.c)
        hyp_bias = self.manifold.expmap0(bias, self.c)
        hyp_bias = self.manifold.proj(hyp_bias, self.c)
        res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
        res = self.manifold.proj(res, self.c)
        return res


class HypAgg(eqx.Module):

    manifold: manifolds.base.Manifold
    c: float
    p: float
    use_att: bool
    attn: DenseAtt

    def __init__(self, manifold, c, in_features, p, use_att, heads=1):
        super(HypAgg, self).__init__()
        self.p = p
        self.manifold = manifold
        self.c = c
        self.p = p
        self.use_att = use_att
        self.attn = DenseAtt(in_features, heads=heads) if use_att else None

    def __call__(self, x, adj, w=None):
        s,r = adj[0],adj[1]
        n = x.shape[0]
        x = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            attn = self.attn(x)
            x_agg = jnp.einsum('kij,jl -> il', attn, x) 
        else:
            x_s = x[s]
            if isinstance(w,jnp.ndarray): x_s = jnp.einsum('ij,i -> ij', x_s, w)
            x_agg = jax.ops.segment_sum(x_s, r, n)
            #x_agg = tree.tree_map(lambda x: jax.ops.segment_sum(x[s], r, n), x_tangent)
        x_agg = self.manifold.proj(self.manifold.expmap0(x_agg, c=self.c), c=self.c)
        return x_agg


class HypAct(eqx.Module):

    manifold: manifolds.base.Manifold
    c_in: float
    c_out: float
    act: Callable

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def __call__(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)


class HNNLayer(eqx.Module):
    
    linear: HypLinear
    hyp_act: HypAct

    def __init__(self, manifold, in_features, out_features, c=1., p=0., act=jax.nn.silu, use_bias=True):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, p, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def __call__(self, x):
        h = self.linear(x)
        h = self.hyp_act(h)
        return h


class HGCNLayer(eqx.Module):

    linear: HypLinear
    agg: HypAgg
    hyp_act: HypAct

    def __init__(self, key, manifold, in_features, out_features, c_in, c_out, p, act, use_bias, use_att):
        super(HGCNLayer, self).__init__()
        self.linear = HypLinear(key, manifold, in_features, out_features, c_in, p, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, p, use_att)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def __call__(self, x, adj, w=None):
        h = self.linear(x)
        h = self.agg(h, adj, w)
        h = self.hyp_act(h)
        output = h, adj
        return output

