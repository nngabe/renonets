from typing import Any, Optional, Sequence, Tuple, Union, Callable
import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import jax.tree_util as tree

import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

import jraph

prng_key = jax.random.PRNGKey(0)
act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu}

def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """

    if args.enc_init:
        act = act_dict[args.act_enc] if args.act_enc in act_dict else jax.nn.silu
        args.num_layers = len(args.enc_dims)
        dims = args.enc_dims
        args.enc_init = 0
        args.skip = 0
    elif args.dec_init:
        act = act_dict[args.act_dec] if args.act_dec in act_dict else jax.nn.silu
        args.num_layers = len(args.dec_dims)
        dims = args.dec_dims
        args.dec_init = 0
    elif args.pde_init:
        act = act_dict[args.act_pde] if args.act_pde in act_dict else jax.nn.silu
        args.num_layers = len(args.pde_dims)
        dims = args.pde_dims
        args.pde_init -= 1
    else:
        print('All layers already init-ed! Define additional layers or reset args.')
        raise

    return dims, act


class Linear(eqx.Module): 
    p: float
    linear: eqx.nn.Linear
    weight: jnp.ndarray
    bias: jnp.ndarray
    act: Callable
    dropout: Callable
    
    def __init__(self, in_features, out_features, p=0., act=jax.nn.silu, use_bias=True):
        super(Linear, self).__init__()
        self.p = p # dropout prob
        self.linear = nn.Linear(in_features, out_features, use_bias, key=prng_key)
        self.weight = jnp.zeros((out_features,in_features))
        self.bias = 1e-7*jnp.ones((out_features,1))
        self.act = act
        self.dropout = lambda x: dropout(self.p)(x, inference=False, key=prng_key)

    def __call__(self, x):
        hidden = x @ self.weight.T
        hidden += self.bias
        hidden = dropout(self.p)(hidden, inference=False, key=prng_key)
        out = self.act(hidden)
        return out


class GraphConvolution(eqx.Module):
    """GCN layer with symmetric normalization"""
    p: float
    linear: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, p=0., act=jax.nn.silu, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.p = p
        self.linear = nn.Linear(in_features, out_features, use_bias, key=prng_key)
        self.act = act
        self.dropout = lambda x: dropout(self.p)(x, inference=False, key=prng_key)

    def __call__(self, x, adj):
        n = x.shape[0]
        s, r = adj[0], adj[1]
        count_edges = lambda x: jax.ops.segment_sum(jnp.ones_like(s), x, n)
        sender_degree = count_edges(s)
        receiver_degree = count_edges(r)    

        h = jax.vmap(self.linear)(x)
        h = jax.vmap(self.dropout)(h) 
        h = tree.tree_map(lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None], h)
        h = tree.tree_map(lambda x: jax.ops.segment_sum(x[s], r, n), h)
        h = tree.tree_map(lambda x: x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None], h)
        h = self.act(h)

        output = h, adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )



class GATConvolution(eqx.Module):
    """GAT  layer."""
    p: float
    linear: eqx.nn.Linear
    query_fn: eqx.nn.Linear
    a: eqx.nn.Linear
    W: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, dropout=0., act=jax.nn.silu, use_bias=True, num_heads=3, query_dim=8):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, query_dim * num_heads, key=prng_key) 
        self.a = nn.linear( 2 * query_dim * num_heads, num_heads, key=prng_key) 
        self.linear = nn.Linear(in_features, out_features,  key=prng_key)
        self.act = act

    def forward(self, input):
        x, adj = input
        n = x.shape[0]
        s, r = adj[0], adj[1]
        attr = self.query_fn(x)
        sender_attr = attr[s]
        receiver_attr = attr[r]

        e = jnp.concatenate((sender_attr,receiver_attr), axis=1)
        e = self.W(e)
        alpha = e 
        
        h = self.linear.forward(x)
        h = dropout(self.p)(hidden, inference=False, key=prng_key)
        
        h = tree.tree_map(lambda x: jax.segment_sum(x[s], r, n), h)
        
        h = self.act(h)

        output = h, adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

