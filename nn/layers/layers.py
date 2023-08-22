from typing import Any, Optional, Sequence, Tuple, Union, Callable, List
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
act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu, 'gelu': jax.nn.gelu}

def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer_
    :param args:
    :return:
    """
    if args.enc_init:
        act = act_dict[args.act_enc] 
        args.num_layers = len(args.enc_dims)
        dims = args.enc_dims
        args.enc_init = 0
        args.skip = 0
    elif args.dec_init:
        act = act_dict[args.act_dec] 
        args.num_layers = len(args.dec_dims)
        dims = args.dec_dims
        args.dec_init = 0
    elif args.pde_init:
        act = act_dict[args.act_pde] 
        args.num_layers = len(args.pde_dims)
        dims = args.pde_dims
        args.pde_init -= 1
    elif args.pool_init:
        args.res = 1
        act = act_dict[args.act_pool] 
        args.num_layers = len(args.pool_dims)
        dims = args.pool_dims
        args.pool_dims[-1] = max(args.pool_dims[-1]//args.pool_red, 1)
        args.pool_init -= 1
        args.use_att = args.use_att_pool
    elif args.embed_init: 
        act = act_dict[args.act_pool] 
        dims = args.embed_dims
        args.embed_init -= 1
    else:
        print('All layers already init-ed! Define additional layers if needed.')
        raise

    
    # for now curvatures are static, change list -> jax.ndarray to make them learnable
    if args.c is None:
        curvatures = [1. for _ in range(args.num_layers)]
    else:
        curvatures = [args.c for _ in range(args.num_layers)]

    return dims, act, curvatures

class Attention(eqx.Module):
    multihead: eqx.nn.MultiheadAttention
    norm: List[eqx.nn.LayerNorm]
    ffn: eqx.nn.Sequential
    dropout: eqx.nn.Dropout

    def __init__(self, in_dim, out_dim, num_heads=4, hidden_dim=None, p=0., affine=True, act=jax.nn.silu, key=prng_key):

        if hidden_dim==None: hidden_dim = 3 * out_dim
        self.multihead = eqx.nn.MultiheadAttention(num_heads, in_dim, key=key)
        self.norm = [eqx.nn.LayerNorm(in_dim, use_weight=affine),
                     eqx.nn.LayerNorm(out_dim, use_weight=affine)]
        self.ffn = eqx.nn.Sequential([eqx.nn.Linear(in_dim, hidden_dim, key=key),
                       eqx.nn.Dropout(p),
                       lambda x,key: act(x),
                       eqx.nn.Linear(hidden_dim, out_dim, key=key)])
        self.dropout = eqx.nn.Dropout(p)
        
    def __call__(self, x, key=prng_key):

        a = self.multihead(x,x,x)
        x = x + self.dropout(a, key=key)
        x = self.norm[0](x)

        y = jax.vmap(self.ffn)(x)
        x = self.dropout(y, key=key)
        x = self.norm[1](x)
        return x

class Linear(eqx.Module): 
    linear: eqx.nn.Linear
    act: Callable
    dropout: Callable
    
    def __init__(self, in_features, out_features, p=0., act=jax.nn.silu, key=prng_key):
        super(Linear, self).__init__()
        self.linear = eqx.nn.Linear(in_features, out_features,  key=key)
        self.act = act
        self.dropout = dropout(p)

    def __call__(self, x, key=prng_key):
        x = self.dropout(x, key=key)
        x = self.linear(x)
        out = self.act(x)
        return out


class GCNConv(eqx.Module):
    p: float
    linear: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, p=0., act=jax.nn.silu, use_bias=True):
        super(GCNConv, self).__init__()
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




class GATConv(eqx.Module):
    linear: eqx.nn.Linear
    a: eqx.nn.Linear
    W: eqx.nn.Linear
    act: Callable
    dropout: Callable

    def __init__(self, in_features, out_features, p=0., act=jax.nn.silu, use_bias=True, num_heads=3, query_dim=8):
        super(GATConv, self).__init__()
        self.dropout = dropout(p)
        self.W = nn.Linear(in_features, query_dim * num_heads, key=prng_key) 
        self.a = nn.Linear( 2 * query_dim * num_heads, num_heads, key=prng_key) 
        self.linear = nn.Linear(in_features, out_features,  key=prng_key)
        self.act = act

    def __call__(self, x, key=prng_key):
        x, adj = input
        n = x.shape[0]
        s = r = jnp.arange(0,n)
        attr = jax.vmap(self.W)(x)
        sender_attr = attr[s]
        receiver_attr = attr[r]

        e = jnp.concatenate((sender_attr,receiver_attr), axis=1)
        alpha = jax.vmap(self.a)(e)
        
        h = dropout(h, key=key)
         
        h = tree.tree_map(lambda x: jax.segment_sum(x[s] * alpha[s], r, n), h)
        
        h = self.act(h)

        output = h, adj
        return output


