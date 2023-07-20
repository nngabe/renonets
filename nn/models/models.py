from typing import Any, Optional, Sequence, Tuple, Union

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GCNConv, GATConv, Linear, get_dim_act
import utils.math_utils as pmath

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng_key = jax.random.PRNGKey(0)
prng_fn = lambda i: jax.random.PRNGKey(i)

class null(eqx.Module):
    def __init__(self, args):
        super(null, self).__init__()
    def __call__(self, x=None, adj=None):
        return 0.

class Model(eqx.Module):
    c: float
    skip: bool
    layers: eqx.nn.Sequential
    encode_graph: bool
    manifold: Optional[manifolds.base.Manifold] = None

    def __init__(self, args):
        super(Model, self).__init__()
        self.c = args.c
        self.skip = args.skip

    def __call__(self, x, adj=None, w=None):
        if self.skip:
            return self._cat(x, adj, w)
        else:
            return self._plain(x, adj, w)

    def _plain(self, x, adj=None, w=None):
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x, adj, w)
            else:
                x = layer(x)
        return x 

    def _cat(self, x, adj=None, w=None):
        x_i = [x]
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x, adj, w)
                x_i.append(x)
            else:
                x = layer(x)
                x_i.append(x)
        return jnp.concatenate(x_i, axis=1)

class MLP(eqx.Module):
    layers: eqx.nn.Sequential
    drop_fn: eqx.nn.Dropout

    def __init__(self, args):
        super(MLP, self).__init__()
        dims, act, _ = get_dim_act(args)
        self.drop_fn = eqx.nn.Dropout(args.dropout)
        layers = []
        key, subkey = jax.random.split(prng_key)
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, subkey))
            key, subkey = jax.random.split(key)
        self.layers = nn.Sequential(layers)

    def __call__(self, x, key=prng_key):
        for layer in self.layers:
            x = layer(x, key)
        return x

class GCN(Model):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__(args)
        dims, act, _ = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            gc_layers.append(GCNConv(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(gc_layers)
        self.encode_graph = True

class GAT(Model):
    """
    Graph Attention Networks.
    """

    def __init__(self, args):
        super(GAT, self).__init__(args)
        dims, act, _ = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GATConv(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(gat_layers)
        self.encode_graph = True

class HNN(Model):
    """
    Hyperbolic Neural Networks.
    """
    curvatures: list
    def __init__(self, args):
        super(HNN, self).__init__(args)
        dims, act, self.curvatures = get_dim_act(args)
        self.manifold = getattr(manifolds, args.manifold if args.dec_init else args.manifold_pinn)() 
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            hnn_layers.append( hyp_layers.HNNLayer(self.manifold, in_dim, out_dim, args.c, args.dropout, act, args.bias) )
        self.layers = nn.Sequential(hnn_layers)
        self.encode_graph = False

    def __call__(self, x): 
        return super(HNN, self).__call__(x)

class HGCN(Model):
    """
    Hyperbolic-GCN.
    """
    curvatures: jax.numpy.ndarray 
    def __init__(self, args):
        super(HGCN, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, act, self.curvatures = get_dim_act(args)
        self.curvatures.append(args.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            hgc_layers.append(hyp_layers.HGCNLayer(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att)
                            )
        self.layers = nn.Sequential(hgc_layers)
        self.encode_graph = True

    def __call__(self, x, adj, w=None):
        return super(HGCN, self).__call__(x, adj, w)

