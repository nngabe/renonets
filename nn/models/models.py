from typing import Any, Optional, Sequence, Tuple, Union

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from equinox.module import Module, static_field

prng_key = jax.random.PRNGKey(0)

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

    def __call__(self, x, t=None, adj=None):
        if self.skip:
            return self._cat(x,adj)
        else:
            return self._plain(x,adj)


    def _plain(self, x, adj):
        if self.encode_graph:
            for layer in self.layers:
                x,_ = layer(x,adj)
        else:
            for layer in self.layers:
                x = layer(x)
        return x

    def _cat(self, x, adj):
        x_i = [x]
        if self.encode_graph:
            for layer in self.layers:
                x,_ = layer(x,adj)
                x_i.append(x)
        else:
            for layer in self.layers:
                x = layer(x)
                x_i.append(x)
        return jnp.concatenate(x_i, axis=1)

class MLP(Model):
    """
    Multi-layer perceptron.
    """

    def __init__(self, args):
        super(MLP, self).__init__(args)
        dims, act = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(layers)
        self.encode_graph = False

class GCN(Model):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__(args)
        dims, act = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(gc_layers)
        self.encode_graph = True

class GAT(Model):
    """
    Graph Attention Networks.
    """

    def __init__(self, args):
        super(GAT, self).__init__(args)
        dims, act = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(gat_layers)
        self.encode_graph = True

class HNN(Model):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, args):
        super(HNN, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, act, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            hnn_layers.append( hyp_layers.HNNLayer(self.manifold, in_dim, out_dim, args.c, args.dropout, act, args.bias) )
        self.layers = nn.Sequential(hnn_layers)
        self.encode_graph = False

    def __call__(self, x, t=None, adj=None):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).__call__(x_hyp)

class HGCN(Model):
    """
    Hyperbolic-GCN.
    """
    curvatures: list
    def __init__(self, args):
        super(HGCN, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, act, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(args.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            hgc_layers.append(hyp_layers.HGCNLayer(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                                )
                            )
        self.layers = nn.Sequential(hgc_layers)
        self.encode_graph = True

    def __call__(self, x, adj, t=None):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).__call__(x_hyp,t, adj)




