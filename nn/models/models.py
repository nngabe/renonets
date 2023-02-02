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

    def __call__(self, x, adj=None):
        if self.skip:
            return self._cat(x,adj)
        else:
            return self._plain(x,adj)

    def _plain(self, x, adj=None):
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x,adj)
            else:
                x = layer(x)
        return x 

    def _cat(self, x, adj=None):
        x_i = [x]
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x,adj)
                x_i.append(x)
            else:
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
    curvatures: list
    def __init__(self, args):
        super(HNN, self).__init__(args)
        dims, act, self.curvatures = hyp_layers.get_dim_act_curv(args) 
        self.manifold = getattr(manifolds, args.manifold if args.dec_init else args.manifold_pinn)() 
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            hnn_layers.append( hyp_layers.HNNLayer(self.manifold, in_dim, out_dim, args.c, args.dropout, act, args.bias) )
        if args.dec_init==0:
            for i in range(1,1+args.post_hyp):
                hnn_layers[-i] = hyp_layers.HNNLayer(manifolds.Euclidean(), dims[-(i+1)], dims[-i], args.c, args.dropout, act, args.bias)
        self.layers = nn.Sequential(hnn_layers)
        self.encode_graph = False

    def __call__(self, x): 
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HNN, self).__call__(x)

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

    def __call__(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).__call__(x, adj=adj)

