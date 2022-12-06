from typing import Any, Optional, Sequence, Tuple, Union

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox.module import Module, static_field


class Model(eqx.Module):
    c: float
    skip: bool
    omega: jax.numpy.ndarray
    layers: eqx.nn.Sequential
    encode_graph: bool
    manifold: Optional[manifolds.base.Manifold] = None

    def __init__(self, args):
        super(Model, self).__init__()
        self.c = args.c
        self.skip = args.skip
        self.omega = 1 / 10 ** jnp.linspace(0,6,args.time_dim)

    def time_encode(self, t, d):
        #coefs = torch.cos(self.omega*t)
        coefs = t#torch.cat([t,t**2],requires_grad=True)
        #coefs = torch.cat([coefs,torch.zeros( self.omega.shape[0] - coefs.shape[0])])
        return coefs#*jnp.ones((d,1)) 

    def __call__(self, x, adj=None, t=None):
        if t != None:
            #t = torch.tensor(int(t),dtype=torch.float)
            d = x.shape[0]
            te = self.time_encode(t,d)
            x = cat([te,x], axis=0)
        if self.skip:
            return self._cat(x,adj)
        else:
            return self._plain(x,adj)


    def _plain(self, x, adj):
        if self.encode_graph:
            for layer in self.layers:
                x,_ = layer.forward([x,adj])
        else:
            for layer in self.layers:
                x = layer.forward(x)
        return x

    def _cat(self, x, adj):
        xi = [x]
        if self.encode_graph:
            for layer in self.layers:
                x,_ = layer.forward([x,adj])
                xi.append(x)
        else:
            for layer in self.layers:
                x = layer.forward(x)
                xi.append(x)
        return cat(xi, axis=0)

class MLP(Model):
    """
    Multi-layer perceptron.
    """

    def __init__(self, args):
        super(MLP, self).__init__(args)
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(layers)
        self.encode_graph = False

class GCN(Model):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c,args)
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, args.act[0], args.bias))
        self.layers = nn.Sequential(gc_layers)
        self.encode_graph = True

class HNN(Model):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c,args)
        init_dec = args.enc^1
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1 - init_dec):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        if init_dec:
            in_dim, out_dim = dims[-2], dims[-1]
            hnn_layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class HGCN(Model):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c,args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Model):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c,args)
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True

