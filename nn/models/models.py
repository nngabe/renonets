from typing import Any, Optional, Sequence, Tuple, Union, List

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

class GraphNet(eqx.Module):
    c: float
    skip: bool
    layers: eqx.nn.Sequential
    lin: eqx.nn.Sequential
    encode_graph: bool
    res: bool
    manifold: Optional[manifolds.base.Manifold] = None

    def __init__(self, args):
        super(GraphNet, self).__init__()
        self.c = args.c
        self.skip = args.skip
        self.res = args.res

    def __call__(self, x, adj=None, w=None):
        if self.res:
            return self._res(x, adj, w)
        if self.skip:
            return self._cat(x, adj, w)
        else:
            return self._plain(x, adj, w)

    def exp(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y

    def _plain(self, x, adj=None, w=None):
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x, adj, w)
            else:
                x = layer(x)
        return x 

    def _cat(self, x, adj=None, w=None):
        x_i = [x]
        x = self.exp(x)
        for layer in self.layers:
            x,_ = layer(x, adj, w)
            x_i.append(x)
        return jnp.concatenate(x_i, axis=1)
    
    def _res(self, x, adj=None, w=None):
        x = self.exp(x)
        for conv,lin in zip(self.layers,self.lin):
            h,_ = conv(x, adj, w)
            x = jax.vmap(lin)(self.log(x)) + self.log(h)
            x = self.exp(x)
        return x


class MHA(eqx.Module):
    layers: eqx.nn.Sequential
    encode_graph: bool

    def __init__(self, args):
        super(MHA, self).__init__()
        dims, act, _ = get_dim_act(args)
        layers = []
        key, subkey = jax.random.split(prng_key)
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(MultiheadBlock(in_dim, out_dim, p=args.dropout, key=subkey))
            key= jax.random.split(key)[0]
        self.layers = layers 
        self.encode_graph = False
    def __call__(self, x, adj=None, w=None, key=prng_key):
        for layer in self.layers:
            x = layer(x, key)
            key = jax.random.split(key)[0]
        return x 
        

class MultiheadBlock(eqx.Module):
    multihead: eqx.nn.MultiheadAttention
    norm: List[eqx.nn.LayerNorm]
    ffn: List[eqx.nn.Linear]
    L: jnp.ndarray
    b: jnp.ndarray    
    dropout: eqx.nn.Dropout
    omega: np.ndarray

    def __init__(self, in_dim, out_dim, hidden_dim=256, n_heads=2, p=0.1, affine=True, key=prng_key):
        super(MultiheadBlock, self).__init__()
        key = jax.random.split(jax.random.PRNGKey(0),10)
        self.dropout = eqx.nn.Dropout(p)
        embed_dim = 32
        self.multihead = eqx.nn.MultiheadAttention(n_heads, embed_dim, key=key[0])
        self.norm = [eqx.nn.LayerNorm(embed_dim, elementwise_affine=affine),
                     eqx.nn.LayerNorm(embed_dim, elementwise_affine=affine)]
        self.ffn = [eqx.nn.Linear(embed_dim, hidden_dim, key=key[1]),
                    eqx.nn.Linear(hidden_dim, embed_dim, key=key[2]),
                    eqx.nn.Linear(embed_dim, out_dim, key=key[3])]
        self.L = jax.random.normal(key[4], (in_dim,))
        self.b = 0.01 * jax.random.normal(key[5], (out_dim,))
        self.omega = (10 ** jnp.linspace(-5,2,embed_dim//2))

    def pe(self, x):
        x = x + jnp.linspace(0, jnp.pi, x.shape[-1])
        xo = jnp.einsum('i,j -> ij', x, self.omega)
        return jnp.concatenate([jnp.cos(xo), jnp.sin(xo)],axis=-1)

    def __call__(self, x, key=prng_key):
        x = self.pe(x)
        attn = self.multihead(x,x,x)
        x = x + self.dropout(attn, key=key)
        x = jax.nn.gelu(x)
        x = self.norm[0](x)

        y = jax.vmap(self.ffn[0])(x)
        y = jax.nn.gelu(y)
        y = jax.vmap(self.ffn[1])(y)
        x = x + self.dropout(y, key=key)
        x = jax.nn.gelu(x)
        x = self.norm[1](x)
        x = jax.vmap(self.ffn[2])(x) 
        x = jnp.einsum('ij,i,j -> j', x, self.L, self.b)
        return x



class MLP(eqx.Module):
    layers: eqx.nn.Sequential
    drop_fn: eqx.nn.Dropout
    norm: List[eqx.nn.LayerNorm]
    encode_graph: bool

    def __init__(self, args):
        super(MLP, self).__init__()
        dims, act, _ = get_dim_act(args)
        self.drop_fn = eqx.nn.Dropout(args.dropout)
        layers = []
        self.norm = []
        key, subkey = jax.random.split(prng_key)
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, subkey))
            self.norm.append(eqx.nn.LayerNorm(out_dim))
            key, subkey = jax.random.split(key)
        self.layers = nn.Sequential(layers)
        self.encode_graph = False
    def __call__(self, x, adj=None, w=None, key=prng_key):
        x = self.layers[0](x,key)
        key = jax.random.split(key)[0]
        for layer,norm in zip(self.layers[1:-1],self.norm[1:-1]):
            x = x + layer(x, key)
            x = norm(x)
            key = jax.random.split(key)[0]
        x = self.layers[-1](x,key)
        return x

class GCN(GraphNet):
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


class HNN(GraphNet):
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

class HGCN(GraphNet):
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
        lin_layers = []
        key, subkey = jax.random.split(prng_key)
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            hgc_layers.append( hyp_layers.HGCNLayer( key, self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att))
            if in_dim==out_dim:
                lin_layers.append( lambda x: x)
            else:
                lin_layers.append( eqx.nn.Linear(in_dim, out_dim, key=key))
            key = jax.random.split(key)[0]
        self.layers = nn.Sequential(hgc_layers)
        self.lin = nn.Sequential(lin_layers)
        self.encode_graph = True

    def __call__(self, x, adj, w=None):
        return super(HGCN, self).__call__(x, adj, w)

