from typing import Any, Optional, Sequence, Tuple, Union, List
import copy

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GCNConv, GATConv, Linear, get_dim_act
import utils.math_utils as pmath
import lib.function_spaces

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from equinox import Module, static_field

prng = lambda i:  jax.random.PRNGKey(i)

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

    def __call__(self, x, adj=None, w=None, key=prng(0)):
        if self.res:
            return self._res(x, adj, w, key)
        elif self.skip:
            return self._cat(x, adj, w, key)
        else:
            return self._forward(x, adj, w, key)

    def exp(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y

    def _forward(self, x, adj, w, key):
        for layer in self.layers:
            if self.encode_graph:
                x,_ = layer(x, adj, w, key)
            else:
                x = layer(x)
            key = jax.random.split(key)[0]
        return x 

    def _cat(self, x, adj, w, key):
        x_i = [x]
        x = self.exp(x)
        for layer in self.layers:
            x,_ = layer(x, adj, w, key)
            x_i.append(x)
            key = jax.random.split(key)[0]
        return jnp.concatenate(x_i, axis=1)
    
    def _res(self, x, adj, w, key):
        x = self.exp(x)
        for conv,lin in zip(self.layers,self.lin):
            h,_ = conv(x, adj, w, key)
            x = jax.vmap(lin)(self.log(x)) + self.log(h)
            x = self.exp(x)
            key = jax.random.split(key)[0]
        return x


class MHA(eqx.Module):
    blocks: eqx.nn.Sequential
    encode_graph: bool

    def __init__(self, args):
        super(MHA, self).__init__()
        dims, act, _ = get_dim_act(args)
        blocks = []
        key = jax.random.PRNGKey(0)
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            blocks.append(MultiheadBlock(in_dim, out_dim, p=args.dropout, key=key))
            key = jax.random.split(key)[0]
        self.blocks = blocks
        self.encode_graph = False
    def __call__(self, x, adj=None, w=None, key=prng(0)):
        for block in self.blocks:
            x = block(x, key)
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

    def __init__(self, in_dim, out_dim, hidden_dim=256, n_heads=2, p=0.1, affine=True, key=prng(0)):
        super(MultiheadBlock, self).__init__()
        keys = jax.random.split(jax.random.PRNGKey(0), 10)
        self.dropout = eqx.nn.Dropout(p)
        embed_dim = 64
        self.multihead = eqx.nn.MultiheadAttention(
            n_heads, 
            embed_dim, 
            use_query_bias=True, 
            use_key_bias=True, 
            use_value_bias=True, 
            use_output_bias=True, 
            dropout_p=p, 
            key=keys[0]
        )
        self.norm = [eqx.nn.LayerNorm(embed_dim, elementwise_affine=affine),
                     eqx.nn.LayerNorm(embed_dim, elementwise_affine=affine)]
        self.ffn = [eqx.nn.Linear(embed_dim, hidden_dim, key=keys[1]),
                    eqx.nn.Linear(hidden_dim, embed_dim, key=keys[2]),
                    eqx.nn.Linear(embed_dim, out_dim, key=keys[3])]
        self.L = (1. / in_dim) * jax.random.normal(keys[4], (in_dim,out_dim))
        self.b = 0. * jax.random.normal(keys[5], (out_dim,))
        self.omega = (10 ** jnp.linspace(-3.,1.,embed_dim//2))

    def pe(self, x):
        x = x + jnp.linspace(0, jnp.pi, x.shape[-1])
        xo = jnp.einsum('i,j -> ij', x, self.omega)
        return jnp.concatenate([jnp.cos(xo), jnp.sin(xo)],axis=-1)

    def __call__(self, x, key=prng(0)):
        keys = jax.random.split(key, 10)
        x = self.pe(x)
        attn = self.multihead(x,x,x,key=keys[0])
        x = x + self.dropout(attn, key=keys[1])
        x = jax.nn.gelu(x)
        x = self.norm[0](x)

        y = jax.vmap(self.ffn[0])(x)
        y = jax.nn.gelu(y)
        y = jax.vmap(self.ffn[1])(y)
        x = x + self.dropout(y, key=keys[2])
        x = jax.nn.gelu(x)
        #x = self.norm[1](x)
        x = jax.vmap(self.ffn[2])(x) 
        x = jax.nn.gelu(x)
        x = jnp.einsum('ij,ij -> j', x, self.L)
        #x = jnp.einsum('ij,ij,j -> j', x, self.L, self.b)
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
        key, subkey = jax.random.split(prng(0))
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, subkey))
            self.norm.append(eqx.nn.LayerNorm(out_dim))
            key, subkey = jax.random.split(key)
        self.layers = nn.Sequential(layers)
        self.encode_graph = False
    def __call__(self, x, adj=None, w=None, key=prng(0)):
        x = self.layers[0](x,key)
        key = jax.random.split(key)[0]
        for layer,norm in zip(self.layers[1:-1],self.norm[1:-1]):
            x = layer(x, key) #+ x
            x = norm(x)
            key = jax.random.split(key)[0]
        x = self.layers[-1](x,key)
        return x

class DeepOnet(eqx.Module):
    
    trunk: eqx.Module
    branch: eqx.Module
    func_space: eqx.Module
    drop_fn: eqx.nn.Dropout
    norm: List[eqx.nn.LayerNorm]
    x_dim: int
    tx_dim: int
    u_dim: int
    depth: int
    p: int

    def __init__(self, args, layer=Linear):
        super(DeepOnet, self).__init__()
        self.func_space = getattr(lib.function_spaces, args.func_space)()
        dims, act, _ = get_dim_act(args)
        self.drop_fn = eqx.nn.Dropout(args.dropout)
        self.norm = []
        self.x_dim = args.x_dim
        self.tx_dim = args.time_dim + args.x_dim 
        self.u_dim = args.kappa
        self.p = args.p_basis
        keys = jax.random.split(prng(0))

        # set dimensions of branch net        
        branch_dims = copy.copy(dims)
        branch_dims[0] = self.u_dim
        branch_dims[-1] *= args.x_dim
        branch_dims[-1] *= self.p

        # set dimensions of trunk net
        trunk_dims = copy.copy(dims)
        trunk_dims[0] = self.tx_dim + sum(args.enc_dims[1:]) - self.x_dim
        trunk_dims[-1] *= self.p

        branch,trunk = [],[]
        for i in range(len(dims) - 1):
            trunk.append(Linear(trunk_dims[i], trunk_dims[i+1], args.dropout, act, keys[i]))
            branch.append(Linear(branch_dims[i], branch_dims[i+1], args.dropout, act, keys[i]))
        
        self.trunk = nn.Sequential(trunk)
        self.branch = nn.Sequential(branch)
        self.depth = len(self.trunk)

    def __call__(self, x, adj=None, w=None, key=prng(0)):
        keys = jax.random.split(key,10)
        tx, uz = x[:self.tx_dim], x[self.tx_dim:]
        u, z = uz[:self.u_dim], uz[self.u_dim:]
        txz = jnp.concatenate([tx,z],axis=-1)
        t = self.trunk[0](txz, keys[0])
        b = u #self.func_space(u)
        b = self.branch[0](b, keys[2])
        keys = jax.random.split(keys[0])
        
        for i in range(1,self.depth-1):
            t = self.trunk[i](t, keys[0])
            b = self.branch[i](b, keys[1])
            keys = jax.random.split(keys[0])
        
        t = self.trunk[-1](t, keys[0]).reshape(-1, self.p)
        b = self.branch[-1](b, keys[1]).reshape(-1, self.p, self.x_dim)
        G = jnp.einsum('ijk,ij -> i', b, t) / self.p
        
        return G


class HGCN(GraphNet):
    
    curvatures: jax.numpy.ndarray 
    
    def __init__(self, args):
        super(HGCN, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        dims, act, self.curvatures = get_dim_act(args)
        self.curvatures.append(args.c)
        hgc_layers = []
        lin_layers = []
        key, subkey = jax.random.split(prng(0))
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

    def __call__(self, x, adj, w=None, key=prng(0)):
        return super(HGCN, self).__call__(x, adj, w, key)

