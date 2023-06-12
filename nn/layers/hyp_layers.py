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

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer_
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
        print('All layers already init-ed! Define additional layers if needed.')
        raise

    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [1. for _ in range(args.num_layers)]
    else:
        curvatures = [args.c for _ in range(args.num_layers)]

    return dims, act, curvatures


class DenseAtt(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.linear = nn.Linear(2 * in_features, 1, key=prng_key)

    def forward (self, x, adj):
        n = x.shape[0]
        i,j = jnp.where(jnp.ones((n,n)))

        x_cat = jnp.concatenate((x[i], x[j]),axis=0)
        att_adj = jax.vmap(self.linear)(x_cat)
        att_adj = jax.nn.sigmoid(att_adj).reshape(n,n)
        return att_adj


class HypLinear(eqx.Module):
    """
    Hyperbolic linear layer_
    """
    dropout: eqx.nn.Dropout
    manifold: manifolds.base.Manifold
    c: float
    bias: jax.numpy.ndarray
    weight: jax.numpy.ndarray

    def __init__(self, manifold, in_features, out_features, c, p, use_bias):
        super(HypLinear, self).__init__()
        self.dropout = eqx.nn.Dropout(p) #lambda x: dropout(self.p)(x, inference=False, key=prng_key)
        self.manifold = manifold
        self.c = c
        self.bias = 1e-7*jnp.ones((out_features,1)) 
        self.weight = jnp.zeros((out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_weights = jax.nn.initializers.glorot_uniform()
        self.weight = init_weights(prng_key,self.weight.shape)

    def __call__(self, x):
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
    """
    Hyperbolic aggregation layer_
    """
    manifold: manifolds.base.Manifold
    c: float
    p: float
    use_att: bool
    local_agg: bool
    att: DenseAtt

    def __init__(self, manifold, c, in_features, p, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.p = p
        self.manifold = manifold
        self.c = c
        self.p = p
        self.local_agg = local_agg
        self.use_att = use_att
        self.att = DenseAtt(in_features, p)

    def __call__(self, x, adj):
        s,r = adj[0],adj[1]
        n = x.shape[0]
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.shape[0]):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = jnp.stack(x_local_tangent, axis=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = jnp.expand_dims(adj_att,-1) * x_local_tangent
                support_t = jnp.sum(jnp.expand_dims(adj_att,-1) * x_local_tangent, axis=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = jnp.matmul(adj_att, x_tangent)
        else:
            support_t = tree.tree_map(lambda x: jax.ops.segment_sum(x[s], r, n), x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(eqx.Module):
    """
    Hyperbolic activation layer_
    """
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

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

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

    def __init__(self, manifold, in_features, out_features, c_in, c_out, p, act, use_bias, use_att, local_agg):
        super(HGCNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, p, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, p, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def __call__(self, x, adj):
        h = self.linear(x)
        h = self.agg(h, adj)
        h = self.hyp_act(h)
        output = h, adj
        return output

