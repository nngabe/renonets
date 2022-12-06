from typing import Any, Optional, Sequence, Tuple, Union, Callable
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn
from equinox.nn import Dropout as dropout

prng_key = jax.random.PRNGKey(0)
act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu}

def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """

    if args.enc:
        act = act_dict[args.act_enc] if args.act_enc in act_dict else F.relu
        args.num_layers = len(args.enc_dims) - 1
        dims = args.enc_dims
        args.enc = 0
        args.skip = 0
    else:
        act = act_dict[args.act_enc] if args.act_enc in act_dict else F.relu
        args.num_layers = len(args.dec_dims) - 1
        dims = args.dec_dims
    acts = [act] * (args.num_layers)
    return dims, acts


class Linear(eqx.Module): 
    p: float
    linear: jax.numpy.array
    act: Callable 
    
    def __init__(self, in_features, out_features, p, act, use_bias):
        super(Linear, self).__init__()
        self.p = p # dropout prob
        self.linear = nn.Linear(in_features, out_features, use_bias, key=prng_key)
        self.act = act

    def forward(self, x):
        hidden = self.linear(x)
        hidden = dropout(self.p)(hidden, inference=False, key=prng_key)
        out = self.act(hidden)
        return out


class GraphConvolution(eqx.Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias, key=prng_key)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = dropout(self.p)(hidden, inference=False, key=prng_key)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


