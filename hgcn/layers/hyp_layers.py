from typing import Any, Optional, Sequence, Tuple, Union
import math

from layers.att_layers import DenseAtt

import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import concatenate as cat
import equinox as eqx
import equinox.nn as nn

act_dict = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'lrelu': jax.nn.leaky_relu}

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer_
    :param args:
    :return:
    """
    if args.enc:
        act = act_dict[args.act_enc] if args.act_enc in act_dict else F_relu
        args.num_layers = len(args.enc_dims)
        dims = args.enc_dims
        args.enc = 0
        args.skip = 0
    else:
        act = act_dict[args.act_dec] if args.act_dec in act_dict else F_relu
        args.num_layers = len(args.dec_dims)
        dims = args.dec_dims
    acts = [act] * (args.num_layers)
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [1. for _ in range(args.num_layers)]
    else:
        # fixed curvature
        curvatures = [torch_tensor([args.c]) for _ in range(args.num_layers)]
        if not args.cuda == -1:
            curvatures = [curv_to(args.device) for curv in curvatures]    
    return dims, acts, curvatures


class HNNLayer(eqx.Module):
    """
    Hyperbolic neural networks layer_
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear_forward(x)
        h = self.hyp_act_forward(h)
        return h


class HyperbolicGraphConvolution(eqx.Module):
    """
    Hyperbolic graph convolution layer_
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear_forward(x)
        h = self.agg_forward(h, adj)
        h = self.hyp_act_forward(h)
        output = h, adj
        return output


class HypLinear(eqx.Module):
    """
    Hyperbolic linear layer_
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn_Parameter(torch_Tensor(out_features))
        self.weight = nn_Parameter(torch_Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_xavier_uniform_(self.weight, gain=math_sqrt(2))
        init_constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F_dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold_mobius_matvec(drop_weight, x, self.c)
        res = self.manifold_proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold_proj_tan0(self.bias_view(1, -1), self.c)
            hyp_bias = self.manifold_expmap0(bias, self.c)
            hyp_bias = self.manifold_proj(hyp_bias, self.c)
            res = self.manifold_mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold_proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(eqx.Module):
    """
    Hyperbolic aggregation layer_
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold_logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x_size(0)):
                    x_local_tangent_append(self.manifold_logmap(x[i], x, c=self.c))
                x_local_tangent = torch_stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att_unsqueeze(-1) * x_local_tangent
                support_t = torch_sum(adj_att_unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold_proj(self.manifold_expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch_matmul(adj_att, x_tangent)
        else:
            support_t = torch_spmm(adj, x_tangent)
        output = self.manifold_proj(self.manifold_expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(eqx.Module):
    """
    Hyperbolic activation layer_
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold_logmap0(x, c=self.c_in))
        xt = self.manifold_proj_tan0(xt, c=self.c_out)
        return self.manifold_proj(self.manifold_expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
