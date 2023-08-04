import argparse
import glob

from nn.utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (1e-5, 'learning rate'),
        'dropout': (0.04, 'dropout probability'),
        'epochs': (10001, 'maximum number of epochs to train for'),
        'slaw_iter': (100000, 'iteration to start SLAW'),
        'drop_iter': (0, 'iteration to stop dropout'),
        'weight_decay': (1e-3, 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'log_freq': (20, 'how often to compute print train/val metrics (in epochs)'),
        'max_norm': (1., 'max norm for gradient clipping, or None for no gradient clipping'),
        'verbose': (True, 'print training data to console'),
        'opt_study': (False, 'whether to run a hyperparameter optimization study or not'),
        'batch_red': (2, 'factor of reduction for batch size'),
        'pool_red': (4, 'factor of reduction for each pooling step'),
    },
    'model_config': {
        # init flags for neural nets
        'enc_init': (1, 'flag indicating whether the encoder remains to be init-ed or not.'),
        'dec_init': (1, 'flag indicating whether the decoder remains to be init-ed or not.'),
        'pde_init': (2, 'flag indicating number of pde functions which remain to be init-ed.'),
        'pool_init': (3, 'flag indicating number of pooling modules which remain to be init-ed.'),
        'embed_init': (3, 'flag indicating number of embedding modules which remain to be init-ed.'), 
        # loss weights
        'w_data': (1., 'weight for data loss.'),
        'w_pde': (10., 'weight for pde loss.'),
        'w_gpde': (10., 'weight for gpde loss.'),
        'w_ent': (10., 'weight for assignment matrix entropy loss.'),
        'v_scaler': (0., 'max weight of viscous term.'),
        'input_scaler': (1., 'rescaling of input'),
        'rep_scaler': (10., 'rescaling of graph features'),
        'tau_scaler': (10., 'rescaling of tau encoding'),

        # which layers use time encodings and what dim should encodings be
        'time_enc': ([0,1,1], 'whether to insert time encoding in encoder, decoder, and pde functions, respectively.'),
        'time_dim': (12, 'dimension of time embedding'), 
        'x_dim': (3, 'dimension of differentiable coordinates for PDE'),
 
        # input/output sizes
        'kappa': (60, 'size of lookback window used as input to encoder'),
        'tau_max': (30, 'maximum steps ahead forecast'),
        'tau_num': (5, 'number of tau steps for each training bundle'),
        
        # specify models. pde function layers are the same as the decoder layers by default.
        'encoder': ('HGCN', 'which encoder to use, can be any of [MLP, HNN, GCN, GAT, HGCN]'),
        'decoder': ('MLP', 'which decoder to use, can be any of [MLP, HNN, GCN, GAT, HGCN]'),
        'pde': ('neural_burgers', 'which PDE to use for the PINN loss'),
        'pool': ('HGCN', 'which model to compute coarsening matrices'),

        # dims of neural nets. -1 will be inferred based on args.skip and args.time_enc. 
        'enc_width': (96, 'dimensions of encoder layers'),
        'dec_width': (312,'dimensions of decoder layers'),
        'pde_width': (312, 'dimensions of each pde layers'),
        'enc_depth': (2, 'dimensions of encoder layers'),
        'dec_depth': (4,'dimensions of decoder layers'),
        'pde_depth': (3, 'dimensions of each pde layers'),
        'enc_dims': ([-1,96,-1], 'dimensions of encoder layers'),
        'dec_dims': ([-1,256,256,-1],'dimensions of decoder layers'),
        'pde_dims': ([-1,256,256,1], 'dimensions of each pde layers'),
        'pool_dims': ([-1,96,-1], 'dimesions of pooling layers.'), 
        'embed_dims': ([-1,96,-1], 'dimensions of embedding layers.'),
        #activations for each network
        'act_enc': ('silu', 'which activation function to use (or None for no activation)'),
        'act_dec': ('silu', 'which activation function to use (or None for no activation)'),
        'act_pde': ('silu', 'which activation function to use (or None for no activation)'),
        'act_pool': ('silu', 'which activation function to use (or None for no activation)'),
       
        # additional params for layer specification
        'bias': (1, 'whether to use bias in layers or not'),
        'skip': (1, 'whether to use skip connections or not. set to 0 after encoder init.'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'manifold_pool': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        
        # graph encoder params
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_att_pool': (0, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'path': ('499_k2', 'snippet from which to infer data path'),
        'log_path': (None, 'snippet from which to infer log/model path.'),
        'test_prop': (0.1, 'proportion of test nodes for forecasting'),
    }
}

def set_dims(args):
    args.enc_dims[0] = args.kappa
    args.enc_dims[-1] = args.enc_width//2 
    args.dec_dims[-1] = args.x_dim
    args.enc_dims[1:-1] = (args.enc_depth-1) * [args.enc_width]
    args.dec_dims[1:-1] = (args.dec_depth-1) * [args.dec_width]
    args.pde_dims[1:-1] = (args.pde_depth-1) * [args.pde_width]
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim * 2
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim * 2
    
    args.pde_dims[0] = args.dec_dims[0] + args.x_dim
    args.pool_dims[0] = sum(args.enc_dims) - args.x_dim 
    args.embed_dims[0] = sum(args.enc_dims) - args.x_dim - args.kappa 
    args.embed_dims[-1] = args.embed_dims[0]
    
parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
args = parser.parse_args()
args.manifold_pool = args.manifold
set_dims(args)
