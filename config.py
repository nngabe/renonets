import argparse
import glob

from nn.utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (1e-5, 'learning rate'),
        'dropout': (0., 'dropout probability'),
        'epochs': (60001, 'number of epochs to train for'),
        'slaw': (False, 'whether to use scaled loss approximate weighting (SLAW)'),
        'weight_decay': (1e-3, 'l2 regularization strength'),
        'beta': (0.99, 'moving average coefficient for SLAW'),
        'log_freq': (50, 'how often to compute print train/val metrics (in epochs)'),
        'max_norm': (1.0, 'max norm for gradient clipping, or None for no gradient clipping'),
        'verbose': (True, 'print training data to console'),
        'opt_study': (False, 'whether to run a hyperparameter optimization study or not'),
        'num_col': (10, 'number of colocation points in the time domain'),
        'batch_size': (128, 'number of nodes in test and batch graphs'),
        'batch_red': (2, 'factor of reduction for batch size'),
        'pool_red': (4, 'factor of reduction for each pooling step'),
    },
    'model_config': {

        # init flags for neural nets
        'enc_init': (1, 'flag indicating whether the encoder remains to be init-ed or not.'),
        'dec_init': (1, 'flag indicating whether the decoder remains to be init-ed or not.'),
        'pde_init': (2, 'flag indicating number of pde functions which remain to be init-ed.'),
        'pool_init': (2, 'flag indicating number of pooling modules which remain to be init-ed.'),
        'embed_init': (2, 'flag indicating number of embedding modules which remain to be init-ed.'), 

        # loss weights
        'w_data': (1., 'weight for data loss.'),
        'w_pde': (1e+3, 'weight for pde loss.'),
        'w_gpde': (1e+6, 'weight for gpde loss.'),
        'w_ent': (1e-1, 'weight for assignment matrix entropy loss.'),
        'F_max': (1., 'max value of convective term'),
        'v_max': (.0, 'max value of viscous term.'),
        'input_scaler': (1., 'rescaling of input'),
        'rep_scaler': (1., 'rescaling of graph features'),

        # which layers use time encodings and what dim should encodings be
        'time_enc': ([0,1,1], 'whether to insert time encoding in encoder, decoder, and pde functions, respectively.'),
        'time_dim': (1, 'dimension of time embedding'), 
        'x_dim': (1, 'dimension of differentiable coordinates for PDE'),

        # input/output sizes
        'fe': (0, 'encode features or not'),
        'kappa': (64, 'size of lookback window used as input to encoder'),
        'f_dim': (64, 'size of fourier feature encoding'), 
        'tau_max': (1, 'maximum steps ahead forecast'),
        
        # specify models. pde function layers are the same as the decoder layers by default.
        'encoder': ('HGCN', 'which encoder to use'),
        'decoder': ('MLP', 'which decoder to use'),
        'pde': ('neural_burgers', 'which PDE to use for the PINN loss'),
        'pool': ('HGCN', 'which model to compute coarsening matrices'),
        'func_space': ('PowerSeries', 'function space for DeepOnet.'),
        'p_basis': (20, 'size of DeepOnet basis'),

        # dims of neural nets. -1 will be inferred based on args.skip and args.time_enc. 
        'enc_width': (32, 'dimensions of encoder layers'),
        'dec_width': (512,'dimensions of decoder layers'),
        'pde_width': (512, 'dimensions of each pde layers'),
        'pool_width': (256, 'dimensions of each pde layers'),
        'enc_depth': (2, 'dimensions of encoder layers'),
        'dec_depth': (3,'dimensions of decoder layers'),
        'pde_depth': (3, 'dimensions of each pde layers'),
        'pool_depth': (2, 'dimensions of each pooling layer'),
        'enc_dims': ([-1,96,-1], 'dimensions of encoder layers'),
        'dec_dims': ([-1,256,256,-1],'dimensions of decoder layers'),
        'pde_dims': ([-1,256,256,1], 'dimensions of each pde layers'),
        'pool_dims': ([-1,96,-1], 'dimesions of pooling layers.'), 
        'embed_dims': ([-1,96,-1], 'dimensions of embedding layers.'),

        #activations for each network
        'act': ('gelu', 'which activation function to use (or None for no activation)'),
       
        # additional params for layer specification
        'bias': (1, 'whether to use bias in layers or not'),
        'skip': (1, 'whether to use concat skip connections in encoder.'),
        'res': (0, 'whether to use sum skip connections or not.'),
        'manifold': ('Hyperboloid', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        
        # graph encoder params
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'affine': (True, 'affine transformation in layernorm'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'use_att_pool': (0, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'path': ('2990_c1', 'snippet from which to infer data path'),
        'log_path': (None, 'snippet from which to infer log/model path.'),
        'test_prop': (0.1, 'proportion of test nodes for forecasting'),
    }
}

def set_dims(args):
    if args.decoder=='MHA': 
        args.dec_width = args.enc_width 
        args.pde_width = args.enc_width
        args.dec_depth = 1 
        args.pde_depth = 1
        args.num_col = 4
    args.enc_dims[0] = args.f_dim * 2 if args.fe else args.kappa
    args.enc_dims[-1] = args.enc_width 
    args.dec_dims[-1] = args.x_dim
    args.enc_dims[1:-1] = (args.enc_depth-1) * [args.enc_width]
    args.dec_dims[1:-1] = (args.dec_depth-1) * [args.dec_width]
    args.pde_dims[1:-1] = (args.pde_depth-1) * [args.pde_width]
    args.pool_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]
    args.embed_dims[1:-1] = (args.pool_depth-1) * [args.pool_width]
    if args.res:
        enc_out = args.enc_dims[-1]
        args.kappa = 0
        args.dec_dims[0] = enc_out + args.time_enc[1] * args.time_dim
    elif args.skip: 
        enc_out = sum(args.enc_dims)
        args.dec_dims[0] = enc_out + args.time_enc[1] * args.time_dim 
    else: 
        enc_out = args.enc_dims[-1]
        args.dec_dims[0] = enc_out + args.time_enc[1] * args.time_dim 
    
    args.pde_dims[0] = args.dec_dims[0] 
    args.pool_dims[0] = enc_out - args.x_dim + args.time_dim 
    args.embed_dims[0] = enc_out - args.x_dim - args.kappa + args.time_dim 
    args.embed_dims[-1] = args.embed_dims[0] - args.time_dim
    
parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
args = parser.parse_args()
args.manifold_pool = args.manifold
set_dims(args)
