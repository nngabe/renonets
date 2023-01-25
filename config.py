import argparse

from nn.utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.005, 'learning rate'),
        'dropout': (0.02, 'dropout probability'),
        'epochs': (20000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (100, 'how often to compute print train/val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'max-norm': (100., 'max norm for gradient clipping, or None for no gradient clipping'),
    },
    'model_config': {
        # init flags for neural nets
        'enc_init': (1, 'flag indicating whether the encoder remains to be init-ed or not.'),
        'dec_init': (1, 'flag indicating whether the decoder remains to be init-ed or not.'),
        'pde_init': (2, 'flag indicating number of pde functions which remain to be init-ed.'),
        
        # loss weights
        'w_data': (1., 'weight for data loss'),
        'w_pde': (10., 'weight for pde loss'),

        # which layers use time encodings and what dim should encodings be
        'time_enc': ([0,1,1], 'whether to insert time encoding in encoder, decoder, and pde functions, respectively.'),
        'time_dim': (2, 'dimension of time embedding'), 
       
        # input/output sizes
        'kappa': (60, 'size of lookback window used as input to encoder'),
        'tau_max': (500, 'maximum steps ahead forecast'),
        'tau_num': (10, 'number of tau steps for each training bundle'),
        # specify models. pde function layers are the same as the decoder layers by default.
        'encoder': ('HGCN', 'which encoder to use, can be any of [MLP, HNN, GCN, GAT, HGCN]'),
        'decoder': ('HNN', 'which decoder to use, can be any of [MLP, HNN, GCN, GAT, HGCN]'),
        'pde': ('neural_burgers', 'which PDE to use for the PINN loss'),
        
        # dims of neural nets. -1 will be inferred based on args.skip and args.time_enc. 
        'enc_dims': ([-1,24,5], 'dimensions of encoder layers'),
        'dec_dims': ([-1,256,256,1],'dimensions of decoder layers'),
        'pde_dims': ([-1,192,192,1], 'dimensions of each pde layers'),
        
        #activations for each network
        'act_enc': ('silu', 'which activation function to use (or None for no activation)'),
        'act_dec': ('silu', 'which activation function to use (or None for no activation)'),
        'act_pde': ('silu', 'which activation function to use (or None for no activation)'),
       
        # additional params for layer specification
        'bias': (1, 'whether to use bias in layers or not'),
        'skip': (1, 'whether to use skip connections or not. set to 0 after encoder init.'),
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        
        # graph encoder params
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'data-path': ('../data_cosynn/gels_499_k2.csv', 'path for timeseries data'),
        'adj-path': ('../data_cosynn/adj_499.csv', 'path for adjacency matrix'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
    #    'normalize-feats': (1, 'whether to normalize input node features'),
    #    'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
args = parser.parse_args()
args.enc_dims[0] = args.kappa
if args.skip: 
    args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim * 2
    args.pde_dims[0] = args.dec_dims[0]
else: 
    args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim * 2
    args.pde_dims[0] = args.dec_dims[0]
