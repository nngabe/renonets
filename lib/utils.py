import os
import time
import glob
import pickle
import jax
import equinox as eqx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as onp
import jax.numpy as jnp
from argparse import Namespace

from nn.models.cosynn import COSYNN, _forward

def save_model(model, log, path='../eqx_models/', stamp=None):
    if not os.path.exists(path): os.mkdir(path)
    stamp = stamp if stamp else str(int(time.time())) 
    with open(path + f'log_{stamp}.pkl','wb') as f: 
        pickle.dump(log,f)
    eqx.tree_serialise_leaves(path + f'/cosynn_{stamp}.eqx', model)

def read_model(args):
    if isinstance(args,dict):
        args = Namespace(**args)
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/cosynn*{args.log_path}*')[0]
    elif args.log_path:
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/cosynn*{args.log_path}*')[0]
        with open(args_path, 'rb') as f: data = pickle.load(f)
        args_load = Namespace(**data['args'])
        #print(args, args_load)
        #args = Namespace(**args) #args_load
        for k in args.__dict__.keys():
            if k not in ['lr', 'epochs', 'weight_decay', 'max_norm', 'opt_study', 'w_data', 'w_pde', 'verbose']:
                setattr(args, k, getattr(args_load,k))
    else:
        print('need type(args) == dict or args.log_path == True !')
        raise
    args.enc_init = 1
    args.dec_init = 1
    args.pde_init = 2
    model = COSYNN(args)
    model = eqx.tree_deserialise_leaves(param_path, model)
    return model, args
