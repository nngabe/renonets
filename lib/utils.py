from typing import Any, BinaryIO, Callable, Optional, Union
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

def conservative_filter_spec(f: BinaryIO, x: Any) -> Any:

    try:
        if isinstance(x, jnp.ndarray):
            return jnp.load(f)
        elif isinstance(x, np.ndarray):
            return np.load(f)
        elif isinstance(x, (bool, float, complex, int)):
            return np.load(f).item()
        elif isinstance(x, experimental.StateIndex):
            # Make a new StateIndex. If we happen to load some state then we don't
            # want to affect the `like` as a side-effect.
            y = experimental.StateIndex(inference=x.inference)
            saved_value = np.load(f).item()
            assert isinstance(saved_value, bool)
            if saved_value:
                is_array = np.load(f).item()
                assert isinstance(is_array, bool)
                if is_array:
                    value = jnp.load(f)
                else:
                    tuple_length = np.load(f).item()
                    assert isinstance(tuple_length, int)
                    value = tuple(jnp.load(f) for _ in range(tuple_length))
                experimental.set_state(y, value)
            return y
        else:
            return x

    except:
        return x



def read_model(args,dropout=0.):
    if isinstance(args,dict):
        args = Namespace(**args)
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/cosynn*{args.log_path}*')[0]
    elif args.log_path:
        args_path = glob.glob(f'../eqx_models/log*{args.log_path}*')[0]
        param_path = glob.glob(f'../eqx_models/cosynn*{args.log_path}*')[0]
        with open(args_path, 'rb') as f: data = pickle.load(f)
        args_load = Namespace(**data['args'])
        print(args, args_load)
        args = Namespace(**args)
        # replace all args with model from args.log_path except training args
        for k in args.__dict__.keys():
            if k not in ['lr', 'epochs', 'weight_decay', 'max_norm', 'opt_study', 'w_data', 'w_pde', 'verbose']:
                setattr(args, k, getattr(args_load,k))
    else:
        print('need type(args) == dict or args.log_path == True !')
        raise
    args.enc_init = 1
    args.dec_init = 1
    args.pde_init = 2
    args.dropout = dropout
    model = COSYNN(args)
    model = eqx.tree_deserialise_leaves(param_path, model, filter_spec=conservative_filter_spec)
    f = lambda x: eqx.nn.Dropout(dropout) if isinstance(x,eqx.nn.Dropout) else x
    is_leaf = lambda x: isinstance(x,eqx.nn.Dropout)
    model = jax.tree_map(f, model, is_leaf=is_leaf)
    return model, args
