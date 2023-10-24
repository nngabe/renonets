import os
import time
import math
import glob
import pickle
import jax
import equinox as eqx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as onp
import jax.numpy as jnp
from argparse import Namespace

from nn.models.renonet import RenONet, _forward
def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
  out, in_ = weight.shape
  stddev = math.sqrt(1 / in_)
  return stddev * jax.random.truncated_normal(key, lower=-2, upper=2, shape=weight.shape)

def init_he(model, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  is_bias = lambda x: x.bias!=None if is_linear(x) else False
  get_weights = lambda m: [x.weight  for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
  get_biases = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_bias) if is_bias(x)]
  weights = get_weights(model)
  biases = get_biases(model)
  new_weights = [trunc_init(weight, subkey) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_biases = [jnp.zeros(b.shape) for b in biases]
  model = eqx.tree_at(get_weights, model, new_weights)
  model = eqx.tree_at(get_biases, model, new_biases)
  return model


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
            continue
            if k not in ['lr', 'epochs', 'weight_decay', 'max_norm', 'opt_study', 'w_data', 'w_pde', 'verbose']:
                setattr(args, k, getattr(args_load,k))
    else:
        print('need type(args) == dict or args.log_path == True !')
        raise
    #args.enc_init = 1
    #args.dec_init = 1
    #args.pde_init = 2
    #args.pool_init = 3
    #args.embed_init = 3
    model = RenONet(args)
    model = eqx.tree_deserialise_leaves(param_path, model)
    return model, args
