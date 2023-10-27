import warnings
warnings.filterwarnings('ignore')

import os
import sys
import copy
import time
import glob
import numpy as onp
import pandas as pd
from typing import Any, Optional, Sequence, Tuple, Union, Dict

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from nn.models.renonet import RenONet, loss_scan, loss_terms, make_step
from config import parser, set_dims

from lib import utils
from lib.graph_utils import subgraph, random_subgraph, louvain_subgraph, add_self_loops, sup_power_of_two, pad_graph
from lib.positional_encoding import pe_path_from, pos_enc

jax.config.update("jax_enable_x64", True)
prng = lambda i=0: jax.random.PRNGKey(i)

if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = glob.glob(f'../data_cosynn/gels*{args.path}*')[0]
    args.adj_path = glob.glob(f'../data_cosynn/adj*{args.path.split("_")[0]}*')[0]
    args.pe_path = pe_path_from(args.adj_path)

    print(f'\n w[data,pde,gpde,ent] = {args.w_data:.0e},{args.w_pde:.0e},{args.w_gpde:.0e},{args.w_ent:.0e}')
    print(f'\n data path: {args.data_path}\n adj path: {args.adj_path}\n\n')
    
    pe = pos_enc(args, le_size=args.le_size, rw_size=args.rw_size, n2v_size=args.n2v_size, norm=args.pe_norm, use_cached=args.use_cached)
    pe = pe/pe.max()
    args.pe_dim = pe.shape[1]
    args = set_dims(args)    

    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    adj = add_self_loops(adj)
    x = pd.read_csv(args.data_path, index_col=0).dropna().T
    x = jnp.array(x.to_numpy())
    x = x/x.max() #(x - x.min())/(x.max() - x.min())
    n,T = x.shape

    x_test, adj_test, pe_test, idx_test = random_subgraph(x, adj, pe, batch_size=args.batch_size, key=prng(0))
    idx_train = jnp.where(jnp.ones(n, dtype=jnp.int32).at[idx_test].set(0))[0]    
    x_train, adj_train, pe_train = x, adj, pe #subgraph(idx_train, x, adj)

    args.pool_dims[-1] = 128 #sup_power_of_two(2 * n//args.pool_red)
    if args.log_path:
        model, args = utils.read_model(args)
    else:
        model = RenONet(args)
        model = utils.init_he(model, prng(123))

    if args.verbose: 
        print(f'\nMODULE: MODEL[DIMS](curv)')
        print(f' encoder: {args.encoder}{args.enc_dims}({args.c})')
        print(f' decoder: {args.decoder}{args.dec_dims}')
        print(f' pde: {args.pde}/{args.decoder}{args.pde_dims}')
        print(' pool:')
        for i in model.pool.pools.keys(): 
            pdims = args.pool_dims
            pdims[-1] = model.pool_dims[i] #sup_power_of_two(n // args.batch_red) // (args.pool_red)**(i+1)
            print(f'   pool_{i}: {args.pool}{pdims}')
        print(' embed:')
        for i in model.pool.pools.keys(): 
            print(f'   embed_{i}: {args.pool}{args.embed_dims}')
        print(f' time_enc: linlog[{args.time_dim}]\n')


    
    log = {}
    log['args'] = vars(copy.copy(args))
    log['train_index'] = idx_train
    log['test_index'] = idx_test

    if args.verbose:
        print(f'\nx[train] = {x[idx_train].shape}, adj[train] = {adj_train.shape}')
        print(f'x[test]  = {x[idx_test].shape},  adj[test]  = {adj_test.shape}')
    
    schedule = optax.warmup_exponential_decay_schedule(init_value=0., peak_value=args.lr, warmup_steps=args.epochs//100,
                                                        transition_steps=args.epochs, decay_rate=5e-3, end_value=args.lr/1e+2)
    optim = optax.chain(optax.clip(args.max_norm), optax.adamw(learning_rate=schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def _batch(x, pe, idx):
        win = jnp.arange(1 - args.kappa, 1, 1)
        x = x.at[:, idx + win].get()
        x = jnp.swapaxes(x,0,1)
        pe = jnp.tile(pe, (idx.shape[0],1,1))
        xb = jnp.concatenate([pe, x], axis=-1)
        return xb
   
    stamp = str(int(time.time()))
    log['loss'] = {}
    x, adj, pe, _   = random_subgraph(x_train, adj_train, pe_train, batch_size=args.batch_size, key=prng(0))
   
    print(f'\nSLAW: {args.slaw}, dropout: p={args.dropout}, num_col={args.num_col}, \n')
    
    n = args.num_col # number of temporal colocation points
    tic = time.time()
    mode = 1 if args.slaw else 0
    state = {'a': jnp.zeros(4), 'b': jnp.zeros(4)} if args.slaw else None
    key = jax.random.PRNGKey(0)
    model = eqx.tree_inference(model, value=False)
    n_slabs = 100
    slab = jnp.linspace(args.kappa+1000, T-args.tau_max, n_slabs).astype(int)
    for i in range(args.epochs):
        
        key = jax.random.split(key)[0]
        ks = max(n_slabs, int(n_slabs * i/args.epochs / .6))
        ti = jax.random.randint(key, (n, 1), args.kappa, slab[ks]).astype(jnp.float32)
        idx = ti.astype(int)
        taus = jnp.arange(1, 1+args.tau_max, 1)
        bundles = idx + taus
        yi = x[:,bundles]
        yi = jnp.swapaxes(yi,0,1)
        xi = _batch(x, pe, idx)
        (loss, state), grad = loss_scan(model, xi, adj, ti, yi, key=key, mode=mode, state=state)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        
        model, opt_state = make_step(grad, model, opt_state, optim)
        if i % args.log_freq == 0:
            
            x, adj, pe = x_test, adj_test, pe_test
            model = eqx.tree_inference(model, value=True) 

            ti = jnp.linspace(args.kappa, T - args.tau_max , 10).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(1, 1+args.tau_max, 1).astype(int)
            bundles = idx + taus
            yi = x[:,bundles]
            yi = jnp.swapaxes(yi,0,1)
            xi = _batch(x, pe, idx)
            
            terms, _ = loss_terms(model, xi, adj, ti, yi)
            loss = [term.mean() for term in terms]
            log['loss'][i] = [loss[0].item(), loss[1].item(), loss[2].item(), loss[3].item()]
            
            if args.verbose:
                print(f'{i:04d}/{args.epochs} : loss_data = {loss[0]:.2e}, loss_pde = {loss[1]:.2e}, loss_gpde = {loss[2]:.2e}, loss_ent = {loss[3]:.2e}  lr = {schedule(i).item():.4e} (time: {time.time()-tic:.1f} s)')
            tic = time.time()
            
            x, adj, pe, _   = random_subgraph(x_train, adj_train, pe_train, batch_size=args.batch_size, key=prng(i+j))
            model = eqx.tree_inference(model, value=False)
            
        if i % args.log_freq * 100 == 0:
            utils.save_model(model, log, stamp=stamp)

    log['wall_time'] = int(time.time()) - int(stamp) 
    utils.save_model(model, log, stamp=stamp)
