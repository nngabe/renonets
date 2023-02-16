#import warnings
#warnings.filterwarnings('ignore')

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

from nn import manifolds
from nn.models import models
from pde import pdes
from config import parser

from lib import utils
from lib.graph_utils import subgraph, random_subgraph, louvain_subgraph, add_self_loops, sup_power_of_two, pad_graph


prng = lambda i=0: jax.random.PRNGKey(i)

class COSYNN(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32
    w_data: jnp.float32
    w_pde: jnp.float32
    x_dim: int
    t_dim: int
    kappa: int
    scalers: Dict[str, jnp.ndarray]
    k_lin: Tuple[int,int]
    k_log: Tuple[int,int]

    def __init__(self, args):
        super(COSYNN, self).__init__()
        self.encoder = getattr(models, args.encoder)(args)
        self.decoder = getattr(models, args.decoder)(args)
        self.pde = getattr(pdes, args.pde)(args)
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.x_dim = args.x_dim
        self.t_dim = args.time_dim
        self.kappa = args.kappa
        self.scalers = {'t_lin': 10. ** jnp.arange(2, 2*self.t_dim, 1, dtype=jnp.float32),
                        't_log': 10. ** jnp.arange(-2, 2*self.t_dim, 1, dtype=jnp.float32),
                        'reps' : jnp.array([20.]),
                        'input': jnp.array([1e-1]) 
                       }
        self.k_lin = self.t_dim//2
        self.k_log = self.t_dim - self.k_lin

    def time_encode(self, t):
        t_lin = t / jnp.clip(self.scalers['t_lin'], 1e-7)
        t_log = t / jnp.clip(self.scalers['t_log'], 1e-7)
        t_lin = t_lin[:self.k_lin]
        t_log = jnp.log( t_log[:self.k_log] + 1. )
        t = jnp.concatenate([t_lin, t_log])
        return t

    def logmap0(self, u):
        return self.manifold.logmap0(u,self.c)

# functions for computing losses and gradients
@eqx.filter_value_and_grad(has_aux = True)
def decoder_val_grad(t_x, tau, z, model):
    t,x = t_x[:1], t_x[1:]
    t = model.time_encode(t)
    tau = model.time_encode(tau * 1.)
    z0 = z[:model.kappa]
    zi = z[model.kappa:]
    zp = model.logmap0(zi) * model.scalers['reps'][0]
    txz = jnp.concatenate([t,tau,x,z0,zp], axis=0)
    pred = model.decoder(txz).reshape(())
    return pred, (zi, txz)

def compute_val_grad(x0, adj, t, tau, model):
    x0 *= model.scalers['input'][0]
    z_x = model.encoder(x0, adj)
    z = z_x[:,:-model.x_dim]
    x = 0. * z_x[:,-model.x_dim:]
    t = t*jnp.ones((x.shape[0],1))
    t_x = jnp.concatenate([t, x], axis=1)
    dvg = lambda t_x, z: decoder_val_grad(t_x, tau, z, model)
    return jax.vmap(dvg)(t_x, z)

def compute_loss(model, x0, adj, t, tau, y):
    (u, (zi, txz)), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    mask = ( jnp.abs(x0[:,0] - 10.) > 1e-18)
    loss_data = jax.lax.square(u - y)
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).flatten()
    loss_data *= mask 
    loss_pde *= mask
    return model.w_data * loss_data.sum() + model.w_pde * loss_pde.sum()

def loss_batch(model, xb, adj, tb, tau, yb):
    closs = lambda x,t,y: compute_loss(model, x, adj, t, tau, y)
    return jnp.mean(jax.vmap(closs)(xb,tb,yb))

@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_bundle(model, xb, adj, tb, taus, yb):
    lbatch = lambda tau,y: loss_batch(model, xb, adj, tb, tau, y)
    return jnp.mean(jax.vmap(lbatch)(taus,yb))

# utility functions for reporting or inference
def compute_loss_terms(model, x0, adj, t, tau, y):
    (u, (zi,txz)), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,0], grad_tx[:,1:]
    mask = ( jnp.abs(x0[:,0] - 10.) > 1e-18)
    loss_data = jax.lax.square(u - y)
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).flatten()
    loss_data *= mask
    loss_pde *= mask
    return loss_data.sum(), loss_pde.sum()

def compute_batch_terms(model, xb, adj, tb, tau, yb):
    clt = lambda x,t,y: compute_loss_terms(model, x, adj, t, tau, y)
    return jax.vmap(clt)(xb,tb,yb)

@eqx.filter_jit
def compute_bundle_terms(model, xb, adj, tb, taus, yb):
    cbt = lambda tau,y: compute_batch_terms(model, xb, adj, tb, tau, y)
    return jax.vmap(cbt)(taus,yb)

# updating model parameters and optimizer state
@eqx.filter_jit
def make_step(grads, model, opt_state):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_path = glob.glob(f'../data_cosynn/gels*{args.path}*')[0]
    args.adj_path = glob.glob(f'../data_cosynn/adj*{args.path.split("_")[:-1]}*')[0]

    log = {}
    log['args'] = vars(args)

    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    n,T = x.shape

    adj = add_self_loops(adj)
    x_test, adj_test, idx_test = louvain_subgraph(x, adj, batch_size=n//10)
    idx_train = jnp.where(jnp.ones(n, dtype=jnp.int32).at[idx_test].set(0))[0]    
    x_train, adj_train, idx_train = subgraph(idx_train, x, adj)
    
    print(f'\nx[train] = {x[idx_train].shape}, adj[train] = {adj_train.shape}')
    print(f'x[test]  = {x[idx_test].shape},  adj[test]  = {adj_test.shape}')

    model = COSYNN(args)
     
    print(f'\nMODULE: MODEL[DIMS](curv)')
    print(f' encoder: {args.encoder}{args.enc_dims}({args.c})')
    print(f' decoder: {args.decoder}{args.dec_dims}({args.c})')
    print(f' pde: {args.pde}/{args.decoder}{args.pde_dims}({args.c})')
    print(f' time_enc: linlog[{args.time_dim}]\n')
    
    schedule = optax.warmup_exponential_decay_schedule(args.lr/1e+1, peak_value=args.lr, warmup_steps=args.epochs//10,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+1)
    optim = optax.chain(optax.clip(args.max_norm), optax.adamw(learning_rate=schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def _batch(x,idx):
        win = jnp.arange(1 - args.kappa, 1, 1)
        xb = x.at[:, idx + win].get()
        xb = jnp.swapaxes(xb,0,1)
        return xb

    def _taus(i, size=args.tau_num, tau_max=args.tau_max):
        taus = jax.random.randint(prng(i), (size,), 1, tau_max) #jnp.array(10 * onp.random.exponential(2. + i/1000., size), dtype=jnp.int32)
        taus = jnp.clip(taus, 1, tau_max)
        return taus
    
    stamp = str(int(time.time()))
    log['loss'] = {}
    x, adj, _   = random_subgraph(x_train, adj_train, batch_size=n//2, seed=0)
    for i in range(args.epochs):
        ti = jax.random.randint(prng(i), (50, 1), args.kappa, T - args.tau_max).astype(jnp.float32)
        idx = ti.astype(int)
        taus = _taus(i)
        bundles = idx + taus
        yi = x[:,bundles].T
        xi = _batch(x, idx)
        loss, grad = loss_bundle(model, xi, adj, ti, taus, yi)
        grad = jax.tree_map(lambda x: 0. if jnp.isnan(x).any() else x, grad) 
        if jnp.isnan(loss):
            print('nan loss! breaking...')
            sys.exit()
        
        model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:
            x, adj = x_test, adj_test
            
            ti = jnp.linspace(args.kappa, T - args.tau_max , 100).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(1, args.tau_max, 10).astype(int)
            bundles = idx + taus
            yi = x[:,bundles].T
            xi = _batch(x, idx)
            
            terms = compute_bundle_terms(model, xi, adj, ti, taus, yi)
            loss_data, loss_pde = terms[0].mean(), terms[1].mean()
            log['loss'][i] = [loss_data, loss_pde]
            print(f'{i:04d}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')
            x, adj, _   = random_subgraph(x_train, adj_train, batch_size=n//2, seed=i)
        if i % args.log_freq * 10 == 0:
            utils.save_model(model, log, stamp=stamp)
    
    utils.save_model(model,log, stamp=stamp)
