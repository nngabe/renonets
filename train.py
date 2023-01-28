import warnings
warnings.filterwarnings('ignore')

import os
import sys
import copy
from typing import Any, Optional, Sequence, Tuple, Union

from nn.models import models
from pde import pdes
from config import parser

import numpy as onp
import pandas as pd

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from lib import utils
from lib.graph_utils import batch_graph, remove_nodes, add_self_loops

prng = lambda: jax.random.PRNGKey(0)

class COSYNN(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    w_data: jnp.float32
    w_pde: jnp.float32
    x_dim: int
    t_dim: int
    kappa: int
    scalers: jnp.ndarray
    k_lin: Tuple[int,int]
    k_log: Tuple[int,int]

    def __init__(self, args):
        super(COSYNN, self).__init__()
        self.encoder = getattr(models, args.encoder)(args)
        self.decoder = getattr(models, args.decoder)(args)
        self.pde = getattr(pdes, args.pde)(args)
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.x_dim = args.enc_dims[-1]
        self.t_dim = args.time_dim
        self.kappa = args.kappa
        self.scalers = jnp.array([10.**i for i in range(1, 1 + args.time_dim)])
        start = 1,1
        self.k_lin = start[0], start[0] + self.t_dim//2
        self.k_log = start[1], start[1] + self.t_dim - (self.k_lin[1] - self.k_lin[0])

    def time_encode(self, t):
        t = t/self.scalers
        t_lin = t[self.k_lin[0]:self.k_lin[1]]
        t_log = jnp.log( t[self.k_log[0]:self.k_log[1]] + 1. )
        t = jnp.concatenate([t_lin, t_log])
        return t

# functions for computing losses and gradients
@eqx.filter_value_and_grad(has_aux = True)
def decoder_val_grad(t_x, tau, z, model):
    t,x = t_x[:1], t_x[1:]
    t = model.time_encode(t)
    tau = model.time_encode(tau)
    txz = jnp.concatenate([t,tau,x,z], axis=0)
    pred = model.decoder(txz).reshape(())
    return pred, txz

def compute_val_grad(x0, adj, t, tau, model):
    z_x = model.encoder(x0, adj)
    z = z_x[:,:-model.x_dim]
    x = 0. * z_x[:,-model.x_dim:]
    t = t*jnp.ones((x.shape[0],1))
    t_x = jnp.concatenate([t, x], axis=1)
    dvg = lambda t_x, z: decoder_val_grad(t_x, tau, z, model)
    return jax.vmap(dvg)(t_x, z)

def compute_loss(model, x0, adj, t, tau, y, p = 0.8):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    mask = jax.random.bernoulli(prng(), p=p, shape=u.shape)
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
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,0], grad_tx[:,1:]
    loss_data = jax.lax.square(u - y).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum()    
    return loss_data, loss_pde
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
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model,updates)
    return model, opt_state


if __name__ == '__main__':
    args = parser.parse_args()
    log = {}
    log['args'] = vars(args)

    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    n,T = x.shape
    tau = jnp.array([60])

    adj = add_self_loops(adj)
    idx_test, adj_test = batch_graph(0, adj)
    idx_train, adj_train = remove_nodes(idx_test, adj)

    model = COSYNN(args)
    sys.exit() 
    print()
    print(f'MODULE: MODEL[DIMS](curv)')
    print(f' encoder: {args.encoder}{args.enc_dims}({args.c})')
    print(f' decoder: {args.decoder}{args.dec_dims}({args.c})')
    print(f' pde: {args.pde}/{args.decoder}{args.pde_dims}({args.c})')
    print(f' time_enc: linlog[{args.time_dim}]')
    print()

    schedule = optax.warmup_exponential_decay_schedule(args.lr, peak_value=args.lr, warmup_steps=args.epochs//10,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm),optax.adam(schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def _batch(x,idx):
        win = jnp.arange(1 - args.kappa, 1, 1)
        xb = x.at[:, idx + win].get()
        xb = jnp.swapaxes(xb,0,1)
        return xb

    def _taus(i, size=args.tau_num, tau_max=args.tau_max):
        taus = jnp.array(10 * onp.random.exponential(2. + i/500., size), dtype=jnp.int32)
        taus = jnp.clip(taus, 1, tau_max)
        return taus
      
    log['loss'] = {}
    for i in range(args.epochs):
        ti = jax.random.randint(prng(), (50, 1), args.kappa, T - args.tau_max).astype(jnp.float32)
        idx = ti.astype(int)
        taus = _taus(i)
        bundles = idx + taus
        yi = x[:,bundles].T
        xi = _batch(x, idx)
        sys.exit(0)
        loss, grad = loss_bundle(model, xi, adj, ti, taus, yi)
        model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:
            ti = jnp.linspace(args.kappa, T - args.tau_max , 100).reshape(-1,1)
            idx = ti.astype(int)
            taus = jnp.arange(15,121,15)
            bundles = idx + taus
            yi = x[:,bundles].T
            xi = _batch(x, idx)
            loss, grad = loss_bundle(model, xi, adj, ti, taus, yi)
            model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:   
            model = eqx.tree_inference(model, value=True)
            terms = compute_bundle_terms(model, xi, adj, ti, taus, yi)
            loss_data, loss_pde = terms[0].mean(), terms[1].mean()
            log['loss'][i] = [loss_data, loss_pde]
            print(f'{i:04d}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')
        if i%(3*args.log_freq) == 0 and i < args.epochs / 3000: 
            model = eqx.tree_inference(model, value=False) 
    
    utils.save_model(model,log)
