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

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.2'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

prng = lambda: jax.random.PRNGKey(onp.random.randint(1e+3))

class COSYNN(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
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

def compute_loss(model, x0, adj, t, tau, y, w, p = 0.8):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    #mask = jax.random.bernoulli(prng(),p=p,shape=u.shape)
    loss_data = (jax.lax.square(u - y)).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = (jax.lax.square(resid)).sum()
    return w[0] * loss_data + w[1] * loss_pde

@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_batch(model, xb, adj, tb, tau, yb, w):
    lbatch = lambda x,t,y: compute_loss(model, x, adj, t, tau, y, w)
    return jnp.mean(jax.vmap(lbatch)(xb,tb,yb))

@eqx.filter_jit
def make_step(grads, model, opt_state):
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model,updates)
    return model, opt_state

# utility functions for reporting or inference
@eqx.filter_jit
def compute_loss_terms(model, x0, adj, t, tau, y, w):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,0], grad_tx[:,1:]
    loss_data = jax.lax.square(u - y).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum()    
    return loss_data, loss_pde
clt = lambda model,xi,ti,yi: [loss.mean() for loss in jax.vmap(lambda x,t,y: compute_loss_terms(model, x, adj, t, tau, y, w))(xi,ti,yi)]

if __name__ == '__main__':
    args = parser.parse_args()
    log = {}
    log['args'] = vars(args)

    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    n,T = x.shape
    tau = jnp.array([60])

    model = COSYNN(args)
    
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
 
    w = jnp.array([1., 10.]) 
     
    def _batch(x,idx):
        xi = lambda i: x.at[:,i:i+model.kappa].get()
        xb = jnp.array([xi(i) for i in idx])
        return xb
    
    log['loss'] = {}
    for i in range(args.epochs):
        ti = jax.random.randint(prng(), (50, 1), 1, T-model.kappa).astype(jnp.float32)
        idx = ti.astype(int).flatten()
        yi = x[:,idx+tau].T 
        xi = _batch(x, idx)
        loss, grad = loss_batch(model, xi, adj, ti, tau, yi, w)
        model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:
            ti = jnp.linspace(0., T - model.kappa, 1000).reshape(-1,1)
            idx = ti.astype(int).flatten() 
            yi = x[:,idx+tau].T 
            xi = _batch(x, idx)
            loss, grad = loss_batch(model, xi, adj, ti, tau, yi, w) 
            model, opt_state = make_step(grad, model, opt_state)
            
            model = eqx.tree_inference(model, value=True)
            loss_data, loss_pde = clt(model, xi, ti, yi)
            log['loss'][i] = [loss_data, loss_pde]
            print(f'{i:04d}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')
            if i%(3*args.log_freq) == 0 and i < args.epochs * .333: 
                model = eqx.tree_inference(model, value=False) 
    
    utils.save_model(model,log)
