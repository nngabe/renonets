import warnings
warnings.filterwarnings('ignore')

import sys
from typing import Any, Optional, Sequence, Tuple, Union

#from hgcn import optimizers
from hgcn.models import models
from pde import pdes
from config import parser

import numpy as onp
import pandas as pd

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

#jax.config.update('jax_debug_nans',True)

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

    def __call__(self, x, t, adj):
        z_x = self.encoder(x, adj)
        #z,x = z_x[:,:-model.x_dim], z_x[:,-model.x_dim:]
        z,x = jnp.split(z_x, [ z_x.shape[1] - self.x_dim ], axis=1)
        t = t*jnp.ones((x.shape[0],1))
        t = jax.vmap(self.time_encode)(t)
        txz = jnp.concatenate([t, x, z], axis=1)
        return jax.vmap(self.decoder)(txz), txz


# functions for computing losses and gradients
@eqx.filter_value_and_grad(has_aux = True)
def decoder_val_grad(t_x, z, model):
    t,x = t_x[:1], t_x[1:] 
    t = model.time_encode(t)
    txz = jnp.concatenate([t,x,z], axis=0)
    pred = model.decoder(txz).reshape(())
    return pred, txz

def compute_val_grad(x0, adj, t, model, on_axis=False):
    z_x = model.encoder(x0, adj)
    z = z_x[:,:-model.x_dim]
    if on_axis: x = jnp.ones_like(z_x[:,-model.x_dim:]) 
    else: x = z_x[:,-model.x_dim:] #z,x = jnp.split(z_x, [ z_x.shape[1] - model.x_dim ], axis=1)
    t = t*jnp.ones((x.shape[0],1))
    t_x = jnp.concatenate([t, x], axis=1)
    dvg = lambda t_x, z: decoder_val_grad(t_x, z, model)
    return jax.vmap(dvg)(t_x, z)

#@eqx.filter_value_and_grad
def compute_loss(model, x0, adj, t, y, w):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    loss_data = jax.lax.square(u - y).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum()
    return w[0] * loss_data + w[1] * loss_pde

@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_batch(model, xb, adj, tb, yb, w):
    lbatch = lambda x,t,y: compute_loss(model, x, adj, t, y, w)
    return jnp.mean(jax.vmap(lbatch)(xb,tb,yb))

@eqx.filter_jit
def make_step(grads, model, opt_state):
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model,updates)
    return model, opt_state

# utility functions for reporting or inference
@eqx.filter_jit
def compute_loss_terms(model, x0, adj, t, y, w):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, model)
    grad_t, grad_x = grad_tx[:,0], grad_tx[:,1:]
    loss_data = jax.lax.square(u - y).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum()    
    return loss_data, loss_pde
clt = lambda xi,ti,yi: [loss.mean() for loss in jax.vmap(lambda x,t,y: compute_loss_terms(model, x, adj, t, y, w))(xi,ti,yi)]

if __name__ == '__main__':
    args = parser.parse_args()
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim
        args.pde_dims[0] = args.dec_dims[0]
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim
        args.pde_dims[0] = args.dec_dims[0]

    T = 2000
    n = 100
    tau = 120
    x = jnp.zeros([n, args.kappa + T]) 
    adj = jax.random.randint(prng(),(2,n),0,n)
    

    model = COSYNN(args)
    schedule = optax.warmup_exponential_decay_schedule(args.lr, peak_value=args.lr, warmup_steps=10000,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm),optax.adam(schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
 
    w = jnp.array([1., 1.]) 
    h = lambda t: .01 * t ** 2. + t
    def _batch(x,idx):
        xi = lambda i: x.at[:,i:i+model.kappa].get()
        xb = jnp.array([xi(i) for i in idx])
        return xb
    
    for i in range(args.epochs):
        ti = jax.random.randint(prng(), (50, 1), 1, T).astype(jnp.float32)
        yi = h(ti) * jnp.ones(x.shape[0])
        idx = ti.astype(int).flatten()
        xi = _batch(x, idx)
        loss, grad = loss_batch(model, xi, adj, ti, yi, w)
        #sys.exit(0)
        model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:
            ti = jnp.linspace(0.,T,1000).reshape(-1,1)
            yi = h(ti) * jnp.ones(x.shape[0])
            idx = ti.astype(int).flatten() 
            xi = _batch(x, idx)
            loss, grad = loss_batch(model, xi, adj, ti, yi, w) 
            model, opt_state = make_step(grad, model, opt_state)
            loss_data, loss_pde = clt(xi, ti, yi)
            print(f'{i}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')

    def plot(u,y,i=0,j=0):
        u = onp.array(u)
        y = onp.array(y)
        if j==0: j = u.shape[0]
        ax = pd.DataFrame(u).iloc[i:j,:3].plot()
        idx = range(i,j)
        pd.DataFrame(y[idx],index=idx).plot(ax=ax)
    
    m = lambda x,t: model(x,t,adj)
    tp = jnp.linspace(0,T,T+1)
    xp = _batch(x,tp.astype(int).flatten())
    res = jax.vmap(m)(xp,tp)
    u = res[0].squeeze()
    plot(u, h(tp))

