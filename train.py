import warnings
warnings.filterwarnings('ignore')

import sys
from typing import Any, Optional, Sequence, Tuple, Union

#from hgcn import optimizers
from nn.models import models
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

def compute_loss(model, x0, adj, t, tau, y, w):
    (u, txz), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    loss_data = jax.lax.square(u - y).sum()
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum()
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
clt = lambda xi,ti,yi: [loss.mean() for loss in jax.vmap(lambda x,t,y: compute_loss_terms(model, x, adj, t, tau, y, w))(xi,ti,yi)]

if __name__ == '__main__':
    args = parser.parse_args()
    
    args.enc_dims[0] = args.kappa
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim * 2
        args.pde_dims[0] = args.dec_dims[0]
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim * 2
        args.pde_dims[0] = args.dec_dims[0]

    A = pd.read_csv('../data_hpgn/adj_499.csv',index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv('../data_hpgn/gels_499_k2.csv',index_col=0).dropna().to_numpy().T)

    n,T = x.shape
    tau = jnp.array([60])

    model = COSYNN(args)
    print(f'MODULE: MODEL[DIMS](curv)')
    print(f' encoder: {args.encoder}{args.enc_dims}({args.c})')
    print(f' decoder: {args.decoder}{args.dec_dims}({args.c})')
    print(f' pde: {args.pde}/{args.decoder}{args.pde_dims}({args.c})')
    print(f' time_enc: linlog[{args.time_dim}]')

    schedule = optax.warmup_exponential_decay_schedule(args.lr, peak_value=args.lr, warmup_steps=args.epochs//10,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm),optax.adam(schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
 
    w = jnp.array([1., 10.]) 
    #sys.exit(0) 
    def _batch(x,idx):
        xi = lambda i: x.at[:,i:i+model.kappa].get()
        xb = jnp.array([xi(i) for i in idx])
        return xb
    
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
            loss_data, loss_pde = clt(xi, ti, yi)
            print(f'{i}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')

    def mask_plot(y):
        yy = pd.DataFrame(onp.array(y))
        d = (yy.diff()**2. + .4).fillna(1.)
        for i in range(d.shape[0]):
            d.iloc[i] += d.iloc[i-1]
            if d.iloc[i].item()>25.:
                d.iloc[i] = 0.
        mask = (d==0.)
        return yy[mask]

    def plot(u,y,i=0,j=-1,n=4):
        import matplotlib.pyplot as plt
        plt.rcParams['figure.figsize'] = [8,8]; plt.rcParams['font.size'] = 14; plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True
        u = onp.array(u)
        y = onp.array(y)
        if j==-1: j = u.shape[0]
        colors = ['mediumslateblue','r','g','b','cornflowerblue','gold','purple']

        for k in range(n):
            if k == 0 : 
                ax = pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(60).dropna().plot(color=colors[k])
                df = mask_plot(y[i:j,k])
                df.columns = ['_none']
                df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
            elif k < n-1 : 
                pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(60).dropna().plot(ax=ax, color=colors[k])
                df = mask_plot(y[i:j,k])
                df.columns = ['_none']
                df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
            elif k == n-1 :
                pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(60).dropna().plot(ax=ax, color='C7')
                df = mask_plot(y[i:j,k])
                df.columns = ['data']
                df.plot(ax=ax, color='k', marker='o', markersize=5, markerfacecolor='none', linestyle='none')
    
    def call(x0, t, adj):
        z_x = model.encoder(x0, adj)
        z = z_x[:,:-model.x_dim]
        #x = z_x[:,-model.x_dim:]
        x = 0. * z_x[:,-model.x_dim:]
        t = t*jnp.ones((x.shape[0],1))
        t = jax.vmap(model.time_encode)(t)
        txz = jnp.concatenate([t, x, z], axis=1)
        return jax.vmap(model.decoder)(txz), txz
    
    def inference(i=0,j=-1,n=3):
        m = lambda x,t: call(x,t,adj)
        tp = jnp.linspace(0,T-tau,T+1-tau)
        idx = tp.astype(int).flatten()
        xp = _batch(x,tp.astype(int).flatten())[:,:n]
        res = jax.vmap(m)(xp,tp)
        u = res[0].squeeze()
        y = x[:,idx+tau].T
        plot(u, y, i, j, n)
