from typing import Any, Optional, Sequence, Tuple, Union, Dict

import jax
import jax.numpy as jnp
import equinox as eqx

from nn import manifolds
from nn.models import models
from pde import pdes

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
        #self.F = getattr(models, args.decoder)(args)
        #self.g = getattr(models, args.decoder if not args.g else args.g)(args)
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
                        'reps' : jnp.array([args.rep_scaler]),
                        'input': jnp.array([args.input_scaler]),
                        'tau': jnp.array([args.tau_scaler])
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
   
    def mask(self, x0):
        return jnp.abs(x0[:,0]-10.) > 1e-7

    def encode(self, x0, adj, t, eps=0.):
        x0 *= self.scalers['input'][0]
        z_x = self.encoder(x0, adj)
        z = z_x[:,:-self.x_dim]
        x = eps * z_x[:,-self.x_dim:]
        t = t*jnp.ones((x.shape[0],1))
        tx = jnp.concatenate([t, x], axis=1)
        return tx, z

    def align(self, tx, z, tau):
        t,x = tx[:1], tx[1:]
        t = self.time_encode(t)
        tau = self.time_encode(tau * self.scalers['tau'])
        z0 = z[:self.kappa]
        zi = z[self.kappa:]
        zp = self.logmap0(zi) * self.scalers['reps'][0]
        ttxz = jnp.concatenate([t,tau,x,z0,zp], axis=0)
        return ttxz

    def align_pde(self, tx, z, tau, u):
        ttxz = self.align(tx, z, tau)
        uttxz = jnp.concatenate([u,ttxz],axis=0)
        return uttxz
    
    def decode(self, tx, z, tau):
        ttxz = self.align(tx,z,tau)
        u = self.decoder(ttxz)
        return u, (u, ttxz)

    def val_grad(self, tx, z, tau):
        f = lambda tx: self.decode(tx,z,tau)
        grad, val = jax.jacfwd(f, has_aux=True)(tx)
        grad =  grad[0]
        return grad, (grad, val)

    def val_grad_lap(self, tx, z, tau):
        vg = lambda tx,z: self.val_grad(tx,z,tau)
        grad2, (grad,(u,ttxz)) = jax.jacfwd(vg, has_aux=True)(tx,z)
        hess = jax.vmap(jnp.diag)(grad2)
        lap_x = hess[:,1:].sum(1)
        return (u.flatten(), ttxz), grad, lap_x

    def pde_res(self, tx, z, tau, u, grad, lap_x):
        grad_t = grad[:,:,0]
        grad_x = grad[:,:,1:]
        tau = tau * jnp.ones_like(u[:,:1])  
        xop = jax.vmap(self.align_pde)(tx, z, tau, u) # argument for neural operators
        F = jax.vmap(self.pde.F)(xop).reshape(-1,1)
        v = jax.vmap(self.pde.v)(xop).reshape(-1,1)
        
        f0 = grad_t
        f1 = -F * jnp.einsum('mp,mqp->mq', u, grad_x)
        f2 = v * lap_x
        resid = f0 - f1 - f2
        return resid

    def div(self, grad_x):
        return jax.vmap(jnp.diag).sum()

    def vort(self, grad_x):
        omega = lambda grad_x: jnp.abs(grad_x - grad_x.T).sum()/2.
        return jax.vmap(omega)(grad_x)

    def enstrophy(self, grad_x):
        return jax.vmap(jnp.sum)(jnp.abs(grad_x))

    def loss_single(self, x0, adj, t, tau, y):
        vgl = lambda tx,z: self.val_grad_lap(tx, z, tau)
        tx,z = self.encode(x0, adj, t)
        (u, ttxz), grad, lap_x = jax.vmap(vgl)(tx,z)
        mask = self.mask(x0)
        f = jnp.sqrt(jnp.square(u)).sum(1) - .01
        loss_data = jax.lax.square(f - y)
        resid = self.pde_res(tx, z, tau, u, grad, lap_x) 
        loss_pde = jax.lax.square(resid).sum(1)
        loss_data *= mask
        loss_pde *= mask
        loss = self.w_data * loss_data + self.w_pde * loss_pde
        return loss.sum()

    def loss_batch(self, xb, adj, tb, tau, yb):
        sloss = lambda x,t,y: self.loss_single(x, adj, t, tau, y)
        return jnp.mean(jax.vmap(sloss)(xb, tb, yb))

    def loss_bundle(self, xb, adj, tb, taus, yb):
        lbatch = lambda tau, y: self.loss_batch(xb, adj, tb, tau, y)
        return jnp.mean(jax.vmap(lbatch)(taus,yb))


def _forward(model, x0, t, tau, adj):
    t_x, z = _encode(x0, adj, t, model)
    dec = lambda t_x, z: _decode(t_x, tau, z, model)
    return jax.vmap(dec)(t_x, z)

def compute_loss(model, x0, adj, t, tau, y):
    (u, (zi, txz)), grad_tx = compute_val_grad(x0, adj, t, tau, model)
    grad_t, grad_x = grad_tx[:,:1], grad_tx[:,1:]
    mask = ( jnp.abs(x0[:,0] - 10.) > 1e-18)
    loss_data = jax.lax.square(u - y)
    resid = model.pde.res(u, txz, grad_t, grad_x)
    loss_pde = jax.lax.square(resid).sum(0).flatten()
    loss_data *= mask 
    loss_pde *= mask
    loss_data = loss_data.sum()
    loss_pde = loss_pde.sum()
    return model.w_data * loss_data +  model.w_pde * loss_pde
 
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
    loss_pde = jax.lax.square(resid).sum(0).flatten()
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
def make_step(grads, model, opt_state, optim):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state

