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


# functions for computing losses and gradients
def _encode(x0, adj, t, model):
    x0 *= model.scalers['input'][0]
    z_x = model.encoder(x0, adj)
    z = z_x[:,:-model.x_dim]
    x = 0. * z_x[:,-model.x_dim:]
    t = t*jnp.ones((x.shape[0],1))
    t_x = jnp.concatenate([t, x], axis=1)
    return t_x, z

def _decode(t_x, tau, z, model):
    t,x = t_x[:1], t_x[1:]
    t = model.time_encode(t)
    tau = model.time_encode(tau * model.scalers['tau'])
    z0 = z[:model.kappa]
    zi = z[model.kappa:]
    zp = model.logmap0(zi) * model.scalers['reps'][0]
    txz = jnp.concatenate([t,tau,x,z0,zp], axis=0)
    pred = model.decoder(txz).reshape(())
    return pred, (zi, txz)

def _forward(model, x0, t, tau, adj):
    t_x, z = _encode(x0, adj, t, model)
    dec = lambda t_x, z: _decode(t_x, tau, z, model)
    return jax.vmap(dec)(t_x, z)

def _forward_eps(model, x0, t, tau, adj, prng, eps):
    x0 = x0 + jnp.abs(x0)**.7 * eps * (-1 + 2. * jax.random.beta(prng,4.,4.,x0.shape))
    return _forward(model, x0, t, tau, adj)
    
def _batch_eps(model, x0, t, tau, adj, eps=.05, n=50):
    feps = lambda prng: _forward_eps(model, x0, t, tau, adj, prng, eps)
    prng = jax.random.split(jax.random.PRNGKey(0), n)
    res = jax.vmap(feps)(prng)[0]
    return res.mean(0).T, res.std(0).T

def _batch_drop(model, x0, t, tau, adj, n=50):
    fwd = lambda _: _forward(model, x0, t, tau, adj)
    _ = jnp.empty((n,1))
    res = jax.vmap(fwd)(_)[0]
    return res.mean(0).T, res.std(0).T

def forward(model, x0, t, tau, adj, eps=.1, n=50, err=True):
    if err:
        u = _forward(model, x0, t, tau, adj)[0]
        err = _batch_drop(model, x0, t, tau, adj, n)[1]
        return u, err
    else:
        u = _forward(model, x0, t, tau, adj)[0]
        return u, None

@eqx.filter_value_and_grad(has_aux = True)
def decoder_val_grad(t_x, tau, z, model):
    return _decode(t_x, tau, z, model)

@eqx.filter_value_and_grad(has_aux = True)
def decoder_val_grad_lap(t_x, tau, z, model):
    aux, grad_tx = decoder_val_grad(t_x, tau, z, model)
    return grad_tx, aux

def compute_val_grad_lap(x0, adj, t, tau, model):
    t_x, z = _encode(x0, adj, t, model)
    dvgl = lambda t_x, z: decoder_val_grad_lap(t_x, tau, z, model)
    return jax.vmap(dvgl)(t_x, z) 

def compute_val_grad(x0, adj, t, tau, model):
    t_x, z = _encode(x0, adj, t, model) 
    dvg = lambda t_x, z: decoder_val_grad(t_x, tau, z, model)
    return jax.vmap(dvg)(t_x, z)

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

