from typing import Any, Optional, Sequence, Tuple, Union, Dict, List

import jax
import jax.numpy as jnp
import equinox as eqx

from nn import manifolds
from nn.models import models
from aux import aux
from lib.graph_utils import dense_to_coo

class COSYNN(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    pool: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32
    w_data: jnp.float32
    w_pde: jnp.float32
    x_dim: int
    t_dim: int
    pool_dims: List[int]
    kappa: int
    scalers: Dict[str, jnp.ndarray]
    k_lin: Tuple[int,int]
    k_log: Tuple[int,int]

    def __init__(self, args):
        super(COSYNN, self).__init__()
        self.encoder = getattr(models, args.encoder)(args)
        self.decoder = getattr(models, args.decoder)(args)
        self.pde = aux.neural_burgers(args)
        self.pool = aux.pooling(args)
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.x_dim = args.x_dim
        self.t_dim = args.time_dim
        self.pool_dims = [self.pool.pools[i].layers[-1].linear.bias.shape[0] for i in self.pool.pools]
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

    def project(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x
        
    def logmap0(self, y):
        return self.manifold.logmap0(y, self.c)
   
    def mask(self, x):
        mask0 =  jnp.abs(x[:,0]-10.) > 1e-7
        mask = jnp.concatenate([mask0, jnp.ones(sum(self.pool_dims), dtype=bool)],axis=0)
        #self.mask = mask
        return mask0

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
        zi = z[self.kappa:] * self.scalers['reps'][0]
        ttxz = jnp.concatenate([t,tau,x,z0,zi], axis=0)
        return ttxz

    def align_pde(self, tx, z, tau, u):
        ttxz = self.align(tx, z, tau)
        uttxz = jnp.concatenate([u,ttxz],axis=0)
        return uttxz

    def align_pool(self, z):
        z0 = z[:self.kappa]
        zi = z[self.kappa:]
        zi = self.logmap0(zi)
        return jnp.concatenate([z0,zi], axis=0)
        
    def decode(self, tx, z, tau):
        ttxz = self.align(tx,z,tau)
        u = self.decoder(ttxz)
        return u, (u, ttxz)

    def renorm(self, z, adj, y, inspect=False):
        w = None
        H = 0.
        S = {}
        A = {}
        z_r = z
        y_r = y
        A[0] = jnp.zeros(z.shape[:1]*2).at[adj[0],adj[1]].set(1.)
        for i in self.pool.keys():
            ze = self.align_pool(z)
            S[i] = jax.nn.softmax(self.pool[i](ze, adj, w), axis=0)
            z = jnp.einsum('ij,ik -> jk', S[i], ze)
            y = jnp.einsum('ij,i -> j', S[i], y)
            A[i+1] = jnp.einsum('ji,jk,kl -> il', S[i], A[i], S[i])
            adj, w = dense_to_coo(A[i])
            z_r = jnp.concatenate([z_r, z], axis=0)
            y_r = jnp.concatenate([y_r, y], axis=0)
            H += jax.scipy.special.entr(S[i]).sum()

        if inspect:
            return z_r, y_r, H, S, A
        else:
            return z_r, y_r, H

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
        grad_t = grad[:,0]
        grad_x = grad[:,1:]
        uttxz = self.align_pde(tx, z, tau, u)
        F = 1. * jax.nn.sigmoid(self.pde.F(uttxz))
        v = 0.1 * jax.nn.sigmoid(self.pde.v(uttxz))

        f0 = grad_t
        f1 = -F * jnp.einsum('j,ij -> i', u, grad_x)
        f2 = v * lap_x
        res = f0 - f1 - f2
        return res, res

    def pde_res_grad(self, tx, z, tau, u, grad, lap_x):
        gpde, res = jax.jacfwd(self.pde_res, has_aux=True)(tx, z, tau, u, grad, lap_x)
        return res, gpde

    def div(self, grad_x):
        return jax.vmap(jnp.diag).sum()

    def curl(self, grad_x):
        omega = lambda grad_x: jnp.abs(grad_x - grad_x.T).sum()/2.
        return jax.vmap(omega)(grad_x)

    def enstrophy(self, grad_x):
        return jax.vmap(jnp.sum)(jnp.abs(grad_x))

    def loss_single(self, x0, adj, t, tau, y, test=False):
        tx,z = self.encode(x0, adj, t)
        if test:
            # compute entropy and align separately from renorm
            _, _, H = self.renorm(z, adj, y)
            z = self.align_pool(z)
        else: 
            # renorm z and y only in training loop  
            z, y, H = self.renorm(z, adj, y)
            tx = tx[0] * jnp.ones((z.shape[0],1))
        taus = tau * jnp.ones((z.shape[0],1))
        (u, ttxz), grad, lap_x = jax.vmap(self.val_grad_lap)(tx, z, taus)
        mask = self.mask(x0)
        f = jnp.sqrt(jnp.square(u).sum(1))
        loss_data = jnp.square(f - y).sum()
        resid, gpde = jax.vmap(self.pde_res_grad)(tx, z, taus, u, grad, lap_x) 
        loss_pde = jnp.square(resid).sum() #jnp.einsum('i... -> i', jnp.square(resid))
        loss_gpde = jnp.square(gpde).sum() #jnp.einsum('i... -> i', jnp.square(gpde))
        #loss_data = (mask * loss_data).sum()
        #loss_pde = (mask * loss_pde).sum()
        #loss_gpde = (mask * loss_gpde).sum()
        if test: 
            return loss_data, loss_pde+loss_gpde  
        else:
            loss = self.w_data * loss_data + self.w_pde * (loss_pde + loss_gpde)
            return loss

    def loss_batch(self, xb, adj, tb, tau, yb, test=False):
        sloss = lambda x,t,y: self.loss_single(x, adj, t, tau, y, test=test)
        res = jax.vmap(sloss)(xb, tb, yb)
        if test: 
            return res
        else:
            return jnp.mean(res)

    def loss_bundle(self, xb, adj, tb, taus, yb, test=False):
        lbatch = lambda tau, y: self.loss_batch(xb, adj, tb, tau, y, test=test)
        res = jax.vmap(lbatch)(taus,yb)
        if test:
            return res
        else:
            return jnp.mean(res)

def _forward(model, x0, t, tau, adj):
    tx, z = model.encode(x0, adj, t)
    dec = lambda tx, z: model.decode(tx, z, tau)
    u = jax.vmap(dec)(tx, z)[0]
    return jnp.sqrt(jnp.square(u)).sum(2)

@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_bundle(model, xb, adj, tb, taus, yb):
    return model.loss_bundle(xb, adj, tb, taus, yb)

@eqx.filter_jit
def compute_bundle_terms(model, xb, adj, tb, taus, yb):
    ldata, lpde = model.loss_bundle(xb, adj, tb, taus, yb, test=True)
    return ldata, lpde

# updating model parameters and optimizer state
@eqx.filter_jit
def make_step(grads, model, opt_state, optim):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state
