from typing import Any, Optional, Sequence, Tuple, Union, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from nn import manifolds
from nn.models import models
from aux import aux
from lib.graph_utils import dense_to_coo

prng = lambda i: jax.random.PRNGKey(i)

class COSYNN(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    pde: eqx.Module
    pool: eqx.Module
    manifold: manifolds.base.Manifold
    c: jnp.float32
    w_data: jnp.float32
    w_pde: jnp.float32
    w_gpde: jnp.float32
    w_ent: jnp.float32
    F_max: jnp.float32
    v_max: jnp.float32
    x_dim: int
    t_dim: int
    pool_dims: List[int]
    kappa: int
    scalers: Dict[str, jnp.ndarray]
    k_lin: Tuple[int,int]
    k_log: Tuple[int,int]
    beta: np.float32

    def __init__(self, args):
        super(COSYNN, self).__init__()
        self.kappa = 0 if args.res else args.kappa
        self.encoder = getattr(models, args.encoder)(args)
        self.decoder = getattr(models, args.decoder)(args)
        self.pde = aux.neural_burgers(args)
        self.pool = aux.pooling(args)
        self.manifold = getattr(manifolds, args.manifold)()
        self.c = args.c
        self.w_data = args.w_data
        self.w_pde = args.w_pde
        self.w_gpde = args.w_gpde
        self.w_ent = args.w_ent
        self.F_max = args.F_max
        self.v_max = args.v_max
        self.x_dim = args.x_dim
        self.t_dim = args.time_dim
        self.pool_dims = [self.pool.pools[i].layers[-1].linear.bias.shape[0] for i in self.pool.pools]
        self.scalers = {'t_lin': 10. ** jnp.arange(2, 2*self.t_dim, 1, dtype=jnp.float32),
                        't_log': 10. ** jnp.arange(-2, 2*self.t_dim, 1, dtype=jnp.float32),
                        't_cos': 10. **jnp.arange(-4,2*self.t_dim, 1, dtype=jnp.float32),
                        'reps' : jnp.array([args.rep_scaler]),
                        'input': jnp.array([args.input_scaler]),
                        'tau': jnp.array([args.tau_scaler])
                       }
        self.k_lin = self.t_dim//2
        self.k_log = self.t_dim - self.k_lin
        self.beta = args.beta 

    def time_encode(self, t):
        if self.t_dim==1: 
            return jnp.log( t * 100. + 1. )
        t_lin = t / jnp.clip(self.scalers['t_lin'], 1e-7)
        t_log = t / jnp.clip(self.scalers['t_log'], 1e-7)
        t_cos = t / jnp.clip(self.scalers['t_cos'], 1e-7)
        t_lin = t_lin[:self.k_lin]
        t_log = jnp.log( t_log[:self.k_log] + 1. )
        t_cos = t_cos[:self.k_lin]
        t = jnp.concatenate([t_lin, t_log])
        return t

    def exp(self, x):
        x = self.manifold.proj_tan0(x, c=self.c)
        x = self.manifold.expmap0(x, c=self.c)
        x = self.manifold.proj(x, c=self.c)
        return x

    def log(self, y):
        y = self.manifold.logmap0(y, self.c)
        y = y * jnp.sqrt(self.c) * 1.4763057
        return y
  
    def encode(self, x0, adj, t, eps=0.):
        z_x = self.encoder(x0, adj)
        z = z_x[:,:-self.x_dim]
        x = eps * z_x[:,-self.x_dim:]
        t = t*jnp.ones((x.shape[0],1))
        tx = jnp.concatenate([t, x], axis=1)
        return tx, z

    def align(self, tx, z):
        t,x = tx[:1], tx[1:]
        t = self.time_encode(t)
        z0 = z[:self.kappa]
        zi = z[self.kappa:] * self.scalers['reps'][0]
        txz = jnp.concatenate([t,x,z0,zi], axis=0)
        return txz

    def align_pde(self, tx, z, u):
        txz = self.align(tx, z)
        utxz = jnp.concatenate([u*0.,txz],axis=0)
        return txz

    def align_pool(self, z):
        z0 = z[:,:self.kappa]
        zi = z[:,self.kappa:]
        zi = self.log(zi)
        return jnp.concatenate([z0,zi], axis=-1)

    def embed_pool(self, z, adj, w, i):
        z0 = z[:,:self.kappa]
        zi = z[:,self.kappa:]
        ze = self.pool.embed[i](zi, adj, w)
        zi = self.manifold.mobius_add(zi,ze, self.c)
        s = self.pool[i](z, adj, w)
        z = jnp.concatenate([z0,zi], axis=-1)
        return z,s

    def delta(self, z):
        f = z
        i = self.kappa
        return z
    
    def decode(self, tx, z, key=prng(0)):
        z = self.delta(z) 
        txz = self.align(tx,z)
        u = self.decoder(txz, key=key)
        return u, (u, txz)

    def renorm(self, x, adj, y, inspect=False):
        w = None
        loss_ent = 0.
        S = {}
        A = {}
        z_r = x
        y_r = y
        A[0] = jnp.zeros(x.shape[:1]*2).at[adj[0],adj[1]].set(1.)
        for i in self.pool.keys():
            z,s = self.embed_pool(x, adj, w, i)
            S[i] = jax.nn.softmax(self.log(s) * 200., axis=0)
            x = jnp.einsum('ij,ik -> jk', S[i], z)
            y = jnp.einsum('ij,ki -> kj', S[i], y)
            A[i+1] = jnp.einsum('ji,jk,kl -> il', S[i], A[i], S[i])
            adj, w = dense_to_coo(A[i])
            z_r = jnp.concatenate([z_r, x], axis=0)
            y_r = jnp.concatenate([y_r, y], axis=-1)
            loss_ent += jax.scipy.special.entr(S[i]).sum()

        if inspect:
            return z_r, y_r, loss_ent, S, A
        else:
            return z_r, y_r, loss_ent

    def val_grad(self, tx, z):
        f = lambda tx: self.decode(tx,z)
        grad, val = jax.jacfwd(f, has_aux=True)(tx)
        return grad, (grad, val)

    def val_grad_lap(self, tx, z):
        vg = lambda tx,z: self.val_grad(tx,z)
        grad2, (grad,(u,ttxz)) = jax.jacfwd(vg, has_aux=True)(tx,z)
        hess = jax.vmap(jnp.diag)(grad2)
        lap_x = hess[:,1:].sum(1)
        return (u.flatten(), ttxz), grad, lap_x

    def pde_res(self, tx, z, u, grad, lap_x):
        grad_t = grad[:,0]
        grad_x = grad[:,1:]
        utxz = self.align_pde(tx, z, u)
        F = self.F_max * jax.nn.sigmoid(self.pde.F(utxz))
        v = self.v_max * 1. * jax.nn.sigmoid(self.pde.v(utxz))

        f0 = grad_t
        f1 = -F * jnp.einsum('j,ij -> i', u, grad_x)
        f2 = v * lap_x
        res = f0 - f1 - f2
        return res, res

    def pde_res_grad(self, tx, z, u, grad, lap_x):
        gpde, res = jax.jacfwd(self.pde_res, has_aux=True)(tx, z, u, grad, lap_x)
        return res, gpde

    def div(self, grad_x):
        return jax.vmap(jnp.trace)

    def curl(self, grad_x):
        omega = lambda grad_x: jnp.abs(grad_x - grad_x.T).sum()/2.
        return jax.vmap(omega)(grad_x)

    def enstrophy(self, grad_x):
        return jax.vmap(jnp.sum)(jnp.abs(grad_x))

    def loss_single_auto(self, x0, adj, t, y, key, mode=0):
        tx,z = self.encode(x0, adj, t)
        z = self.align_pool(z)
        if mode==1:
            # compute entropy and align separately from renorm
            _, _, loss_ent = self.renorm(z, adj, y)
        else: 
            # renorm z and y only in training loop  
            z, y, loss_ent = self.renorm(z, adj, y)
            tx = tx[0] * jnp.ones((z.shape[0],1))
        loss_data = loss_pde = loss_gpde = 0.
        for i in range(y.shape[0]):
            (u, txz), grad, lap_x = jax.vmap(self.val_grad_lap)(tx, z)
            f =  jnp.sqrt(jnp.square(u).sum(1))
            loss_data += jnp.square(f - y[i]).sum()
            resid, gpde = jax.vmap(self.pde_res_grad)(tx, z, u, grad, lap_x)
            loss_pde += jnp.square(resid).sum()
            loss_gpde += jnp.square(gpde).sum()      
            tx = tx.at[:,0].set(tx[:,0]+1.)
            x = jnp.concatenate([z[:,1:self.kappa],f.reshape(-1,1)], axis=-1)
            z = z.at[:,:self.kappa].set(x)

        if mode==-1: #reporting
            return jnp.array([loss_data, loss_pde, loss_gpde, loss_ent])
        elif mode==1: #slaw
            return jnp.array([loss_data, loss_pde, loss_gpde, loss_ent])
        elif mode==0: #default
            loss = self.w_data * loss_data + self.w_pde * loss_pde + self.w_gpde * loss_gpde + self.w_ent * loss_ent
            return loss
       
    def loss_vmap(self, xb, adj, tb, yb, key=prng(0), mode=0, state=None):
        n = xb.shape[0]
        kb = jax.random.split(key, n) 
        loss_vec = lambda x,t,y,k: self.loss_single_auto(x, adj, t, y, k, mode=mode)
        loss = jax.vmap(loss_vec)(xb, tb, yb, kb)
        if mode==1:
            assert state != None
            a,b = state['a'], state['b']
            loss = loss/n
            a = self.beta * a + (1. - self.beta) * loss**2
            b = self.beta * b + (1. - self.beta) * loss
            s = jnp.sqrt(a - b**2)
            w = loss.shape[0] / s / (1./s).sum()
            loss = w * loss
            loss = self.w_data * loss[0] + self.w_pde * loss[1] + self.w_gpde * loss[2] + self.w_ent * loss[3] 
            state['a'], state['b'] = a,b
            return loss.sum(), state
        else:
            return loss/n, state

    def loss_scan(self, xb, adj, tb, yb, key=prng(0), mode=0, state=None):
        n = xb.shape[0] # batch size
        kb = jax.random.split(key,n) 
        body_fun = lambda i,val: val + self.loss_single_auto(xb[i], adj, tb[i], yb[i], kb[i], mode=mode)
        loss = 0. if mode==0 else jnp.zeros(4)
        loss = jax.lax.fori_loop(0, n, body_fun, loss)
        if mode==1:
            assert state != None
            a,b = state['a'], state['b']
            loss = loss/n
            a = self.beta * a + (1. - self.beta) * loss**2
            b = self.beta * b + (1. - self.beta) * loss
            s = jnp.sqrt(a - b**2)
            w = loss.shape[0] / s / (1./s).sum()
            loss = w * loss
            loss = self.w_data * loss[0] + self.w_pde * loss[1] + self.w_gpde * loss[2] + self.w_ent * loss[3] 
            state['a'], state['b'] = a,b
            return loss.sum(), state
        else:
            return loss/n, state

def _forward(model, x0, t, tau, adj):
    tx,z = model.encode(x0, adj, t)
    z = model.align_pool(z)
    z, y, loss_ent = model.renorm(z, adj, z[:,0])
    tx = tx[0] * jnp.ones((z.shape[0],1))
    taus = tau * jnp.ones((z.shape[0],1))
    (u, ttxz), grad, lap_x = jax.vmap(self.val_grad_lap)(tx, z, taus)
    f = jnp.sqrt(jnp.square(u).sum(1)) 
    return f, grad_x 

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_scan(model, xb, adj, tb, yb, key=prng(0), mode=0, state=None):
    return model.loss_scan(xb, adj, tb, yb, key=key, mode=mode, state=state)

@eqx.filter_jit
def loss_terms(model, xb, adj, tb, yb):
    return model.loss_scan(xb, adj, tb, yb, mode=-1)

# updating model parameters and optimizer state
@eqx.filter_jit
def make_step(grads, model, opt_state, optim):
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model,updates)
    return model, opt_state
