import sys
import warnings
warnings.filterwarnings('ignore')

from hgcn import optimizers
from hgcn.models import models
from config import parser

import pandas as pd
import numpy as onp

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import equinox as eqx
import optax

prng = lambda: jax.random.PRNGKey(onp.random.randint(1e+3))

t2df = lambda tensor: pd.DataFrame(tensor.detach().numpy())

def get_series():
    print('Calculating entire time series...')
    res = jnp.zeros_like(xx)
    for t in jnp.arange(1,T).reshape(-1,1):
        x = xx[:,int(t):int(t)+tau]
        t_ = t*jnp.ones((x.shape[0],1))
        G = jax.vmap(model)(x,t_,t_)
        res = res.at[:,t].set(G)
    g = res[:,1:-60]
    return onp.asarray(g)

def plot(g,P,i=0,j=0):
    if j==0: j = g.shape[1]
    ax = pd.DataFrame(g.T).iloc[i:j,:4].plot()
    idx = range(i,j)
    y = [P(i) for i in idx]
    pd.DataFrame(y,index=idx).plot(ax=ax)

class COSYNN(eqx.Module):
    enc: eqx.Module
    dec: eqx.Module

    def __init__(self, enc, dec):
        super(COSYNN,self).__init__()
        self.enc = enc
        self.dec = dec

    def __call__(self,x,A,t):
        z = enc(x,A)
        G = dec(z,A,t)
        return G

@eqx.filter_value_and_grad
def compute_loss(model, y, x, A, t):
    t_ = t*jnp.ones((x.shape[0],1))
    pred = jax.vmap(model)(x,t_,t_)
    return jnp.sum((y-pred)**2)

@eqx.filter_jit
def make_step(model, y, x, A, t, opt_state):
    loss, grads = compute_loss(model,y,x,A,t)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model,updates)
    return loss, model, opt_state


if __name__ == '__main__':
    args = parser.parse_args()
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim
    enc = getattr(models, args.encoder)(args)
    dec = getattr(models, args.decoder)(args)
    
    T = 2000
    n = 100
    tau = 60
    xx = jnp.zeros([n, tau + T]) 
    adj = jax.random.randint(prng(),(2,n),0,n)
    A = sparse.BCOO((jnp.ones(adj.shape[1]), (adj[0],adj[1])), shape=(n,n))

    c = jax.nn.sigmoid(jnp.zeros(3))
    model = COSYNN(enc,dec) 
    optim = optax.adam(args.lr) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
 
    w = jnp.zeros(2)
    lam = jnp.array(1e-4)
    for i in range(args.epochs):
        w = jnp.ones(2) * (1. - jnp.exp(-lam*i))
        ti = jax.random.randint(prng(),(30,1),1,T).astype(jnp.float32)
        loss = jnp.array(0.)
        total = 0.
        for t in ti:
            x = xx[:,int(t):int(t)+tau]
            h = lambda t: (.01*t**2.+t)
            y = h(t) * jnp.ones(x.shape[0])
            loss, model, opt_state = make_step(model, y, x, A, t, opt_state)
            total += loss/len(ti)
            #pinn
        if i % args.log_freq == 0:
            #print(f'w = {w}')
            print(f'{i}/{args.epochs}: loss = {total.item():.4e}')
            #print(f'{i}/{args.epochs}: loss_sq = {loss_sq:.4e}, loss_pde = {loss_pde:.4e}, loss_gpde = {loss_gpde:.4e}, c = {c.detach().numpy().round(4)[:3]}')

    #g = get_series()
    #plot(g,h)

