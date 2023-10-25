import sys
import warnings
warnings.filterwarnings('ignore')

#from hgcn import optimizers
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


class COSYNN(eqx.Module):
    #enc: eqx.Module
    #dec: eqx.Module
    M: eqx.nn.MLP
    scalers: jax.numpy.ndarray

    def __init__(self, args):
        super(COSYNN,self).__init__()
        #self.enc = enc
        #self.dec = dec
        start = 1
        self.scalers = jnp.array([10**i for i in range(start,start+args.time_dim)])
        hidden = 256
        self.M = [eqx.nn.Linear(args.time_dim * 2, hidden, key=prng()), jax.nn.silu,
                        eqx.nn.Linear(hidden, hidden, key=prng()), jax.nn.silu,
                        eqx.nn.Linear(hidden, hidden, key=prng()), jax.nn.silu,
                        eqx.nn.Linear(hidden, 1, key=prng()), jax.nn.silu]

    def __call__(self,x,t,A=None):
        h = t.reshape(1)/self.scalers
        h = jnp.concatenate([h,jnp.log(h+1.)])
        if not A:
            for m in self.M:
                h = m(h) 
        return h

@eqx.filter_value_and_grad
def compute_val(t, x, model, A=None):
    pred = model(x,t)
    return pred[0]

def compute_batch(t, x, model, A=None):
    comp = lambda x: compute_val(t,x,model)
    return jax.vmap(comp)(x)

#@eqx.filter_value_and_grad
def compute_loss(model, t, x, y, w):
    val, grad_t = compute_batch(t,x,model)
    loss_data = jnp.square(val - y).sum()
    loss_pde = jnp.square(grad_t - (.02*t + 1.)).sum()
    return w[0] * loss_data + w[1] * loss_pde

def compute_loss_terms(model, t, x, y, w):
    val, grad_t = compute_batch(t,x,model)
    loss_data = jnp.square(val - y).sum()
    loss_pde = jnp.square(grad_t - (.02*t + 1.)).sum()
    return loss_data, loss_pde

clt = lambda ti,yi: [loss.mean() for loss in jax.vmap(lambda t,y: compute_loss_terms(model,t,x,y,w))(ti,yi)]

@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_batch(model,t,x,y,w):
    lbatch = lambda t,y: compute_loss(model,t,x,y,w)
    return jnp.mean(jax.vmap(lbatch)(t,y))


@eqx.filter_jit
def make_step(grads, model, opt_state):
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model,updates)
    return model, opt_state


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
    model = COSYNN(args)
    schedule = optax.warmup_exponential_decay_schedule(args.lr, peak_value=args.lr, warmup_steps=10000,
                                                        transition_steps=args.epochs, decay_rate=1e-2, end_value=args.lr/1e+3)
    optim = optax.chain(optax.clip(args.max_norm),optax.adam(schedule)) 
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
 
    w = jnp.array([1.,2.]) 
    t = 5.*jnp.ones(1)
    h = lambda t: (.01*t**2.+t)
    x = xx[:,int(t):int(t)+tau]
    
    for i in range(args.epochs):
        ti = jax.random.randint(prng(),(50,1),1,T).astype(jnp.float32)
        loss = jnp.array(0.)
        yi = h(ti) * jnp.ones(x.shape[0])
        loss, grad = loss_batch(model, ti, x, yi, w) 
        model, opt_state = make_step(grad, model, opt_state)
        if i % args.log_freq == 0:
            ti = jnp.linspace(0.,T,1000).reshape(-1,1)
            loss = jnp.array(0.)
            yi = h(ti) * jnp.ones(x.shape[0])
            loss, grad = loss_batch(model, ti, x, yi, w) 
            model, opt_state = make_step(grad, model, opt_state)
            
            loss_data, loss_pde = clt(ti,yi)
            print(f'{i}/{args.epochs}: loss_data = {loss_data:.4e}, loss_pde = {loss_pde:.4e}, lr = {schedule(i).item():.4e}')
    

    def plot(res,h,i=0,j=0):
        if j==0: j = res.shape[0]
        ax = pd.DataFrame(res).iloc[i:j,:3].plot()
        idx = range(i,j)
        y = [h(i) for i in idx]
        pd.DataFrame(y,index=idx).plot(ax=ax)
    
    m = lambda t: model(x,t)
    res = onp.array(jax.vmap(m)(jnp.linspace(0,T,T+1)))
    plot(res,h)

