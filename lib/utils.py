import os
import equinox as eqx
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp

def _batch(model,x,idx):
    xi = lambda i: x.at[:,i:i+model.kappa].get()
    xb = jnp.array([xi(i) for i in idx])
    return xb


def mask_plot(y):
    yy = pd.DataFrame(onp.array(y))
    d = (yy.diff()**2. + .4).fillna(1.)
    for i in range(d.shape[0]):
        d.iloc[i] += d.iloc[i-1]
        if d.iloc[i].item()>25.:
            d.iloc[i] = 0.
    mask = (d==0.)
    return yy[mask]

def plot(u,y,tau,i=0,j=-1,n=4):
    plt.rcParams['figure.figsize'] = [8,8]; plt.rcParams['font.size'] = 14; plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True
    u = onp.array(u)
    y = onp.array(y)
    if j==-1: j = u.shape[0]
    colors = ['mediumslateblue','r','g','b','cornflowerblue','gold','purple']

    for k in range(n):
        if k == 0 : 
            ax = pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(-tau.item()).dropna().plot(color=colors[k])
            df = mask_plot(y[i:j,k])
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k < n-1 : 
            pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(-tau.item()).dropna().plot(ax=ax, color=colors[k])
            df = mask_plot(y[i:j,k])
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k == n-1 :
            pd.DataFrame(u[i:j,k],columns=[rf'$G_{k+1}(t)$ PINN[{args.encoder},{args.decoder}]']).shift(-tau.item()).dropna().plot(ax=ax, color='C7')
            df = mask_plot(y[i:j,k])
            df.columns = ['data']
            df.plot(ax=ax, color='k', marker='o', markersize=5, markerfacecolor='none', linestyle='none')

def call(model, x0, t, tau, adj):
    z_x = model.encoder(x0, adj)
    z = z_x[:,:-model.x_dim]
    x = 0. * z_x[:,-model.x_dim:]
    t = t*jnp.ones((x.shape[0],1))
    tau = tau*jnp.ones((x.shape[0],1))
    t = jax.vmap(model.time_encode)(t)
    tau = jax.vmap(model.time_encode)(tau)
    txz = jnp.concatenate([t, tau, x, z], axis=1)
    return jax.vmap(model.decoder)(txz), txz

def inference(model, tau, i=0, j=-1, n=3):
    m = lambda x,t: call(model,x,t,tau,adj)
    tp = jnp.linspace(0,T-tau[0],T+1-tau[0])
    idx = tp.astype(int).flatten()
    xp = _batch(model, x, tp.astype(int).flatten())[:,:n]
    res = jax.vmap(m)(xp,tp)
    u = res[0].squeeze()
    y = x[:,idx+tau].T
    plot(u, y, tau, i, j, n)

def save_model(model, log, path='../eqx_models'):
    if not os.path.exists(path): os.mkdir(path)
    time_str = str(int(time.time()))
    with open(f'log_{time_str}.pkl','wb') as f: 
        pickle.dump(vars(log),f)
    eqx.tree_serialise_leaves(path + '/cosynn.eqx', model)
