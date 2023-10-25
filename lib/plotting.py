import jax
import jax.numpy as jnp
import numpy as onp
import pickle
import glob
import pandas as pd
from argparse import Namespace

from lib import utils
from nn.models.cosynn import COSYNN, _forward

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.family'] = 'STIXgeneral'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = 'Computer Modern Roman'

def get_log_data(max_loss = 3.0e+4, n_files = 0): 

    keys =  ['x_dim', 'manifold', 'weight_decay', 'path', 'g', 'epochs', 'w_pde', 'tau_scaler']
    files = glob.glob('../eqx_models/*.pkl'); files.sort()
    dargs = {}
    for file in files[-n_files:]:
        with open(file,'rb') as f:
            data = pickle.load(f)
        args = data['args']
        losses = onp.array([data['loss'][k] for k in data['loss'].keys()])
        
        loss = losses[onp.argsort(losses[:,0])[:9]]
        if sum([k in args for k in keys]) < len(keys) or loss[0,0] > max_loss:
            continue
      
        dargs[file] = {key:args[key] for key in keys}
        dargs[file]['loss[data]'] = loss[:,0].min()
        dargs[file]['loss[pde]'] = loss[:,1].min()
        dargs[file]['error_data'] = loss[:,0].std().item()
        dargs[file]['error_pde'] = loss[:,1].std().item()

    df = pd.DataFrame(dargs).T
    return df

def plot_xy(x='x_dim', y=['loss[data]','loss[pde]'], show_err=False, **kwargs):
    plt.rcParams['figure.figsize'] = [20,8]; 
    df = get_log_data(**kwargs)
    g_dict = {None:r'$g = \tilde{g}(u;x,z,t)$','null':r'$g = 0$'}
    df.g = df.g.apply(lambda key: g_dict[key])
    grouped = df.groupby(['manifold', 'g'])
    fig, ax = plt.subplots(1, len(y))
    mkr = ['^', 'o', 's', 'D']
    err = ['error_data', 'error_pde'] if show_err else [None, None]
    for j,key in enumerate(grouped.groups.keys()):
        for i,yi in enumerate(y):
            grouped.get_group(key).groupby(x).min(y[0]).plot(y=yi, yerr=err[i] , ax=ax[i], legend=' '.join(key), logy=True, marker=mkr[j%len(mkr)], markersize=8, markerfacecolor='none', linestyle=None)
    
    #plt.ylim(df[y_name].min(), 1.1*10**jnp.round(jnp.log10(df[y_name].max())) )
    legend_data = [r'$L_{\mathrm{data}}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    legend_pde = [r'$L_{\mathrm{pde}}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    ax[0].margins(y=0.25)
    ax[1].margins(y=0.25)
    ax[0].legend(legend_data)
    ax[1].legend(legend_pde)
    return ax, df


def mask_plot(y):
    yy = pd.DataFrame(onp.array(y))
    d = (yy.diff()**2. + .4).fillna(1.)
    for i in range(d.shape[0]):
        d.iloc[i] += d.iloc[i-1]
        if d.iloc[i].item()>25.:
            d.iloc[i] = 0.
    mask = (d==0.)
    return yy[mask]

def _plot(model, u, y, tau, i=0, j=-1, n=4, offset=0):
    plt.rcParams['figure.figsize'] = [12,8]; 
    u = onp.array(u)
    y = onp.array(y)
    if j==-1: j = u.shape[0]
    colors = ['mediumslateblue','r','g','b','cornflowerblue','gold','purple']

    enc,dec = model.encoder.__class__.__name__, model.decoder.__class__.__name__
    for k in range(n):
        if k == 0 : 
            ax = pd.DataFrame(u[i:j,k], columns=[rf'$u_{k+1}(t)$ PINN[{enc},{dec}]']).shift(offset).dropna().plot(color=colors[k])
            df = mask_plot(y[i:j,k])
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k < n-1 : 
            pd.DataFrame(u[i:j,k], columns=[rf'$u_{k+1}(t)$ PINN[{enc},{dec}]']).shift(offset).dropna().plot(ax=ax, color=colors[k])
            df = mask_plot(y[i:j,k])
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k == n-1 :
            pd.DataFrame(u[i:j,k], columns=[rf'$u_{k+1}(t)$ PINN[{enc},{dec}]']).shift(offset).dropna().plot(ax=ax, color='C7')
            df = mask_plot(y[i:j,k])
            df.columns = ['data']
            df.plot(ax=ax, color='k', marker='o', markersize=5, markerfacecolor='none', linestyle='none')

#@jax.jit
def _batch(model, x, idx):
    idx = idx.reshape(-1,1)
    win = jnp.arange(1 - model.kappa, 1, 1)
    xb = x.at[:, idx + win].get()
    xb = jnp.swapaxes(xb,0,1)
    return xb

def inference_plot(model, x, adj, tau, i=0, j=-1, n=4):
    _,T = x.shape
    m = lambda x,t: _forward(model,x,t,tau,adj)
    t_min = model.kappa
    t_max = T - tau
    tp = jnp.arange(t_min, t_max, 10)
    idx = tp.astype(int)
    xp = _batch(model, x, idx)
    res = jax.vmap(m)(xp,tp)
    u = res.squeeze()
    y = x[:,idx+tau].T
    #return res,u,y
    _plot(model, u, y, tau, i, j, n)

def plot_u(k=0, tau=60, i=0, j=-1, n=4, **kwargs):
    df = get_log_data(**kwargs)
    idx = df['loss[data]'].sort_values().index[k]
    with open(idx, 'rb') as f: data = pickle.load(f)
    data['args']['log_path'] = idx.split('_')[-1][:-4]
    model, args = utils.read_model(data['args'])
    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    inference_plot(model, x, adj, tau, i=i, j=j, n=n)
    return model, args, data, df, x, adj

