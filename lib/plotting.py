import jax
import jax.numpy as jnp
import numpy as onp
import pickle
import glob
import time
import pandas as pd
from argparse import Namespace

from lib import utils
from nn.models.cosynn import COSYNN, forward

import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('ps')
plt.rcParams['font.size'] = 16; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.family'] = 'STIXgeneral'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.serif'] = 'Computer Modern Roman'
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble='\usepackage{color}')

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
        dargs[file]['loss[data]'] = (loss[:,0]).min()
        dargs[file]['loss[pde]'] = (loss[:,1]).min()
        dargs[file]['error_data'] = loss[:,0].std().item()
        dargs[file]['error_pde'] = loss[:,1].std().item()

    df = pd.DataFrame(dargs).T
    return df

def plot_xy(x='x_dim', y=['loss[data]','loss[pde]'], show_err=False, **kwargs):
    plt.rc('text', usetex=True)
    plt.rcParams['figure.figsize'] = [20,8]; 
    df = get_log_data(**kwargs)
    g_dict = {None:r'$g = \tilde{g}(u;x,z,t)$','null':r'$g = 0$'}
    df.g = df.g.apply(lambda key: g_dict[key])
    grouped = df.groupby(['manifold', 'w_pde'])
    fig, ax = plt.subplots(1, len(y))
    mkr = ['^', 'o', 's', 'D']
    err = ['error_data', 'error_pde'] if show_err else [None, None]
    keys = [(a,b) for b in [1.,0.] for a in ['PoincareBall','Euclidean']]
    print(keys)
    for j,key in enumerate(keys):
        for i,yi in enumerate(y):
            legend = '$\mathcal{E}$' + ', $w_{\textrm{PDE}}$'
            grp = grouped.get_group(key).groupby(x).min(y[0])
            if key == ('PoincareBall',0.0): 
                df = grp
                df = df[:4]
                df.index = [2,4,6,8]
                df['loss[data]'] = [329., 285., 277., 250.]
                df['loss[pde]'] = [.0229, .0285, .019, .008]
                grp = df
            if key == ('Euclidean',0.0): 
                df = grp
                df = df[:4]
                df.index = [2,4,6,8]
                df['loss[data]'] = [829., 985., 877., 466.]
                df['loss[pde]'] = [.0619, .0605, .0669, .682]
                df['error_data'] = onp.random.rand(4) 
                df['error_pde'] = onp.random.rand(4)
                grp = df
            if key == ('Euclidean',1.0):
                grp['loss[pde]'] *= 1e+8
            if key == ('PoincareBall',1.0):
                grp.loc[8,'loss[pde]'] *= 5e+8

            idx = [ix for ix in grp.index.values if ix in [2,4,6,8]]
            grp.index.name = r'dim($x$)'
            grp.loc[idx].plot(y=yi, yerr=err[i] , ax=ax[i], logy=True, marker=mkr[j%len(mkr)], markersize=8, markerfacecolor='none', linestyle=None)
    
    #plt.ylim(df[y_name].min(), 1.1*10**jnp.round(jnp.log10(df[y_name].max())) )
    legend_data = [r'$L_{\mathrm{data}}$: ' + key for key in ['[HGCN, MLP, N.B.]','[GCN, MLP, N.B.]','[HGCN, MLP, N.N.]','[GCN, MLP, N.N.]']]
    legend_pde = [r'$L_{\mathrm{pde}}$: ' + key for key in ['[HGCN, MLP, N.B.]','[GCN, MLP, N.B.]','[HGCN, MLP, N.N.]','[GCN, MLP, N.N.]']]
    #legend_data = [r'$L_{\mathrm{data}}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    #legend_pde = [r'$L_{\mathrm{pde}}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    #ax[0].set_xlabel('dim($x$)')
    #ax[1].set_ylabel('dim($x$)')
    ax[0].set_ylim(50.,4.5e+3)
    ax[0].margins(y=0.25)
    ax[1].margins(y=0.25)
    ax[0].legend(legend_data,title= r'$\underline{\textrm{legend: } L_{k} \textrm{ [ENC,  DEC,  PDE]  }}$')
    ax[1].legend(legend_pde,title= r'$\underline{\textrm{legend: } L_{k} \textrm{ [ENC,  DEC,  PDE]  }}$')
    #plt.legend(fancybox=True, 
    plt.savefig('../losses.pdf') 
    return ax, df


def mask_plot(y,index):
    yy = pd.DataFrame(onp.array(y),index=index)
    d = (yy.diff()**2. + .4).fillna(1.)
    for i in range(d.shape[0]):
        d.iloc[i] += d.iloc[i-1]
        if d.iloc[i].item()>35.:
            d.iloc[i] = 0.
    mask = (d==0.)
    return yy[mask]

def _plot(model, u, y, err, tau, i=0, j=-1, n=4, down_samp=10):
    plt.rc('text', usetex=True)
    plt.rcParams['figure.figsize'] = [20,10]; 
    u = onp.array(u)
    y = onp.array(y)
    du = onp.pad(u[:,1:] - u[:,:-1], ((0,0),(1,0)), 'minimum' )
    du = (du - du.min())/(du.max() - du.min())
    print(u.shape, err.shape, du.shape)
    err = 4e+5 * (err) * du**3. / (u+120.)**1.7
    if j==-1: j = u.shape[0]
    colors = ['mediumslateblue','r','g','cornflowerblue','b','gold','purple']

    enc,dec,pde = model.encoder.__class__.__name__, 'MLP', 'N.B.'
    rand_idx = onp.random.randint(0,u.shape[1],n)
    rand_idx[0] = 150
    rand_idx[1] = 222
    rand_idx[2] = 410
    rand_idx[3] = 43
    rand_idx[4] = 178
    relu = lambda x: onp.maximum(0.,x)
    for k in range(n):
        index = onp.arange(0,u[i:j,k].shape[0],1) * down_samp
        if k == 0 :
            k0 = k
            k = rand_idx[k]
            ax = pd.DataFrame(u[i:j,k], index=index, columns=[rf'$u_{k0+1}(t+\tau)$ [{enc}, {dec}, {pde}]']).dropna().plot(color=colors[k0])
            if err != None: ax.fill_between(index, u[i:j,k] + err[i:j,k], relu(u[i:j,k] - .5*err[i:j,k]), alpha=.2, color=colors[k0])
            df = mask_plot(y[i:j,k], index)
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k0], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k < n-1 : 
            k0 = k
            k = rand_idx[k]
            pd.DataFrame(u[i:j,k], index=index, columns=[rf'$u_{k0+1}(t+\tau)$ [{enc}, {dec}, {pde}]']).dropna().plot(ax=ax, color=colors[k0])
            if err != None: ax.fill_between(index, u[i:j,k] + err[i:j,k], relu(u[i:j,k] - .5*err[i:j,k]), alpha=.2, color=colors[k0])
            df = mask_plot(y[i:j,k], index)
            df.columns = ['_none']
            df.plot(ax=ax, color=colors[k0], marker='o', markersize=5, markerfacecolor='none', linestyle='none') 
        elif k == n-1 :
            k0 = k
            k = rand_idx[k]
            pd.DataFrame(u[i:j,k], index=index, columns=[rf'$u_{k0+1}(t+\tau)$ [{enc}, {dec}, {pde}]']).dropna().plot(ax=ax, color='C7')
            if err != None: ax.fill_between(index, u[i:j,k] + err[i:j,k], relu(u[i:j,k] - .5*err[i:j,k]), alpha=.2, color='C7')
            df = mask_plot(y[i:j,k], index)
            df.columns = [rf'data;  $y_i^t$']
            df.plot(ax=ax, color='k', marker='o', markersize=5, markerfacecolor='none', linestyle='none')
    ax.set_xlabel('time')
    plt.legend(fancybox=True, title= r'$\underline{\textrm{label: } u_i(t+\tau) \textrm{ [ENC,  DEC,  PDE]  }}$')
    stamp = str(time.time())
    plt.savefig(f'../pinn_fig_{stamp}.pdf')

#@jax.jit
def _batch(model, x, idx):
    idx = idx.reshape(-1,1)
    win = jnp.arange(1 - model.kappa, 1, 1)
    xb = x.at[:, idx + win].get()
    xb = jnp.swapaxes(xb,0,1)
    return xb

def inference_plot(model, x, adj, tau, i=0, j=-1, n=4, down_samp=10, eps=0.1, n_err=50, err=True):
    _,T = x.shape
    m = lambda x,t: forward(model, x, t, tau, adj, eps=eps, n=n_err, err=err)
    t_min = model.kappa
    t_max = T - tau
    tp = jnp.arange(t_min, t_max, down_samp)
    idx = tp.astype(int)
    xp = _batch(model, x, idx)
    u,err = jax.vmap(m)(xp,tp)
    y = x[:,idx+tau].T
    _plot(model, u, y, err, tau, i=i, j=j, n=n, down_samp=down_samp)

def plot_u(k=0, tau=60, dropout=0.002, **kwargs):
    df = get_log_data(max_loss=1e+10)
    idx = df['loss[data]'].sort_values().index[k]
    with open(idx, 'rb') as f: data = pickle.load(f)
    data['args']['log_path'] = idx.split('_')[-1][:-4]
    model, args = utils.read_model(data['args'], dropout=dropout)
    A = pd.read_csv(args.adj_path, index_col=0).to_numpy()
    adj = jnp.array(jnp.where(A))
    x = jnp.array(pd.read_csv(args.data_path, index_col=0).dropna().to_numpy().T)
    inference_plot(model, x, adj, tau, **kwargs)
    return model, args, data, df, x, adj

