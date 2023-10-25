import warnings
warnings.filterwarnings('ignore')

## plotting preferences
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,8]; plt.rcParams['font.size'] = 24; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [8,8]; plt.rcParams['font.size'] = 24; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True

import argparse
import time
import pandas as pd
import networkx as nx
import numpy as np
import glob
import os
from collections import defaultdict

from community import community_louvain
import jax
import jax.numpy as jnp
from jax.experimental import ode
import jraph
import scipy

from graph_utils import gen_graph

from jax.config import config; config.update("jax_enable_x64", True)
np.random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--precision', nargs='?', default=1e-16, type=float)
parser.add_argument('--N', nargs='?', default=10000, type=int)
parser.add_argument('--c', nargs='?', default=1., type=float)
parser.add_argument('--K', nargs='?', default=5., type=float)
parser.add_argument('--steps', nargs='?', default=100000, type=int)
parser.add_argument('--dt', nargs='?', default=1e-3, type=float)
parser.add_argument('--save', nargs='?', default=0, type=int)
parser.add_argument('--seed', nargs='?', default=123, type=int)
args = parser.parse_args()

tic = time.time()

avg_degree = 3.
c=1.0
N=10000
seed=71

G, part, re = gen_graph(N, avg_degree, c, seed=seed)

com = defaultdict(list)
for key, value in part.items():
    com[value].append(re[key])
    
edge_list = np.array(list(G.edges)).T
edge_file = f'../../kur/edges_c{c}_N{G.number_of_nodes()}_t{int(tic)}.csv'
pd.DataFrame(edge_list).to_csv(edge_file)

print(f' N = {G.number_of_nodes()}')
print(f'|E| = {G.number_of_edges()}')

A = np.array(nx.adjacency_matrix(G).todense())
s,r = np.where(A)
N = A.shape[0]


sigma = 1.
omega_i =  np.clip(np.random.normal(.0, sigma, N), -20*sigma, 20*sigma)
alpha = 1.
for i in range(2): omega_i = alpha * omega_i + (1.-alpha) * jraph.segment_mean(omega_i[s],r,N)
omega_i -= omega_i.mean()
print(f'E[omega] = {omega_i.mean():.3f}, Var[omega] = {omega_i.std():.3f}')

K = args.K
eps = .00
dt = args.dt
steps = args.steps
p=args.precision

x = np.zeros((N,steps+1))
x[:,0] =  5. * np.random.normal(0.,1.,N)

w = (A.sum(1)/A.shape[0])**.5
norm = jraph.segment_sum(w[s], r, N)
w_ij = w[s]/norm[r]

@jax.jit
def f(x,t):
    m_ij = jnp.sin(x[s] - x[r])
    msg_i = K * jraph.segment_sum(m_ij*w_ij, r, N)
    #U = jnp.abs(jnp.modf(1e+2*msg_i)[0])
    #zeta_i = eps * jax.scipy.special.ndtri(U)
    return omega_i + msg_i


key = jax.random.PRNGKey(0)

x = ode.odeint(f, x[:,0], dt*jnp.arange(0,steps), rtol=p, atol=p)
print(f'compute time (jax DP): {time.time()-tic:.3f} (sec)'); tic = time.time()
x = np.array(x).T

ax = pd.DataFrame(x.T).diff().iloc[steps//10::100,:40].plot(legend=False,logx=False)
ax.set_ylabel(r'$\theta_i$', rotation=90, fontsize=30); ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
plt.tight_layout()
plt.savefig(f'../../kur/kur_K{K}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.pdf')

def _R(x,i=None):
    x = x if i==None else x[i]
    R = np.exp(1j*x).mean(0)
    return np.abs(R)

colors = ['purple', 'forestgreen', 'royalblue', 'crimson']*2
R = _R(x)
ax = pd.DataFrame(R).plot(logx=False,logy=True,legend=False,ylim=[min(1e-2,.9*min(R)), 1.], color='k')
ax.set_ylabel(r'$R(t)$', rotation=90, fontsize=30,labelpad=5) ;ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
for j in range(0):
    i = com[j]
    R = _R(x,i)
    pd.DataFrame(R).plot(ax=ax, logx=False, logy=True, legend=False, ylim=[min(1e-2,.9*min(R)), 1.], color = colors[j])
    ax.set_ylabel(r'$R(t)$', rotation=90, fontsize=30,labelpad=5) ;ax.set_xlabel(r'$t$', rotation=0, fontsize=30)

plt.tight_layout()
plt.savefig(f'../../kur/order_K{K}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.pdf')

dx = pd.DataFrame(x).diff(axis=1).dropna(axis=1).to_numpy()
dx = dx-dx.mean(1).reshape(-1,1)
dx = dx[:,dx.shape[1]//5:]

k = 4.
def fftc(idx):
    n = int((dx.shape[1]) * k) # scipy.fft.next_fast_len(1*dx.shape[1])
    window = np.blackman(dx.shape[1])
    #conv = lambda x: np.convolve(x,window)
    conv = lambda x: x*window
    cdx = np.array([conv(di) for di in dx[idx]])
    return np.abs(scipy.fft.fft(cdx, n=n, norm=None,workers=-1).mean(0))

idx = [0,1,2]#list(np.random.choice(len(s), 3, False))
i_c = [com[i] for i in idx]
m = int(30*k)
amp = [fftc(i)[:m] for i in i_c]
colors = ['purple', 'forestgreen', 'royalblue', 'crimson']*2
for i,a in enumerate(amp):
    df = pd.DataFrame(a)
    df.index = df.index / k
    num_nodes = len(i_c[i])
    df.columns = [f'$ k_{i}={num_nodes}$']
    if i==0: ax = df.iloc[:].plot(logy=True, color=colors[i])
    else: df.iloc[:].plot(logy=True, ax=ax, color=colors[i])
ax.set_ylabel(r'$\textrm{amplitude}$', rotation=90, fontsize=30,labelpad=5) 
ax.set_xlabel(r'$\textrm{wavenumber}$', rotation=0, fontsize=30)
ymin,ymax=ax.get_ybound()
ax.set_ylim([min(1e-2,.9*ymin),ymax])

textstr = f'$K = {K:.0f}$'
props = dict(boxstyle='square', facecolor='white', alpha=1.)
ax.text(0.75, 0.69, textstr, transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig(f'../../kur/spectrum_K{K}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.pdf')

#pd.DataFrame(A).to_csv(f'../../kur/adj_K{K}_N{A.shape[0]}_c{c}_t{int(tic)}.csv')
pd.DataFrame(x.T[::10], columns=omega_i).to_csv(f'../../kur/kur_K{K}_e{eps}_N{A.shape[0]}_c{c}_p{p}_t{int(tic)}.csv')
