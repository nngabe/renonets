import warnings
warnings.filterwarnings('ignore')

## plotting preferences
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = [8,8]; plt.rcParams['font.size'] = 24; plt.rcParams['xtick.major.size'] = 8
#plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; plt.rcParams['text.usetex'] = True

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

from gen_hyp_graph import HyperbolicGraphSimulation as HGS

parser = argparse.ArgumentParser()
parser.add_argument('--precision', nargs='?', default=1e-16, type=float)
parser.add_argument('--K', nargs='?', default=5., type=float)
parser.add_argument('--steps', nargs='?', default=30000, type=int)
parser.add_argument('--dt', nargs='?', default=1e-3, type=float)
parser.add_argument('--save', nargs='?', default=0, type=int)
parser.add_argument('--seed', nargs='?', default=123, type=int)
args = parser.parse_args()

hgs = HGS()
hgs.setOutputFolder(os.getcwd()+'/out/')
avg_degree = 3.
c=1.
N=1000

G = nx.Graph()
G.add_edges_from(pd.read_csv('edges_N905.csv',index_col=0).to_numpy().T)
print(f' N = {G.number_of_nodes()}')
print(f'|E| = {G.number_of_edges()}')



#hgs.generateGraph(N, avg_degree, c)
lcc = max(nx.connected_components(G), key=len)
G = G.subgraph(lcc)
re = {old:new for new,old in enumerate(lcc)}
print(f'c = {c}, N = {G.number_of_nodes()}')
partition = community_louvain.best_partition(G)

p = partition
com = defaultdict(list)
for key, value in p.items():
    com[value].append(re[key])


from jax.config import config; config.update("jax_enable_x64", True)
np.random.seed(123)

A = np.array(nx.adjacency_matrix(G).todense())
s,r = np.where(A)
N = A.shape[0]


sigma = 1.
omega_i =  np.clip(np.random.normal(.0, sigma, N), -2*sigma, 2*sigma)
alpha = .9
for i in range(2): omega_i = alpha * omega_i + (1.-alpha) * jraph.segment_mean(omega_i[s],r,N)
omega_i -= omega_i.mean()
print(f'E[omega] = {omega_i.mean():.3f}, Var[omega] = {omega_i.std():.3f}')

K = args.K
eps = .00
dt = args.dt
steps = args.steps

x = np.zeros((N,steps))
x[:,0] =  1. * np.random.normal(0.,1.,N)

w = (A.sum(1)/A.shape[0])**.5
w_ij = w[s]/w.max()

@jax.jit
def f(x,t):
    m_ij = jnp.sin(x[s] - x[r])
    msg_i = K * jraph.segment_mean(m_ij*w_ij, r, N)
    return omega_i +  msg_i

tic = time.time()
key = jax.random.PRNGKey(0)

x = ode.odeint(f, x[:,0], dt*jnp.arange(0,steps), rtol=args.precision, atol=args.precision)
print(f'compute time (jax DP): {time.time()-tic:.3f} (sec)'); tic = time.time()

x = np.array(x).T
ax = pd.DataFrame(x.T).diff().iloc[steps//4::20,:50].plot(legend=False,logx=False)
ax.set_ylabel(r'$\theta_i$', rotation=90, fontsize=30); ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
plt.savefig(f'../../kur_K{K}_N{A.shape[0]}_c{c}_tol{args.precision}.pdf')

R = np.exp(1j*x).mean(0); Rt = np.abs(R)
ax = pd.DataFrame(Rt).plot(logx=False,logy=True,legend=False,ylim=[min(1e-2,.9*min(Rt)), 1.])
ax.set_ylabel(r'$R(t)$', rotation=90, fontsize=30,labelpad=5) ;ax.set_xlabel(r'$t$', rotation=0, fontsize=30)
plt.savefig(f'../../order_K{K}_N{A.shape[0]}_c{c}_tol{args.precision}.pdf')


dx = pd.DataFrame(x).diff(axis=1).dropna(axis=1).to_numpy()
dx = dx-dx.mean(1).reshape(-1,1)
dx = dx[:,dx.shape[1]//5:]

k = 4
def fftc(idx):
    n = int((dx.shape[1]) * k) # scipy.fft.next_fast_len(1*dx.shape[1])
    window = np.blackman(dx.shape[1])
    #conv = lambda x: np.convolve(x,window)
    conv = lambda x: x*window
    cdx = np.array([conv(di) for di in dx[idx]])
    return np.abs(scipy.fft.fft(cdx, n=n, norm=None,workers=-1).mean(0))

idx = [2,4,5]
i_c = [com[i] for i in idx]
m = int(120*k)
amp = [fftc(i)[:m] for i in i_c]
colors = ['purple', 'forestgreen', 'royalblue', 'crimson']*2
for i,a in enumerate(amp):
    df = pd.DataFrame(a)
    df.index = df.index / k
    num_nodes = len(com[idx[i]])
    df.columns = [f'$c_{i}, k_{i}={num_nodes}$']
    if i==0: ax = df.iloc[:].plot(logy=True, color=colors[i])
    else: df.iloc[:].plot(logy=True, ax=ax, color=colors[i])
ax.set_ylabel(r'$\textrm{amplitude}$', rotation=90, fontsize=30,labelpad=5) 
ax.set_xlabel(r'$\textrm{wavenumber}$', rotation=0, fontsize=30)
ymin,ymax=ax.get_ybound()
ax.set_ylim([min(1e-2,.9*ymin),ymax])

textstr = f'$K = {K:.0f}$'
props = dict(boxstyle='square', facecolor='white', alpha=1.)
ax.text(0.75, 0.65, textstr, transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig(f'../../spectrum_K{K}_N{A.shape[0]}_c{c}_tol{args.precision}.pdf')

#pd.DataFrame(A).to_csv(f'adj_K{K}_e{eps}_N{A.shape[0]}_c{c}.csv')
if args.save: pd.DataFrame(x.T,columns=omega_i).to_csv(f'../../kur_K{K}_e{eps}_N{A.shape[0]}_c{c}_tol{args.precision}.csv')

