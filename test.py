from hgcn import optimizers
from hgcn.models import models
from config import parser
import torch
import pandas as pd
import warnings
from torch.autograd.functional import jacobian
from torch.optim import Adam
import sys
warnings.filterwarnings('ignore')

t2df = lambda tensor: pd.DataFrame(tensor.detach().numpy())

@torch.no_grad()
def get_series():
    res = torch.zeros_like(xx)
    for t in range(1,T):
        x = xx[:,int(t):int(t)+tau]
        z = enc(x,A)
        G = dec(z,A,t)
        res[:,t] = G.flatten()
    g = res[:,1:-60]
    return g

def plot(g,P,i=0,j=0):
    if j==0: j = g.shape[1]
    ax = t2df(g.T).iloc[i:j,:4].plot()
    idx = range(i,j)
    y = [P(i) for i in idx]
    pd.DataFrame(y,index=idx).plot(ax=ax)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim
    enc = getattr(models, args.encoder)(1., args)
    dec = getattr(models, args.decoder)(1., args)
    T = 2000
    n = 100
    tau = 60
    xx = torch.zeros([n, tau + T]) 
    adj = torch.randint(0, n, (2,n), dtype=torch.int64)
    A = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), size=(n,n))

    c = torch.nn.Parameter(torch.zeros(3).sigmoid(),requires_grad=True)
    model = torch.nn.ParameterList([enc,dec,c])
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
 
    w = torch.zeros(2)
    lam = torch.tensor(1e-4)
    for i in range(args.epochs):
        w = torch.ones(2) * (1. - torch.exp(-lam*i))
        ti = torch.randint(1,T,(1,30),dtype=torch.float,requires_grad=True)[0]
        loss = torch.tensor(0.) 
        optimizer.zero_grad()
        for t in ti:
            t.retain_grad()
            x = xx[:,int(t):int(t)+tau]
            z = enc(x,A)
            #z.retain_grad()
            G = dec(z,A,t)
            G.retain_grad()
            h = lambda t: (.01*t**2.+t)
            y = h(t) * torch.ones_like(G)
            loss_sq = (G-y).square().sum()
            
            #pinn
            grad_t = torch.tensor([torch.autograd.grad(g,t,retain_graph=True)[0] for g in G.unbind()])
            f = grad_t-c[0]-c[1]*t-c[2]*t**2
            #grad_f = torch.tensor([torch.autograd.grad(ff,t,retain_graph=True)[0] for ff in f.unbind()])
            loss_pde = w[0] * f.square().sum()
            loss_gpde = w[1] #* grad_f.square().sum()
            loss_l1 = 0.#c.abs().sum() * 1e+2
            loss += loss_pde + loss_sq + loss_gpde + loss_l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_norm)
        optimizer.step()

        if i % args.log_freq == 0:
            print(f'w = {w}')
            print(f'{i}/{args.epochs}: loss_sq = {loss_sq:.4e}, loss_pde = {loss_pde:.4e}, loss_gpde = {loss_gpde:.4e}, c = {c.detach().numpy().round(4)[:3]}')

    g = get_series()
    plot(g,h)

