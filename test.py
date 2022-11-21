from hgcn.models.models import HNN,HGCN
from config import parser
import torch

if __name__ == '__main__':
    args = parser.parse_args()
    if args.skip: 
        args.dec_dims[0] = sum(args.enc_dims) + args.time_enc[1] * args.time_dim
    else: 
        args.dec_dims[0] = args.enc_dims[-1] + args.time_enc[1] * args.time_dim
    enc = HGCN(1., args)
    dec = HNN(1., args)
    T = 2000
    n = 1000
    tau = 60
    xx = torch.rand([n, tau + T]) 
    adj = torch.randint(0, n, (2,n), dtype=torch.int64)
    A = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), size=(n,n))
    
    for i in range(args.epochs):
        ti = torch.randint(1,T,(1,5))[0]
        for t in ti:
            x = xx[:,int(t):int(t)+tau]
            z = enc(x,A)
            G = dec(z,A,t)
            y = t*torch.ones_like(G)

        if i % args.log_freq == 0:
            print(f'{i}/{args.epochs}')
