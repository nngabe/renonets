from hgcn.models.encoders import HNN,HGCN
from config import parser
import torch

if __name__ == '__main__':
    args = parser.parse_args()
    args.dec_dims[0] = sum(args.enc_dims)
    args.num_layers
    enc = HGCN(1.,args)
    dec = HNN(1.,args)
    n = 100
    x,adj = torch.rand([10,60]),torch.randint(0,10,(2,10),dtype=torch.int64)
    A = torch.sparse_coo_tensor(adj,torch.ones(adj.shape[1]),size=(10,10))
    
    z = enc.encode(x,A)
