import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
OPTS = []

if batch == '':
    OUT_FILE = 'args/base.txt'
    opts = {'x_dim': [1,2,3,4,5,6,7,8,9,10], 'epochs': [8001], 'manifold': ['PoincareBall', 'Euclidean'], 'g': ['null', None], 'opt_study': [1]}
elif batch == 'var':
    OUT_FILE = 'args/var.txt'
    opts = {'encoder': ['HGCN'], 'x_dim': [2, 4, 8, 16, 24] }
    OPTS.append(opts)
    opts = {'encoder': ['HGAT'], 'x_dim': [4, 4, 8, 16, 24] }
    OPTS.append(opts)
else:
    print('argv[1] not recognized!')
    raise

print(f'out file: {OUT_FILE}')

if __name__ == '__main__':
    
    ### take the cartesian product of all opts and write 
    ### arg strings (e.g. --model MLP --dropout 0.6 ...) to file
    #OUT_FILE = sys.argv[1] if len(sys.argv)>1 else 'args.txt'
    
    if len(OPTS)==0:
        vals = list(itertools.product(*opts.values()))
        args = [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    elif len(OPTS)>0:
        vals = []
        args = []
        for opts in OPTS: 
            vals = list(itertools.product(*opts.values()))
            args += [''.join([f'--{k} {str(v[i])} ' for i,k in enumerate(opts)]) for v in vals]
    with open(OUT_FILE,'w') as fp: fp.write('\n'.join(args))
