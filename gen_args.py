import itertools
import sys

batch = sys.argv[1] if len(sys.argv)>1 else ''
OPTS = []

if batch == '':
    OUT_FILE = 'args/base.txt'
    opts = {'epochs': [20001], 'manifold': ['PoincareBall'], 'w_pde':[0., 10.], 'w_gpde': [0., 1e+5] , 'path': ['2982_c1']}
    OPTS.append(opts)
    opts = {'epochs': [20001], 'manifold': ['PoincareBall'], 'v_scaler':[.01] , 'path': ['2982_c1']}
    OPTS.append(opts)
elif batch == 'var':
    OUT_FILE = 'args/var.txt'
    opts = {'v_scaler': [1e-0, 1e-1, 1e-2, 1e-3, 1e-4], 'path': ['993_c1']}
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
