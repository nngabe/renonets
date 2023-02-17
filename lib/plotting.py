import jax
import jax.numpy as jnp
import pickle,glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20,8]; plt.rcParams['font.size'] = 14; plt.rcParams['xtick.major.size'] = 8
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'; #plt.rcParams['text.usetex'] = True

max_loss = 9e+4
n_files = 40
y_name = 'loss[pde]'

if __name__ == '__main__':

    keys =  ['x_dim', 'manifold', 'weight_decay', 'path', 'g', 'epochs']
    files = glob.glob('../eqx_models/*.pkl'); files.sort()
    dargs = {}
    for file in files[-n_files:]:
        with open(file,'rb') as f:
            data = pickle.load(f)
        loss = data['loss'][max(list(data['loss'].keys()))]
        args = data['args']
        if sum([k in args for k in keys]) < len(keys) or loss[0] > max_loss:
            continue
        dargs[file] = {key:args[key] for key in keys}
        dargs[file]['loss[data]'] = loss[0].item()
        dargs[file]['loss[pde]'] = loss[1].item()
        #print(f'\n {dargs[file]}')
        #print(f'    loss[data,pde] = ({loss[0]:.3e},{loss[1]:.3e})')

    df = pd.DataFrame(dargs).T
    g_dict = {None:r'$g = \tilde{g}(u;x,z,t)$','null':r'$g = 0$'}
    g_dict = {None:r'$g = \tilde{g}(u;x,z,t)$','null':r'$g = 0$'}; df.g = df.g.apply(lambda x: g_dict[x])
    grouped = df.groupby(['manifold', 'g'])
    fig, ax = plt.subplots(1,2)
    for key in grouped.groups.keys():
        grouped.get_group(key).groupby('x_dim').min(y_name).plot(y='loss[data]', ax=ax[0], legend=' '.join(key), logy=True)
        grouped.get_group(key).groupby('x_dim').min(y_name).plot(y='loss[pde]', ax=ax[1], legend=' '.join(key), logy=True)
    plt.ylim(df[y_name].min(), 1.1*10**jnp.round(jnp.log10(df[y_name].max())) )
    legend_data = [r'Loss$_{data}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    legend_pde = [r'Loss$_{pde}$: ' + ', '.join(k) for k in grouped.groups.keys()]
    ax[0].legend(legend_data)
    ax[1].legend(legend_pde)
