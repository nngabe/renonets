from typing import Union, List

import jax
import jax.numpy as jnp


def index_to_mask(index, size):
    
    if not isinstance(index, jnp.ndarray): index = jnp.array(index)

    mask = jnp.zeros((size), dtype=bool)
    mask = mask.at[index].set(True)
    return mask

def subgraph(
    subset: Union[jnp.ndarray, List[int]],
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray = None,
    relabel_nodes: bool = True
    ):

    if not isinstance(subset, jnp.ndarray): subset = jnp.array(subset)

    num_nodes = jnp.unique(jnp.concatenate(edge_index)).size
    node_mask = index_to_mask(subset, size=num_nodes)
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = jnp.zeros(node_mask.size, dtype=jnp.int32)
        node_idx = node_idx.at[subset].set( jnp.arange(subset.shape[0]) )
        edge_index = node_idx[edge_index]

    return edge_index, edge_attr 

def batch_graph(
    nodes: Union[jnp.ndarray, List[int]], 
    edge_index: jnp.array, 
    batch_size: int = 100,
    relabel_nodes: bool = True
    ):

    if not isinstance(nodes, jnp.ndarray): nodes = jnp.array(nodes)
    
    num_nodes = jnp.unique(jnp.concatenate(edge_index)).size
    node_mask = index_to_mask(nodes, num_nodes)
    while nodes.size < batch_size:
        edge_mask = node_mask[edge_index[0]]
        edge_idx = edge_index[:,edge_mask]
        nodes = jnp.unique(jnp.concatenate(edge_idx))
        node_mask = node_mask.at[nodes].set(True)
    nodes = nodes[:batch_size]
    node_mask = index_to_mask(nodes, num_nodes)
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_idx = edge_index[:, edge_mask]
    
    if relabel_nodes:
        node_idx = jnp.zeros(node_mask.size, dtype=jnp.int32)
        node_idx = node_idx.at[nodes].set( jnp.arange(nodes.shape[0]) )
        edge_idx = node_idx[edge_idx]

    return nodes, edge_idx

def remove_nodes(
    nodes: Union[jnp.ndarray, List[int]], 
    edge_index: jnp.array,
    relabel_nodes: bool = True
    ):

    if not isinstance(nodes, jnp.ndarray): nodes = jnp.array(nodes)
    
    num_nodes = jnp.unique(jnp.concatenate(edge_index)).size
    node_mask = ~index_to_mask(nodes, num_nodes)
        
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_idx = edge_index[:,edge_mask]
    nodes = jnp.unique(jnp.concatenate(edge_idx))
    
    if relabel_nodes:
        node_idx = jnp.zeros(node_mask.size, dtype=jnp.int32)
        node_idx = node_idx.at[nodes].set( jnp.arange(nodes.shape[0]) )
        edge_idx = node_idx[edge_idx]

    return nodes, edge_idx

def to_undirected( 
    edge_index: jnp.ndarray
    ):

    tmp = jnp.concatenate([edge_index, edge_index[::-1]],axis=1)
    _, idx = jnp.unique(tmp.T, return_index=True, axis=0)
    tmp = tmp[:,idx]

    return tmp

def add_self_loops(
    edge_index: jnp.ndarray
    ):

    idx = jnp.unique(jnp.concatenate(edge_index))
    self_loops = jnp.array([idx,idx])
    tmp = jnp.concatenate([edge_index,self_loops], axis=1)
    _, reidx = jnp.unique(tmp.T, return_index=True, axis=0)
    tmp = tmp[:,reidx]

    return tmp

