"""use networkx for graph plotting"""
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def corr_plot(corr_mat, file_name, categories, node_names=None, layout=None):
    """plot network graph using correlation as inverse distance"""
    length = corr_mat.shape[0]
    graph = nx.from_numpy_matrix(corr_mat)
    if not layout:
        layout = nx.spring_layout(graph)
    if node_names is not None and len(node_names) == length:
        nx.relabel_nodes(graph, dict(zip(range(length), node_names)), copy=False)
    indices = {name: idx for idx, name in enumerate(node_names)}
    positive_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if corr_mat[indices[u], indices[v]] > 0.3]
    positive_weights = [corr_mat[indices[u], indices[v]] for u, v, _ in positive_edges]
    negative_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if corr_mat[indices[u], indices[v]] < -0.1]
    negative_weights = [-corr_mat[indices[u], indices[v]] for u, v, _ in negative_edges]
    plt.figure()
    nx.draw_networkx_nodes(graph, layout, node_size=200, nodelist=list(categories['good']),
                           node_color='#EA120D')  # red
    nx.draw_networkx_nodes(graph, layout, node_size=200, nodelist=list(categories['unrelated']),
                           node_color='#0AB815')  # green
    nx.draw_networkx_edges(graph, layout, edgelist=negative_edges, edge_color=negative_weights,
                           edge_vmin=0, edge_vmax=0.3,
                           edge_cmap=plt.cm.Blues)  # pylint:disable=no-member
    nx.draw_networkx_edges(graph, layout, edgelist=positive_edges, edge_color=positive_weights,
                           edge_vmin=0.3, edge_vmax=0.7,
                           edge_cmap=plt.cm.Reds)  # pylint:disable=no-member
    plt.savefig(file_name)
    return layout


def scatter_plot(corr_mat, file_name, categories, layout=None):
    """plot network with no edges by weight"""
    graph = nx.from_numpy_matrix(corr_mat)
    if not layout:
        layout = nx.spring_layout(graph, iterations=150)
    nx.draw_networkx_nodes(graph, layout, node_size=10,
                           nodelist=list(np.flatnonzero(categories['unrelated'])),
                           node_color='#0AB815')  # green
    nx.draw_networkx_nodes(graph, layout, node_size=10,
                           nodelist=list(np.flatnonzero(categories['good'])),
                           node_color='#EA120D')  # red
    plt.savefig(file_name)
    return layout


def get_layout(weight_mat, node_names=None):
    graph = nx.from_numpy_matrix(np.power(weight_mat, 2) * 5)
    if node_names is not None and len(node_names) == weight_mat.shape[0]:
        nx.relabel_nodes(graph, dict(zip(range(weight_mat.shape[0]), node_names)), copy=False)
    return nx.spring_layout(graph, iterations=150)
