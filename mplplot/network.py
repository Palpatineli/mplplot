"""use networkx for graph plotting"""
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
from networkx import draw_networkx_nodes as nodes, draw_networkx_edges as edges
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap

Layout = Dict[int, np.ndarray]
Categories = List[Tuple[np.ndarray, str]]


def scatter_plot(ax: Axes, corr_mat, file_name, categories: Categories, layout: Layout = None) -> Layout:
    """plot network with no edges by weight"""
    graph = nx.from_numpy_matrix(corr_mat)
    layout = (nx.spring_layout(graph, iterations=150) if layout is None else layout)
    for category, color in categories:
        nodes(graph, layout, node_size=10, nodelist=list(category), node_color=color, ax=ax)
    return layout

def get_layout(weight_mat: np.ndarray, node_names=None) -> Layout:
    graph = nx.from_numpy_matrix(np.power(weight_mat, 2) * 5)
    if node_names is not None and len(node_names) == weight_mat.shape[0]:
        nx.relabel_nodes(graph, dict(zip(range(weight_mat.shape[0]), node_names)), copy=False)
    return nx.spring_layout(graph, iterations=150)

def corr_plot(ax: Axes, corr_mat: np.ndarray, categories: Categories = None, node_names: List[str] = None,
              layout: Layout = None) -> Layout:
    """Plot network graph using correlation as inverse distance."""
    length = corr_mat.shape[0]
    graph = nx.from_numpy_matrix(corr_mat)
    if node_names is not None and len(node_names) == length:
        nx.relabel_nodes(graph, dict(zip(range(length), node_names)), copy=False)
    layout = (nx.spring_layout(graph) if layout is None else layout)
    if categories is None:
        nodes(graph, layout, node_size=200, ax=ax)
    else:
        for node_list, color in categories:
            nodes(graph, layout, node_size=200, nodelist=list(node_list), node_color=color, ax=ax)
    for classifier, lims, color in ((lambda x: x[2]['weight'] > 0.3, (0.3, 0.7), get_cmap('Reds')),
                                    (lambda x: x[2]['weight'] < -0.1, (-0.3, -0.1), get_cmap('Blues_r'))):
        edge_list = list(filter(classifier, graph.edges(data=True)))
        weight_list = list(map(lambda x: x[2]['weight'], edge_list))
        edges(graph, layout, edgelist=edge_list, edge_color=weight_list, edge_vmin=lims[0], edge_vmax=lims[1],
              edge_cmap=color, ax=ax)
    return layout
