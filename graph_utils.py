"""
graph_utils.py
==============
Utility functions for graph generation, coloring validation, and visualization.

Used by the Karger-Motwani-Sudan approximate graph coloring implementation.
"""

import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def make_k_colorable_graph(k: int, n: int, p_intra: float = 0.0, p_inter: float = 0.5,
                            seed: int = 42) -> nx.Graph:
    """
    Generate a random k-colorable graph with n vertices.

    Partitions vertices into k equal groups (color classes).  Edges are added
    only between vertices in *different* groups, each with probability p_inter.
    Edges within a group (p_intra) default to 0, guaranteeing k-colorability.

    Args:
        k:         Number of color classes (chromatic number ≤ k).
        n:         Number of vertices.
        p_intra:   Probability of edges within the same color class (0 = truly k-colorable).
        p_inter:   Probability of edges between different color classes.
        seed:      Random seed for reproducibility.

    Returns:
        A NetworkX Graph together with a 'true_color' node attribute.
    """
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Assign ground-truth colors
    partition = {}
    for v in range(n):
        partition[v] = v % k
    nx.set_node_attributes(G, partition, "true_color")

    # Add inter-class edges
    for u in range(n):
        for v in range(u + 1, n):
            same = partition[u] == partition[v]
            prob = p_intra if same else p_inter
            if rng.random() < prob:
                G.add_edge(u, v)

    return G


def make_cycle_graph(n: int) -> nx.Graph:
    """Return the cycle graph C_n (2-colorable if n even, 3-colorable if n odd)."""
    return nx.cycle_graph(n)


def make_petersen_graph() -> nx.Graph:
    """Return the Petersen graph (3-colorable, 10 vertices)."""
    return nx.petersen_graph()


def make_complete_graph(k: int) -> nx.Graph:
    """Return K_k (the complete graph on k vertices, exactly k-colorable)."""
    return nx.complete_graph(k)


# ---------------------------------------------------------------------------
# Coloring validation
# ---------------------------------------------------------------------------

def is_valid_coloring(G: nx.Graph, color_dict: dict) -> bool:
    """
    Check that color_dict is a proper coloring of G.

    A proper coloring assigns each vertex a color such that no two adjacent
    vertices share the same color.

    Args:
        G:          NetworkX graph.
        color_dict: Mapping vertex -> color (any hashable).

    Returns:
        True if valid, False if any edge is monochromatic.
    """
    for u, v in G.edges():
        if color_dict.get(u) == color_dict.get(v):
            return False
    return True


def num_colors_used(color_dict: dict) -> int:
    """Return the number of distinct colors in color_dict."""
    return len(set(color_dict.values()))


def greedy_coloring_baseline(G: nx.Graph) -> dict:
    """
    Greedy graph coloring (networkx built-in, DSATUR strategy).

    Returns:
        color_dict mapping vertex -> int color.
    """
    return nx.coloring.greedy_color(G, strategy="DSATUR")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_coloring(G: nx.Graph, color_dict: dict, title: str = "Graph Coloring",
                  figsize: tuple = (8, 6), save_path: str = None) -> None:
    """
    Draw the graph with vertices colored by color_dict.

    Args:
        G:          NetworkX graph.
        color_dict: Mapping vertex -> int color index.
        title:      Plot title.
        figsize:    Matplotlib figure size.
        save_path:  If given, save figure to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors_list = [color_dict.get(v, 0) for v in G.nodes()]
    n_colors = max(colors_list) + 1
    cmap = cm.get_cmap("tab20", max(n_colors, 1))
    node_colors = [cmap(c) for c in colors_list]

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        with_labels=True,
        node_size=600,
        font_size=9,
        edge_color="#555555",
        ax=ax,
    )

    valid = is_valid_coloring(G, color_dict)
    k_used = num_colors_used(color_dict)
    status = "✓ Valid" if valid else "✗ Invalid"
    ax.set_title(f"{title}\n{k_used} colors — {status}", fontsize=13)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Independent set extraction (used by projection_coloring)
# ---------------------------------------------------------------------------

def maximal_independent_set_in_subgraph(G: nx.Graph, candidates: set) -> set:
    """
    Find a maximal independent set among candidate vertices in G.

    Uses a greedy approach: iterate over candidates in random order, add each
    vertex if none of its neighbors have already been selected.

    Args:
        G:          The full graph.
        candidates: Subset of vertices to consider.

    Returns:
        A set of vertices forming a maximal independent set within G[candidates].
    """
    independent = set()
    blocked = set()
    # Shuffle for randomness
    order = list(candidates)
    random.shuffle(order)
    for v in order:
        if v not in blocked:
            independent.add(v)
            blocked.update(G.neighbors(v))
    return independent
