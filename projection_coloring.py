"""
projection_coloring.py
======================
Implements the Projection Rounding algorithm from Section 7 of:

  Karger, Motwani & Sudan (1998) — "Approximate Graph Coloring by
  Semidefinite Programming"

Algorithm (for k-colorable G, typically k=3)
--------------------------------------------
Repeat until all vertices are colored:

  1. Sample a random Gaussian vector r ~ N(0, I_n).
  2. Project each uncolored vertex i onto r:  x_i = v_i · r.
  3. Compute threshold  c = sqrt(2 ln Δ),  where Δ is the max degree
     in the *remaining* subgraph.
  4. Let S = { i : x_i ≥ c }.  Vertices in S tend to have large
     positive projections, meaning their vectors point "in the direction
     of r", which geometrically means they are clustered together —
     but edge constraints (v_i · v_j ≤ −1/(k−1) < 0) ensure neighbors
     are *not* in S simultaneously (with high probability for large Δ).
  5. Find a maximal independent set I ⊆ S in the subgraph G[S].
  6. Assign a fresh color to I and remove I from the remaining graph.

Approximation guarantee (Thm 7.3):
  For 3-colorable G,  O(Δ^{1/3} √(ln Δ) log n) colors are used.
"""

from __future__ import annotations

import math
import random
import numpy as np
import networkx as nx

from graph_utils import maximal_independent_set_in_subgraph


def projection_coloring(
    G: nx.Graph,
    vectors: np.ndarray,
    nodes: list,
    seed: int = None,
    max_rounds: int = None,
    verbose: bool = False,
) -> dict:
    """
    Color G using the projection rounding algorithm.

    Args:
        G:          Original NetworkX graph.
        vectors:    ndarray shape (n, n) — row i is the unit vector for nodes[i].
        nodes:      Ordered list of nodes corresponding to vector rows.
        seed:       Random seed for reproducibility.
        max_rounds: Safety limit on the number of coloring rounds (default: 10*n).
        verbose:    Print progress information.

    Returns:
        color_dict: Mapping  node → color (0-indexed integer).
    """
    rng = np.random.default_rng(seed)

    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    if max_rounds is None:
        max_rounds = 10 * n

    # Work on a copy so we can remove vertices
    remaining = nx.Graph(G)
    color_dict = {}
    current_color = 0

    round_num = 0
    while remaining.number_of_nodes() > 0 and round_num < max_rounds:
        round_num += 1

        remaining_nodes = list(remaining.nodes())
        n_rem = len(remaining_nodes)

        # Threshold c = sqrt(2 ln Δ);  Δ = max degree in remaining subgraph
        degrees = dict(remaining.degree())
        max_deg = max(degrees.values()) if degrees else 1
        max_deg = max(max_deg, 1)                      # avoid log(0)

        c = math.sqrt(2.0 * math.log(max_deg + 1))    # +1 for isolated vertices

        # ---- Step 1-2: random projection ------------------------------------
        dim = vectors.shape[1]
        r = rng.standard_normal(dim)
        r /= np.linalg.norm(r)

        # Projections for uncolored vertices
        proj = {}
        for v in remaining_nodes:
            i = node_to_idx[v]
            proj[v] = float(np.dot(vectors[i], r))

        # ---- Step 3-4: threshold set S ------------------------------------
        S = {v for v, x in proj.items() if x >= c}

        if len(S) == 0:
            # Fallback when c is too large (e.g. small low-degree graphs):
            # take the top 1/(k-1+1) fraction of vertices by projection value.
            # This mirrors how the paper expects Δ to be large.
            n_take = max(1, n_rem // 3)
            sorted_verts = sorted(proj, key=proj.get, reverse=True)
            S = set(sorted_verts[:n_take])

        if verbose:
            print(
                f"Round {round_num:3d} | remaining={n_rem:4d} | "
                f"Δ={max_deg:3d} | c={c:.3f} | |S|={len(S):4d}"
            )

        # ---- Step 5: maximal independent set in G[S] -----------------------
        ind_set = maximal_independent_set_in_subgraph(remaining, S)

        # ---- Step 6: color and remove --------------------------------------
        for v in ind_set:
            color_dict[v] = current_color

        remaining.remove_nodes_from(ind_set)
        current_color += 1

    # Fallback: any uncolored vertex (shouldn't happen under max_rounds)
    for v in remaining.nodes():
        color_dict[v] = current_color
        current_color += 1

    return color_dict


def projection_coloring_multi_trial(
    G: nx.Graph,
    vectors: np.ndarray,
    nodes: list,
    num_trials: int = 5,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run projection_coloring multiple times and return the best (fewest-color) result.

    Due to the randomness in projection direction r, different trials may
    yield different numbers of colors. This helper picks the best.

    Args:
        G:          Graph.
        vectors:    Unit vectors from SDP.
        nodes:      Node ordering.
        num_trials: Number of independent trials.
        seed:       Base random seed (each trial uses seed+trial_index).
        verbose:    Show per-trial info.

    Returns:
        Best color_dict found across all trials.
    """
    best = None
    best_k = math.inf

    for t in range(num_trials):
        coloring = projection_coloring(
            G, vectors, nodes, seed=seed + t, verbose=False
        )
        k = len(set(coloring.values()))
        if verbose:
            print(f"  Trial {t+1}/{num_trials}: {k} colors")
        if k < best_k:
            best_k = k
            best = coloring

    if verbose:
        print(f"  Best: {best_k} colors")

    return best
