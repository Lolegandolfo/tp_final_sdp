"""
hyperplane_coloring.py
======================
Implements the Hyperplane Rounding algorithm from Section 6 of:

  Karger, Motwani & Sudan (1998) — "Approximate Graph Coloring by
  Semidefinite Programming"

Algorithm
---------
Given unit vectors v_i (from the SDP solution):

  1. Sample t random Gaussian vectors  r_1, …, r_t ~ N(0, I_n).
  2. Each vertex i receives a **t-bit signature**:
       σ(i) = ( sgn(v_i · r_1), …, sgn(v_i · r_t) )
  3. All vertices sharing the same signature form a candidate color class.
  4. Because adjacent vertices have v_i · v_j ≤ −1/(k−1) < 0, they tend
     to have *different* signatures, but not always — edges within a class
     must be repaired by a sequential greedy coloring step.

Approximation guarantee (Thm 6.2):
  For 3-colorable G with t = ⌈log₃(n)⌉ random hyperplanes, the expected
  number of colors is  O(Δ^{log₃ 2} log n).

This file provides two functions:
  - hyperplane_coloring:  the core single-round algorithm.
  - hyperplane_coloring_multi_trial: best of several random trials.
"""

from __future__ import annotations

import math
import numpy as np
import networkx as nx


def _choose_t(G: nx.Graph, k_hint: int = 3) -> int:
    """
    Choose the number of random hyperplanes t.

    From the paper (Section 6, proof of Theorem 6.2):
      t = ⌈log₂(k)⌉ + 2·⌈log₂(log₂(Δ))⌉

    where k is the vector chromatic number hint and Δ is the maximum degree.
    For very small Δ (≤ 4) we fall back to ⌈ln(n)/ln(k/(k-1))⌉.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 1

    max_deg = max(dict(G.degree()).values()) if G.number_of_edges() > 0 else 1
    max_deg = max(max_deg, 2)

    if max_deg > 4:
        # Paper formula: t = ceil(log2(k)) + 2*ceil(log2(log2(Δ)))
        log2_k = math.ceil(math.log2(max(k_hint, 2)))
        log2_delta = math.log2(max_deg)
        log2_log2_delta = math.ceil(math.log2(max(log2_delta, 1)))
        t = log2_k + 2 * log2_log2_delta
    else:
        # Fallback for very sparse graphs
        base = k_hint / (k_hint - 1)    # e.g. 1.5 for k=3
        t = math.ceil(math.log(n) / math.log(base))

    return max(t, 1)


def hyperplane_coloring(
    G: nx.Graph,
    vectors: np.ndarray,
    nodes: list,
    t: int = None,
    k_hint: int = 3,
    seed: int = None,
    verbose: bool = False,
) -> dict:
    """
    Color G using the hyperplane rounding algorithm.

    Args:
        G:       Original NetworkX graph.
        vectors: ndarray shape (n, dim) — row i is the unit vector for nodes[i].
        nodes:   Ordered list of nodes corresponding to vector rows.
        t:       Number of random hyperplanes; auto-chosen if None.
        k_hint:  Used only to choose t automatically (default 3).
        seed:    Random seed.
        verbose: Print info.

    Returns:
        color_dict: Mapping node → color (integer).
    """
    rng = np.random.default_rng(seed)

    n = len(nodes)
    dim = vectors.shape[1]

    if t is None:
        t = _choose_t(G, k_hint)

    if verbose:
        print(f"Hyperplane coloring: n={n}, dim={dim}, t={t} hyperplanes")

    # ---- Step 1: sample t random hyperplanes --------------------------------
    R = rng.standard_normal((t, dim))          # shape (t, dim)
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    R = R / norms

    # ---- Step 2: compute t-bit signatures -----------------------------------
    # projections[i, l] = v_i · r_l
    projections = vectors @ R.T               # shape (n, t)
    # Convert to sign bits (0 or 1)
    signs = (projections >= 0).astype(int)    # shape (n, t)

    # Signature: tuple of t bits → candidate color class
    sig_to_class = {}
    for i, v in enumerate(nodes):
        sig = tuple(signs[i])
        if sig not in sig_to_class:
            sig_to_class[sig] = []
        sig_to_class[sig].append(v)

    if verbose:
        print(f"  {len(sig_to_class)} candidate classes from {2**t} possible signatures")

    # ---- Step 3-4: repair monochromatic edges within each class -------------
    # Assign each candidate class an initial color; then fix conflicts inside.
    color_dict = {}
    current_color = 0

    for sig, class_vertices in sig_to_class.items():
        if len(class_vertices) == 1:
            color_dict[class_vertices[0]] = current_color
            current_color += 1
            continue

        # Greedy coloring of the sub-graph induced by this class
        subG = G.subgraph(class_vertices)
        # Sequential greedy: assign the lowest available color offset
        sub_colors = {}
        for v in class_vertices:
            neighbor_colors = {sub_colors[nb] for nb in subG.neighbors(v)
                               if nb in sub_colors}
            c = 0
            while c in neighbor_colors:
                c += 1
            sub_colors[v] = c

        max_sub = max(sub_colors.values()) if sub_colors else 0
        for v, c in sub_colors.items():
            color_dict[v] = current_color + c
        current_color += max_sub + 1

    return color_dict


def hyperplane_coloring_multi_trial(
    G: nx.Graph,
    vectors: np.ndarray,
    nodes: list,
    num_trials: int = 5,
    t: int = None,
    k_hint: int = 3,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run hyperplane_coloring multiple times and return the best result.

    Args:
        G:          Graph.
        vectors:    Unit vectors from SDP.
        nodes:      Node ordering.
        num_trials: Number of independent trials.
        t:          Hyperplanes per trial (auto if None).
        k_hint:     Used to auto-select t.
        seed:       Base seed (each trial = seed + trial_index).
        verbose:    Show per-trial info.

    Returns:
        Best color_dict across trials.
    """
    best = None
    best_k = math.inf

    for trial in range(num_trials):
        coloring = hyperplane_coloring(
            G, vectors, nodes, t=t, k_hint=k_hint,
            seed=seed + trial, verbose=False
        )
        k = len(set(coloring.values()))
        if verbose:
            print(f"  Trial {trial+1}/{num_trials}: {k} colors")
        if k < best_k:
            best_k = k
            best = coloring

    if verbose:
        print(f"  Best: {best_k} colors")

    return best
