"""
sdp_solver.py
=============
Solves the Vector k-Coloring SDP using MOSEK Fusion API.

Based on: Karger, Motwani & Sudan (1998) — "Approximate Graph Coloring
by Semidefinite Programming", Section 3–5.

SDP formulation
---------------
Given graph G = (V, E) with n = |V|:

  Variables : Gram matrix X ∈ S^n  (symmetric n×n)
              λ ∈ ℝ

  Minimize  : λ

  Subject to:
    X[i,i] = 1          for all i ∈ V          (unit vectors)
    X[i,j] ≤ λ          for all (i,j) ∈ E      (edge dot-product)
    X ⪰ 0                                        (positive semidefinite)

After solving: X = V^T V, so rows of the Cholesky factor give
               the n-dimensional unit vectors v_i assigned to vertices.

The vector chromatic number is  χ_v(G) = 1 - 1/λ*.
For a 3-colorable graph: λ* ≈ -1/2 → χ_v = 3.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix


def solve_vector_coloring(
    G: nx.Graph,
    verbose: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Solve the vector coloring SDP for graph G.

    Args:
        G:       NetworkX graph.
        verbose: If True, print MOSEK solver log.

    Returns:
        vectors  : ndarray of shape (n, n) — row i is the unit vector for vertex i.
        lam_opt  : The optimal value of λ (most negative = tighter coloring).
        X_gram   : The n×n Gram matrix (PSD solution).

    Raises:
        RuntimeError: If the SDP is infeasible or MOSEK returns an error.
    """
    n = G.number_of_nodes()
    # Re-index vertices to 0..n-1 in a stable order
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]

    with Model("VectorColoring") as M:
        if verbose:
            import sys
            M.setLogHandler(sys.stdout)

        # ---- Variables -------------------------------------------------------
        # X: symmetric n×n PSD matrix
        X = M.variable("X", Domain.inPSDCone(n))
        # λ: scalar objective / edge dot-product bound
        lam = M.variable("lam", 1, Domain.unbounded())

        # ---- Objective -------------------------------------------------------
        M.objective(ObjectiveSense.Minimize, lam)

        # ---- Constraints -----------------------------------------------------
        # (1) Diagonal entries = 1  (unit vectors)
        for i in range(n):
            M.constraint(f"diag_{i}", X.index(i, i), Domain.equalsTo(1.0))

        # (2) X[i,j] ≤ λ  for every edge (i,j)
        for i, j in edges:
            # X[i,j] - λ ≤ 0
            M.constraint(
                f"edge_{i}_{j}",
                Expr.sub(X.index(i, j), lam),
                Domain.lessThan(0.0),
            )

        # ---- Solve -----------------------------------------------------------
        M.solve()

        sol_status = M.getPrimalSolutionStatus()
        if str(sol_status) not in ("SolutionStatus.Optimal",
                                    "SolutionStatus.NearOptimal"):
            raise RuntimeError(
                f"SDP did not reach optimality. Solution status: {sol_status}"
            )

        lam_opt = float(lam.level()[0])
        X_flat = X.level()                          # flat length n*n
        X_gram = np.array(X_flat).reshape(n, n)

        # ---- Extract vectors via eigendecomposition -------------------------
        # X_gram should be PSD; numerical noise may give tiny negative eigenvalues
        # → clamp them to 0.
        eigvals, eigvecs = np.linalg.eigh(X_gram)
        eigvals = np.maximum(eigvals, 0.0)           # numerical safety
        # V = eigvecs @ diag(sqrt(eigvals))  →  V @ V^T = X_gram
        sqrt_D = np.diag(np.sqrt(eigvals))
        V = eigvecs @ sqrt_D                         # shape (n, n)
        # Row i of V is the vector for vertex i; normalise to unit length
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        vectors = V / norms

    return vectors, lam_opt, X_gram, nodes


def vector_chromatic_number(lam_opt: float) -> float:
    """
    Compute the vector chromatic number from the SDP optimal value.

    χ_v(G) = 1 - 1/λ*

    For a 3-colorable graph λ* ≈ -0.5, giving χ_v ≈ 3.
    """
    return 1.0 - 1.0 / lam_opt
