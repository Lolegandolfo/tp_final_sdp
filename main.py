"""
main.py
=======
Demo entry point for the Karger-Motwani-Sudan approximate graph coloring
implementation using MOSEK + Python.

Runs both algorithms on several test graphs:
  - Cycle C_5  (3-colorable, odd cycle)
  - Petersen graph (3-colorable)
  - Random 3-colorable graph (n=30)
  - Complete graph K_4 (4-colorable)

For each graph, prints:
  - SDP optimal value λ* and vector chromatic number χ_v
  - Number of colors used by each algorithm vs. greedy baseline
  - Whether the coloring is valid

Run:
  source .venv/bin/activate
  python main.py
"""

import os
import sys
import networkx as nx
import numpy as np

from sdp_solver import solve_vector_coloring, vector_chromatic_number
from projection_coloring import projection_coloring_multi_trial
from hyperplane_coloring import hyperplane_coloring_multi_trial
from graph_utils import (
    is_valid_coloring,
    num_colors_used,
    greedy_coloring_baseline,
    make_k_colorable_graph,
    make_cycle_graph,
    make_petersen_graph,
    make_complete_graph,
    plot_coloring,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70


def run_experiment(
    name: str,
    G: nx.Graph,
    num_trials: int = 10,
    save_plots_dir: str = "plots",
) -> None:
    """
    Run full pipeline on graph G: solve SDP → projection → hyperplane → greedy.
    Prints a formatted report and saves coloring plots.
    """
    print(f"\n{SEPARATOR}")
    print(f"  Graph: {name}")
    print(f"  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    print(SEPARATOR)

    # ---- 1. Solve the SDP ---------------------------------------------------
    print("\n[SDP] Solving vector coloring SDP via MOSEK …")
    try:
        vectors, lam_opt, X_gram, nodes = solve_vector_coloring(G, verbose=False)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    chi_v = vector_chromatic_number(lam_opt)
    print(f"  λ* (optimal)             = {lam_opt:.6f}")
    print(f"  χ_v (vector chromatic #) = {chi_v:.4f}")
    print(f"  (For k-colorable: χ_v ≈ k)")

    # Verify edge constraints are satisfied
    edge_violations = 0
    for u, v in G.edges():
        i, j = nodes.index(u), nodes.index(v)
        dot = float(X_gram[i, j])
        if dot > lam_opt + 1e-4:
            edge_violations += 1
    print(f"  SDP edge constraint violations: {edge_violations}")

    # ---- 2. Projection coloring (Section 7) ---------------------------------
    print(f"\n[PROJECTION] Running {num_trials} trials …")
    proj_colors = projection_coloring_multi_trial(
        G, vectors, nodes,
        num_trials=num_trials,
        seed=42,
        verbose=True,
    )
    proj_k = num_colors_used(proj_colors)
    proj_valid = is_valid_coloring(G, proj_colors)
    print(f"  Colors used : {proj_k}")
    print(f"  Valid       : {'✓ Yes' if proj_valid else '✗ NO — BUG!'}")

    # ---- 3. Hyperplane coloring (Section 6) ---------------------------------
    print(f"\n[HYPERPLANE] Running {num_trials} trials …")
    hyp_colors = hyperplane_coloring_multi_trial(
        G, vectors, nodes,
        num_trials=num_trials,
        k_hint=max(2, round(chi_v)),
        seed=42,
        verbose=True,
    )
    hyp_k = num_colors_used(hyp_colors)
    hyp_valid = is_valid_coloring(G, hyp_colors)
    print(f"  Colors used : {hyp_k}")
    print(f"  Valid       : {'✓ Yes' if hyp_valid else '✗ NO — BUG!'}")

    # ---- 4. Greedy baseline -------------------------------------------------
    greedy_colors = greedy_coloring_baseline(G)
    greedy_k = num_colors_used(greedy_colors)
    greedy_valid = is_valid_coloring(G, greedy_colors)
    print(f"\n[GREEDY]    Colors used: {greedy_k}  Valid: {'✓' if greedy_valid else '✗'}")

    # ---- Summary table ------------------------------------------------------
    print(f"\n{'Algorithm':<22} {'Colors':>7}  {'Valid':>6}")
    print("-" * 38)
    print(f"{'Projection (KMS§7)':<22} {proj_k:>7}  {'✓' if proj_valid else '✗':>6}")
    print(f"{'Hyperplane (KMS§6)':<22} {hyp_k:>7}  {'✓' if hyp_valid else '✗':>6}")
    print(f"{'Greedy baseline':<22} {greedy_k:>7}  {'✓' if greedy_valid else '✗':>6}")

    # ---- Plots --------------------------------------------------------------
    os.makedirs(save_plots_dir, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "-")

    plot_coloring(
        G, proj_colors,
        title=f"{name} — Projection Coloring (KMS §7)",
        save_path=os.path.join(save_plots_dir, f"{safe_name}_projection.png"),
    )
    plot_coloring(
        G, hyp_colors,
        title=f"{name} — Hyperplane Coloring (KMS §6)",
        save_path=os.path.join(save_plots_dir, f"{safe_name}_hyperplane.png"),
    )
    plot_coloring(
        G, greedy_colors,
        title=f"{name} — Greedy Coloring (baseline)",
        save_path=os.path.join(save_plots_dir, f"{safe_name}_greedy.png"),
    )
    print(f"\n  Plots saved to '{save_plots_dir}/'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Approximate Graph Coloring via SDP")
    print("  Karger, Motwani & Sudan (1998)")
    print("  MOSEK Fusion Python implementation")
    print("=" * 70)

    # Check for MOSEK license — try common locations
    lic_path = os.environ.get("MOSEKLM_LICENSE_FILE", "")
    if not lic_path:
        candidates = [
            os.path.join(os.path.dirname(__file__), "mosek.lic"),  # project root
            os.path.expanduser("~/mosek/mosek.lic"),                # MOSEK default
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                os.environ["MOSEKLM_LICENSE_FILE"] = candidate
                print(f"\n[INFO] Using MOSEK license: {candidate}")
                break

    NUM_TRIALS = 15   # number of random projection/hyperplane trials per graph

    # ---- Test graphs --------------------------------------------------------
    experiments = [
        ("Cycle C_5  (3-colorable)", make_cycle_graph(5)),
        ("Cycle C_7  (3-colorable)", make_cycle_graph(7)),
        ("Petersen graph (3-colorable)", make_petersen_graph()),
        ("Complete K_4 (4-colorable)", make_complete_graph(4)),
        ("Random 3-colorable (n=20)", make_k_colorable_graph(3, 20, p_inter=0.4, seed=7)),
    ]

    for name, G in experiments:
        # Remove self-loops if any
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_edges() == 0:
            print(f"\nSkipping '{name}' — no edges.")
            continue
        run_experiment(name, G, num_trials=NUM_TRIALS)

    print(f"\n{SEPARATOR}")
    print("  All experiments complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
