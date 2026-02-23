# Approximate Graph Coloring via Semidefinite Programming

Implementation of the **Karger, Motwani & Sudan (1998)** algorithm:
> *"Approximate Graph Coloring by Semidefinite Programming"*, JACM 45(2), 1998.

This project relaxes the NP-hard graph coloring problem into a **Semidefinite Program (SDP)**, solves it with [MOSEK](https://www.mosek.com/), and applies two geometric rounding procedures to extract an approximate valid coloring.

---

## Algorithm Overview

### 1. Vector k-Coloring SDP (Sections 3–5)

For a graph $G = (V, E)$ with $n = |V|$, we solve:

$$\min \lambda \quad \text{s.t.} \quad X_{ii} = 1 \; \forall i, \quad X_{ij} \leq \lambda \; \forall (i,j) \in E, \quad X \succeq 0$$

The solution $X^* \in \mathbb{S}^n$ is a Gram matrix: $X^*_{ij} = \mathbf{v}_i \cdot \mathbf{v}_j$, where each $\mathbf{v}_i$ is a unit vector assigned to vertex $i$. The **vector chromatic number** is recovered as $\chi_v(G) = 1 - 1/\lambda^*$.

For a $k$-colorable graph: $\lambda^* = -1/(k-1)$, e.g. $\lambda^* \approx -0.5$ for $k=3$.

### 2. Projection Rounding — `projection_coloring.py` (Section 7)

Iteratively extracts independent sets via random projections:

1. Sample random Gaussian $\mathbf{r} \sim \mathcal{N}(0, I)$.
2. Project each remaining vertex: $x_i = \mathbf{v}_i \cdot \mathbf{r}$.
3. Threshold $c = \sqrt{2 \ln \Delta}$; collect high-projection vertices $S = \{i : x_i \geq c\}$.
4. Find a maximal independent set $I \subseteq S$ and assign it a new color.
5. Remove $I$ and repeat.

**Guarantee (Thm. 7.3):** For 3-colorable $G$, uses $O(\Delta^{1/3} \sqrt{\ln \Delta} \log n)$ colors.

### 3. Hyperplane Rounding — `hyperplane_coloring.py` (Section 6)

Uses $t$ random hyperplanes to partition vertices by sign patterns:

1. Sample $t$ random Gaussian vectors $\mathbf{r}_1, \ldots, \mathbf{r}_t$.
2. Color vertex $i$ by its $t$-bit signature $\sigma(i) = (\text{sgn}(\mathbf{v}_i \cdot \mathbf{r}_1), \ldots, \text{sgn}(\mathbf{v}_i \cdot \mathbf{r}_t))$.
3. Repair intra-class edge conflicts with a greedy fix-up.

**Guarantee (Thm. 6.2):** For 3-colorable $G$, uses $O(\Delta^{\log_3 2} \log n)$ colors.

---

## Repository Structure

```
tp_final_sdp/
├── .venv/                  # Python virtual environment
├── sdp_solver.py           # MOSEK Fusion SDP formulation & vector extraction
├── projection_coloring.py  # Projection rounding algorithm (KMS §7)
├── hyperplane_coloring.py  # Hyperplane rounding algorithm (KMS §6)
├── graph_utils.py          # Graph generation, validation, visualization
├── main.py                 # Demo entry point
├── requirements.txt        # Python dependencies
├── mosek.lic               # MOSEK license file (required)
└── karger_colore.pdf       # Original paper
```

---

## Prerequisites

- **Python ≥ 3.10**
- **MOSEK license** (free academic license at [mosek.com/products/academic-licenses](https://www.mosek.com/products/academic-licenses/))  
  Place `mosek.lic` in the project root (already present) or set `$MOSEKLM_LICENSE_FILE`.

---

## Setup

```bash
# 1. Clone / enter the project directory
cd tp_final_sdp

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full demo

```bash
source .venv/bin/activate
python main.py
```

This runs both algorithms on five test graphs (C₅, C₇, Petersen, K₄, random 3-colorable) and saves coloring plots to `plots/`.

### Use as a library

```python
import networkx as nx
from sdp_solver import solve_vector_coloring, vector_chromatic_number
from projection_coloring import projection_coloring_multi_trial
from hyperplane_coloring import hyperplane_coloring_multi_trial
from graph_utils import is_valid_coloring

# Build or load a graph
G = nx.petersen_graph()

# 1. Solve the vector coloring SDP
vectors, lam_opt, X_gram, nodes = solve_vector_coloring(G)
print(f"λ* = {lam_opt:.4f}  →  χ_v = {vector_chromatic_number(lam_opt):.2f}")

# 2. Projection rounding (Section 7)
coloring = projection_coloring_multi_trial(G, vectors, nodes, num_trials=20)
print(f"Projection: {len(set(coloring.values()))} colors, valid={is_valid_coloring(G, coloring)}")

# 3. Hyperplane rounding (Section 6)
coloring2 = hyperplane_coloring_multi_trial(G, vectors, nodes, num_trials=20)
print(f"Hyperplane: {len(set(coloring2.values()))} colors, valid={is_valid_coloring(G, coloring2)}")
```

---

## Key Dependencies

| Package     | Role                                      |
|-------------|-------------------------------------------|
| `mosek`     | SDP solver (requires license)             |
| `numpy`     | Linear algebra, eigendecomposition        |
| `scipy`     | Cholesky decomposition utilities          |
| `networkx`  | Graph data structures & algorithms        |
| `matplotlib`| Coloring visualization / plots            |

---

## Reference

> D. Karger, R. Motwani, M. Sudan.  
> **Approximate Graph Coloring by Semidefinite Programming.**  
> *Journal of the ACM*, 45(2):246–265, March 1998.  
> DOI: [10.1145/274787.274791](https://doi.org/10.1145/274787.274791)
