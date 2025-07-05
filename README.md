# Method of Averaged Marginals (MAM)

**Exact Wasserstein Barycenter via Douglas–Rachford / Averaged Marginals**  
*Mimouni, Malisani, Zhu & de Oliveira, SIAM J. Math. Data Sci. 2024* :contentReference[oaicite:3]{index=3}

This repository contains a clean, modular implementation of the MAM algorithm for computing **exact** (unregularized) Wasserstein barycenters of discrete measures—capable of handling both **balanced** and **unbalanced** cases via operator splitting. The algorithm is based on the Douglas–Rachford framework and efficiently performs exact projections on marginals, suitable for parallel processing and large-scale data :contentReference[oaicite:4]{index=4}.

---

## 🚀 Highlights

- **Exact solution** of the linear Wasserstein barycenter problem (no entropic smoothing)
- Supports **balanced** (equal mass) and **unbalanced** (mass creation/destruction) variants
- Based on operator splitting (Douglas–Rachford) – interpretable as averaging marginals
- Fully **parallelizable**, scalable to large discrete datasets :contentReference[oaicite:5]{index=5}
- Demonstrated **state-of-the-art** performance in convergence and precision :contentReference[oaicite:6]{index=6}

---

## 📦 Installation

```bash
git clone https://github.com/dan‑mim/MAM.git
cd MAM
pip install -r requirements.txt

---

## Cite this article
@article{mimouni2024computing,
  title={Computing Wasserstein Barycenter via Operator Splitting: The Method of Averaged Marginals},
  author={Mimouni, Daniel and Malisani, Paul and Zhu, Jiamin and de Oliveira, Welington},
  journal={SIAM Journal on Mathematics of Data Science},
  volume={6},
  number={4},
  pages={1000–1026},
  year={2024},
  doi={10.1137/23M1584228}
}
