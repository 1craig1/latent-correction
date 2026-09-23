"""
SAE... no -- CL map-type test: is ORTHOGONAL Procrustes SPECIFICALLY better than an unconstrained/ridge LINEAR
drift map (LDC/DPCR territory), or is 'orthogonal' not special?  Decides the CL paper's remaining method-edge.

Reuses the cached Split-CIFAR SEQUENCE features (sae... EXP-CL-SEQUENCE/seq_seed*.npz). Same anchors, same info;
only the map family changes:
  forgotten (identity) | ORTHO (Procrustes) | LINEAR (unconstrained lstsq) | RIDGE (l2) | oracle
Also sweeps ANCHOR COUNT -- orthogonal's regularization edge (fewer params) should grow as anchors get scarce.
Instant (no retrain, no CIFAR load). Run: /home/hgao0864/miniconda3/envs/segearth_de/bin/python scripts/geom_cl_maptype.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config as C
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression

RES = os.path.join(C.ROOT, "experiments/activation_geometry/EXP-CL-SEQUENCE/results")
NTASK = 5; ANCHORS = [100, 300, 1000, 2000]


def ortho(S, T): R, _ = orthogonal_procrustes(S, T); return R                  # S@R ~ T
def linmap(S, T): return np.linalg.lstsq(S, T, rcond=None)[0]                   # unconstrained
def ridge(S, T, a): return np.linalg.solve(S.T @ S + a * np.eye(S.shape[1]), S.T @ T)


def main():
    seeds = sorted(int(f.split("seed")[1].split(".")[0]) for f in os.listdir(RES) if f.startswith("seq_seed"))
    D = {s: {k: v for k, v in np.load(f"{RES}/seq_seed{s}.npz").items()} for s in seeds}
    print(f"[maptype] {len(seeds)} seeds, tasks {NTASK}, chance 0.10")

    for nA in ANCHORS:
        acc = {m: [] for m in ["forgot", "ortho", "linear", "ridge", "oracle"]}
        for s in seeds:
            d = D[s]
            for t in range(NTASK - 1):                                          # old tasks only
                ytr, yte = d[f"y{t}tr"], d[f"y{t}te"]
                ro = LogisticRegression(max_iter=2000).fit(d[f"A{t}"], ytr)
                Gf, Gt = d["Gf"][:nA], d[f"G{t}"][:nA]                          # final vs time-t generic anchors
                Ate = d[f"Ate{t}"]
                acc["forgot"].append(ro.score(Ate, yte))
                acc["ortho"].append(ro.score(Ate @ ortho(Gf, Gt), yte))
                acc["linear"].append(ro.score(Ate @ linmap(Gf, Gt), yte))
                acc["ridge"].append(ro.score(Ate @ ridge(Gf, Gt, 1.0), yte))
                acc["oracle"].append(LogisticRegression(max_iter=2000).fit(d[f"Af{t}"], ytr).score(Ate, yte))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        print(f"  anchors={nA:>5} | forgot {m['forgot']:.3f} | ORTHO {m['ortho']:.3f} | LINEAR {m['linear']:.3f} "
              f"| RIDGE {m['ridge']:.3f} | oracle {m['oracle']:.3f}   (ortho-linear {m['ortho']-m['linear']:+.3f})")
    print("\nverdict: if ORTHO > LINEAR (esp. at small anchors) -> orthogonality regularizes = real edge; "
          "if LINEAR >= ORTHO -> 'orthogonal' not special, pivot to the boundary/detector SCIENCE framing.")


if __name__ == "__main__":
    main()
