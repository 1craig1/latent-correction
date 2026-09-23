"""
EXP-CL-SEQUENCE -- multi-task class-incremental Split-CIFAR-100: learn tasks T0->T1->...->T4 (5 tasks x 10
classes), then recover EVERY earlier task's readout on the FINAL drifted model via one rotation each.

Generic rotation anchors = a held-out pool from classes 50-99 (NEVER trained on = truly task-agnostic,
rehearsal-free). Task-relevant anchors = that task's own (unlabeled) images.
For each old task t: ref (fresh) / forgotten (final model) / oracle (retrain, needs labels) / rot_gen / rot_taskrel.
Shows recovery as a function of task-age (T0 = oldest = most forgotten). Seeds x resumable.
Run in VS Code: /home/hgao0864/miniconda3/envs/segearth_de/bin/python scripts/geom_cl_sequence.py
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config as C
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from scipy.linalg import orthogonal_procrustes

OUT = os.path.join(C.ROOT, "experiments/activation_geometry/EXP-CL-SEQUENCE/results")
DEV = "cuda"; NPT = 10; NTASK = 5; EP, BS, LR, NFEAT = 6, 128, 1e-3, 256
SEEDS = [0, 1, 2]
MEAN = np.array([0.5071, 0.4865, 0.4409]); STD = np.array([0.2673, 0.2564, 0.2762])


class Net(nn.Module):
    def __init__(self, nf=NFEAT, nc=NPT):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.b = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, nf), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(nf, nc)
    def feat(self, x): return self.b(x)
    def forward(self, x): return self.head(self.b(x))


def pack(ds, cls):
    y = np.array(ds["fine_label"]); keep = np.isin(y, cls)
    X = np.stack([np.array(im) for im, k in zip(ds["img"], keep) if k]).astype(np.float32) / 255.0
    X = ((X - MEAN) / STD).transpose(0, 3, 1, 2); yy = y[keep]; remap = {c: i for i, c in enumerate(cls)}
    return torch.tensor(X, dtype=torch.float32), np.array([remap[v] for v in yy])


def train(net, X, y, seed):
    net.to(DEV).train(); opt = torch.optim.Adam(net.parameters(), lr=LR); lf = nn.CrossEntropyLoss()
    yt = torch.tensor(y, dtype=torch.long); g = torch.Generator().manual_seed(seed)
    for e in range(EP):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), BS):
            bi = perm[s:s+BS]; opt.zero_grad(); lf(net(X[bi].to(DEV)), yt[bi].to(DEV)).backward(); opt.step()
    net.eval()


@torch.no_grad()
def feats(net, X):
    o = [net.feat(X[s:s+256].to(DEV)).cpu().numpy() for s in range(0, len(X), 256)]
    F = np.concatenate(o).astype(np.float64); F = F - F.mean(0, keepdims=True)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


# ---- NCM + SDC baseline (all rehearsal-free, same framework for a fair rotation-vs-SDC comparison) ----
def protos(feat, y):
    cs = np.unique(y); return cs, np.stack([feat[y == c].mean(0) for c in cs])
def ncm_acc(test, cs, P, ytest):
    d = ((test[:, None, :] - P[None, :, :]) ** 2).sum(-1); return float(np.mean(cs[d.argmin(1)] == ytest))
def sdc_protos(cs, P0, Gt, Gf):
    """Semantic Drift Compensation (Yu et al. CVPR2020): estimate each old prototype's drift from anchors that
    have BOTH old(Mt) and new(Mfinal) embeddings, weighted by Mt-space proximity; move P0 into Mfinal space."""
    Delta = Gf - Gt
    sub = Gt[::max(1, len(Gt)//400)]
    sigma = np.median(np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)) + 1e-9)
    Pc = np.empty_like(P0)
    for i in range(len(P0)):
        w = np.exp(-((Gt - P0[i]) ** 2).sum(1) / (2 * sigma ** 2 + 1e-9)); w = w / (w.sum() + 1e-9)
        Pc[i] = P0[i] + w @ Delta
    return Pc


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = load_dataset("uoft-cs/cifar100", split="train"); te = load_dataset("uoft-cs/cifar100", split="test")
    tasks = [(pack(tr, list(range(10*t, 10*t+10))), pack(te, list(range(10*t, 10*t+10)))) for t in range(NTASK)]
    Xgen, _ = pack(tr, list(range(50, 100))); Xgen = Xgen[:2000]
    print(f"[data] {NTASK} tasks x {NPT} cls; generic anchors {len(Xgen)} (cls 50-99, untrained)", flush=True)

    allrows = {}
    for seed in SEEDS:
        cp = f"{OUT}/seq_seed{seed}.npz"
        if os.path.exists(cp):
            z = np.load(cp); D = {k: z[k] for k in z.files}; print(f"  [seed {seed}] loaded", flush=True)
        else:
            torch.manual_seed(seed); np.random.seed(seed); net = Net()
            D = {}
            for t in range(NTASK):
                (Xtr, ytr), (Xte, yte) = tasks[t]
                net.head = nn.Linear(NFEAT, NPT); train(net, Xtr, ytr, seed*10 + t)   # learn task t (backbone drifts)
                D[f"A{t}"] = feats(net, Xtr); D[f"B{t}"] = feats(net, Xte); D[f"G{t}"] = feats(net, Xgen)  # snapshot @ time t
                D[f"y{t}tr"] = ytr; D[f"y{t}te"] = yte
                print(f"    [seed {seed}] learned task {t}", flush=True)
            for t in range(NTASK):                                                     # final model on every task
                (Xtr, _), (Xte, _) = tasks[t]
                D[f"Af{t}"] = feats(net, Xtr); D[f"Ate{t}"] = feats(net, Xte)
            D["Gf"] = feats(net, Xgen)
            del net; torch.cuda.empty_cache(); np.savez(cp, **{k: np.asarray(v) for k, v in D.items()})
            print(f"  [seed {seed}] saved", flush=True)

        rows = {}
        for t in range(NTASK):
            ytr, yte = D[f"y{t}tr"], D[f"y{t}te"]
            ro = LogisticRegression(max_iter=2000).fit(D[f"A{t}"], ytr)                 # readout when task t was fresh
            Rg, _ = orthogonal_procrustes(D["Gf"], D[f"G{t}"])                          # final->t via generic anchors
            Ra, _ = orthogonal_procrustes(D[f"Af{t}"], D[f"A{t}"])                      # final->t via task-t anchors
            # NCM framework (fair rotation-vs-SDC): all rehearsal-free, differ only in drift handling
            cs, P0 = protos(D[f"A{t}"], ytr)                                            # M0/Mt prototypes
            _, P1 = protos(D[f"Af{t}"], ytr)                                            # Mfinal prototypes (oracle, needs labels)
            Psdc = sdc_protos(cs, P0, D[f"G{t}"], D["Gf"])                              # SDC: drift-compensated into Mfinal
            rows[t] = {
                "ref": float(ro.score(D[f"B{t}"], yte)),
                "forgotten": float(ro.score(D[f"Ate{t}"], yte)),
                "oracle": float(LogisticRegression(max_iter=2000).fit(D[f"Af{t}"], ytr).score(D[f"Ate{t}"], yte)),
                "rot_gen": float(ro.score(D[f"Ate{t}"] @ Rg, yte)),
                "rot_taskrel": float(ro.score(D[f"Ate{t}"] @ Ra, yte)),
                # --- NCM comparison block ---
                "ncm_forgot": ncm_acc(D[f"Ate{t}"], cs, P0, yte),                       # no compensation
                "ncm_sdc": ncm_acc(D[f"Ate{t}"], cs, Psdc, yte),                        # SDC baseline
                "ncm_rot": ncm_acc(D[f"Ate{t}"] @ Rg, cs, P0, yte),                     # OURS (rotation), rehearsal-free
                "ncm_oracle": ncm_acc(D[f"Ate{t}"], cs, P1, yte),                       # oracle
            }
        allrows[seed] = rows

    print(f"\n===== CL SEQUENCE ({NTASK} tasks x {NPT} cls, chance {1/NPT:.2f}, n={len(allrows)} seeds) =====")
    print(f"  {'task-age':9s}{'ref':>7}{'forgot':>8}{'rot_gen':>9}{'rot_rel':>9}{'oracle':>8}")
    agg = {}
    for t in range(NTASK):
        m = {k: np.mean([allrows[s][t][k] for s in allrows]) for k in ["ref", "forgotten", "oracle", "rot_gen", "rot_taskrel"]}
        agg[t] = m; tag = f"T{t}" + ("(old)" if t < NTASK-1 else "(cur)")
        print(f"  {tag:9s}{m['ref']:>7.3f}{m['forgotten']:>8.3f}{m['rot_gen']:>9.3f}{m['rot_taskrel']:>9.3f}{m['oracle']:>8.3f}")
    old = list(range(NTASK-1))
    def mo(k): return float(np.mean([agg[t][k] for t in old]))
    print(f"\n  OLD tasks (T0..T{NTASK-2}) mean: forgot {mo('forgotten'):.3f} -> rot_gen {mo('rot_gen'):.3f} "
          f"(+{mo('rot_gen')-mo('forgotten'):.3f}) / rot_rel {mo('rot_taskrel'):.3f} (+{mo('rot_taskrel')-mo('forgotten'):.3f}) | oracle {mo('oracle'):.3f}")
    print(f"  recovery of forgot->oracle gap: rot_gen {(mo('rot_gen')-mo('forgotten'))/(mo('oracle')-mo('forgotten')+1e-9):.0%}, "
          f"rot_rel {(mo('rot_taskrel')-mo('forgotten'))/(mo('oracle')-mo('forgotten')+1e-9):.0%}")

    print(f"\n===== NCM framework: OURS (rotation) vs SDC (Yu 2020) — both rehearsal-free =====")
    print(f"  {'task-age':9s}{'ncm_forgot':>11}{'SDC':>8}{'OURS(rot)':>10}{'ncm_oracle':>11}")
    for t in range(NTASK):
        m = {k: np.mean([allrows[s][t][k] for s in allrows]) for k in ["ncm_forgot", "ncm_sdc", "ncm_rot", "ncm_oracle"]}
        agg[t].update(m); tag = f"T{t}" + ("(old)" if t < NTASK-1 else "(cur)")
        print(f"  {tag:9s}{m['ncm_forgot']:>11.3f}{m['ncm_sdc']:>8.3f}{m['ncm_rot']:>10.3f}{m['ncm_oracle']:>11.3f}")
    print(f"\n  OLD-task mean:  no-comp {mo('ncm_forgot'):.3f} | SDC {mo('ncm_sdc'):.3f} | OURS(rot) {mo('ncm_rot'):.3f} | oracle {mo('ncm_oracle'):.3f}")
    print(f"  >>> OURS vs SDC = {mo('ncm_rot')-mo('ncm_sdc'):+.3f}  ({'OURS wins' if mo('ncm_rot')>mo('ncm_sdc')+0.005 else 'SDC wins' if mo('ncm_sdc')>mo('ncm_rot')+0.005 else 'tie'})")
    json.dump({str(s): allrows[s] for s in allrows}, open(f"{OUT}/cl_sequence.json", "w"), indent=1)
    print(f"saved -> {OUT}/cl_sequence.json")


if __name__ == "__main__":
    main()
