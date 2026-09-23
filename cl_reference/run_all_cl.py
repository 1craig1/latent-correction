"""
run_all_cl.py — overnight batch for the CL drift-compensation paper. Fully self-contained; launch
detached (setsid nohup). Saves results INCREMENTALLY per (dataset, seed) so partial runs survive.

Three deliverables:
  1. Aligned LDC budget (task0=200ep, inc=100ep) on CIFAR-100 -> leaderboard-comparable absolute A_last.
  2. Multi-seed x multi-dataset: CIFAR-100, Tiny-ImageNet, ImageNet-100.
  3. ANCHOR-BUDGET sweep (rotation vs affine at few anchors) = the "why rotation not affine" argument
     (orthogonal Procrustes is well-posed with few anchors; unconstrained lstsq overfits).

Compensation methods on one shared LwF trajectory: none / SDC / ours(rot) / ours(rot+scale) /
affine(lstsq=LDC-closed-form) / oracle. Metric A_last, A_inc. Ordered by importance so early death
still delivers CIFAR. Results -> EXP-CL-FULL/results/results.json ; log -> EXP-CL-FULL/results/run.log
"""
import os, sys, json, time, traceback, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "scripts"); import config as C
import geom_common as G
from scipy.linalg import orthogonal_procrustes
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(G.exp_dir("EXP-CL-FULL"), "results"); os.makedirs(OUT, exist_ok=True)
RESJSON = os.path.join(OUT, "results.json"); FIGP = os.path.join(C.ROOT, "figures_paper")
BS, LAMBDA = 128, 1.0; ANCHORS = [16, 64, 256, "all"]

# dataset configs: (hf_name, img_key, label_key, n_classes, per_task, img_size, t0_ep, inc_ep, seeds, cap_tr, cap_te, anchor_sweep)
DATASETS = [
    ("cifar100", "uoft-cs/cifar100", "img", "fine_label", 100, 10, 32, 200, 100, [0, 1, 2], 500, 100, True),
    ("tinyimagenet", "zh-plus/tiny-imagenet", "image", "label", 200, 20, 64, 100, 60, [0, 1], 500, 50, True),
    ("imagenet100", "clane9/imagenet-100", "image", "label", 100, 10, 64, 100, 60, [0, 1], 500, 100, False),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_np(hf, imk, lbk, size, cap_tr, cap_te):
    from datasets import load_dataset
    import numpy as np
    def grab(split, cap):
        ds = load_dataset(hf, split=split)
        lbls = np.array(ds[lbk]); Xs, ys = [], []
        percls = {}
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        idxs = np.random.default_rng(0).permutation(len(ds))
        for i in idxs:
            c = int(lbls[i])
            if percls.get(c, 0) >= cap: continue
            im = ds[int(i)][imk]
            if im.mode != "RGB": im = im.convert("RGB")
            if im.size != (size, size): im = im.resize((size, size))
            a = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
            Xs.append(a.transpose(2, 0, 1).astype(np.float32)); ys.append(c); percls[c] = percls.get(c, 0) + 1
        return np.stack(Xs), np.array(ys)
    # tiny-imagenet test split is 'valid'
    tr_split = "train"
    te_split = "valid" if "tiny" in hf else ("validation" if "imagenet-100" in hf else "test")
    try:
        Xtr, ytr = grab(tr_split, cap_tr); Xte, yte = grab(te_split, cap_te)
    except Exception:
        Xtr, ytr = grab("train", cap_tr); Xte, yte = grab("test", cap_te)
    return Xtr, ytr, Xte, yte


def make_net(n_classes, size):
    from torchvision.models import resnet18
    m = resnet18(num_classes=n_classes)
    if size <= 64:
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
    return m


def _bb(m, x):
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    return torch.flatten(m.avgpool(x), 1)


def feats(m, X, bs=512):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(_bb(m, torch.tensor(X[i:i+bs]).to(DEV)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train_task(model, old_model, Xt, yt, seen, new_cls, epochs, n_classes):
    model.train(); opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = torch.tensor(Xt); y = torch.tensor(yt, dtype=torch.long)
    old_cls = [c for c in seen if c not in new_cls]
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BS):
            idx = perm[i:i+BS]; xb = X[idx].to(DEV); yb = y[idx].to(DEV)
            lg = model.fc(_bb(model, xb)); loss = F.cross_entropy(lg, yb)
            if old_model is not None and old_cls:
                with torch.no_grad(): lo = old_model.fc(_bb(old_model, xb))[:, old_cls]
                loss = loss + LAMBDA * F.kl_div(F.log_softmax(lg[:, old_cls]/2, 1), F.softmax(lo/2, 1), reduction="batchmean")*4
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return model


def m_rot(P, A, B, idx=None):
    if idx is not None: A, B = A[idx], B[idx]
    ma, mb = A.mean(0), B.mean(0); R, _ = orthogonal_procrustes(A-ma, B-mb); return (P-ma) @ R + mb
def m_rotscale(P, A, B, idx=None):
    if idx is not None: A, B = A[idx], B[idx]
    ma, mb = A.mean(0), B.mean(0); R, sc = orthogonal_procrustes(A-ma, B-mb); s = sc/np.sum((A-ma)**2); return (P-ma) @ R*s + mb
def m_affine(P, A, B, idx=None):
    if idx is not None: A, B = A[idx], B[idx]
    Ac = np.hstack([A, np.ones((len(A), 1))]); W, *_ = np.linalg.lstsq(Ac, B, rcond=1e-3)
    return np.hstack([P, np.ones((len(P), 1))]) @ W
def m_sdc(P, A, B, idx=None):
    if idx is not None: A, B = A[idx], B[idx]
    delta = B - A; out = []
    for p in P:
        d2 = np.sum((A-p)**2, 1); w = np.exp(-d2/(2*np.median(d2)+1e-9)); w /= w.sum()+1e-12
        out.append(p + w @ delta)
    return np.array(out)
CORE = {"none": lambda P, A, B, idx=None: P, "SDC": m_sdc, "ours(rot)": m_rot,
        "ours(rot+scale)": m_rotscale, "affine": m_affine}


def run_one(cfg, seed):
    name, hf, imk, lbk, ncls, per, size, t0e, ince, _, cap_tr, cap_te, asweep = cfg
    ntask = ncls // per
    log(f"{name} seed{seed}: loading data ...")
    Xtr, ytr, Xte, yte = load_np(hf, imk, lbk, size, cap_tr, cap_te)
    rng = np.random.default_rng(seed); order = rng.permutation(ncls)
    tasks = [order[i*per:(i+1)*per] for i in range(ntask)]
    model = make_net(ncls, size).to(DEV); old = None; seen = []
    methods = dict(CORE)
    if asweep:
        for K in ANCHORS:
            methods[f"rot@{K}"] = (lambda K: lambda P, A, B: m_rot(P, A, B, _sub(A, K, seed)))(K)
            methods[f"affine@{K}"] = (lambda K: lambda P, A, B: m_affine(P, A, B, _sub(A, K, seed)))(K)
    protos = {m: {} for m in methods}; protos["oracle"] = {}
    accs = {m: [] for m in list(methods)+["oracle"]}
    for t, cls in enumerate(tasks):
        mtr = np.isin(ytr, cls); Xt, yt = Xtr[mtr], ytr[mtr]
        A_old = feats(model, Xt) if t > 0 else None
        model = train_task(model, old, Xt, yt, seen+list(cls), list(cls), t0e if t == 0 else ince, ncls)
        B_new = feats(model, Xt); newp = {int(c): B_new[yt == c].mean(0) for c in cls}
        for mname, fn in methods.items():
            if t > 0 and protos[mname]:
                oc = list(protos[mname]); Pm = fn(np.array([protos[mname][c] for c in oc]), A_old, B_new)
                for c, v in zip(oc, Pm): protos[mname][c] = v
            protos[mname].update(newp)
        for c in seen+list(cls):
            protos["oracle"][int(c)] = feats(model, Xtr[np.isin(ytr, [c])]).mean(0)
        seen = seen + list(cls)
        mte = np.isin(yte, seen); Fte = feats(model, Xte[mte]); yy = yte[mte]
        for mname in accs:
            cs = list(protos[mname]); Pmat = np.array([protos[mname][c] for c in cs])
            d = ((Fte[:, None, :]-Pmat[None, :, :])**2).sum(2); accs[mname].append(float((np.array(cs)[d.argmin(1)] == yy).mean()))
        old = make_net(ncls, size).to(DEV); old.load_state_dict(model.state_dict()); old.eval()
        log(f"{name} seed{seed} task{t} ({len(seen)}cls): none={accs['none'][-1]:.3f} SDC={accs['SDC'][-1]:.3f} "
            f"rot={accs['ours(rot)'][-1]:.3f} rot+s={accs['ours(rot+scale)'][-1]:.3f} affine={accs['affine'][-1]:.3f} oracle={accs['oracle'][-1]:.3f}")
    return {"acc_curves": accs, "A_last": {m: accs[m][-1] for m in accs}, "A_inc": {m: float(np.mean(accs[m])) for m in accs},
            "ntask": ntask, "per": per}


def _sub(A, K, seed):
    if K == "all" or K >= len(A): return np.arange(len(A))
    return np.random.default_rng(1000+seed).choice(len(A), K, replace=False)


def save(allres):
    json.dump(allres, open(RESJSON, "w"), indent=1)


def main():
    t0 = time.time()
    global DATASETS
    if os.environ.get("SMOKE"):
        DATASETS = [(n,hf,ik,lk,nc,pr,sz,2,1,[0],40,20,asw) for (n,hf,ik,lk,nc,pr,sz,t0e,ie,sd,ctr,cte,asw) in DATASETS]
    allres = json.load(open(RESJSON)) if os.path.exists(RESJSON) else {}
    log(f"=== run_all_cl START (device={DEV}) ===")
    for cfg in DATASETS:
        name = cfg[0]; allres.setdefault(name, {})
        for seed in cfg[9]:
            key = f"seed{seed}"
            if key in allres[name]:
                log(f"{name} {key} already done, skip"); continue
            try:
                r = run_one(cfg, seed); allres[name][key] = r; save(allres)
                log(f"{name} {key} DONE: A_last rot={r['A_last']['ours(rot)']:.3f} affine={r['A_last']['affine']:.3f} "
                    f"SDC={r['A_last']['SDC']:.3f} ({time.time()-t0:.0f}s elapsed)")
            except Exception as e:
                log(f"!! {name} {key} FAILED: {e}\n{traceback.format_exc()}")
        # plot after each dataset (partial figures survive)
        try:
            plot(allres)
        except Exception as e:
            log(f"plot failed: {e}")
    log(f"=== ALL DONE ({time.time()-t0:.0f}s) ===")


def plot(allres):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order_m = ["none", "SDC", "ours(rot)", "ours(rot+scale)", "affine", "oracle"]
    cols = {"none": "#B0B0B0", "SDC": "#E08A1E", "ours(rot)": "#2E7D32", "ours(rot+scale)": "#1B5E20", "affine": "#7A4FA3", "oracle": "#000000"}
    dsets = [d for d in allres if allres[d]]
    fig, axes = plt.subplots(1, max(len(dsets), 1), figsize=(5.5*max(len(dsets), 1), 5), squeeze=False)
    for j, dn in enumerate(dsets):
        seeds = list(allres[dn]); ax = axes[0][j]
        means = {m: np.mean([allres[dn][s]["A_last"][m] for s in seeds])*100 for m in order_m}
        sds = {m: np.std([allres[dn][s]["A_last"][m] for s in seeds])*100 for m in order_m}
        x = np.arange(len(order_m))
        ax.bar(x, [means[m] for m in order_m], yerr=[sds[m] for m in order_m], color=[cols[m] for m in order_m], capsize=3)
        for i, m in enumerate(order_m): ax.text(i, means[m]+0.3, f"{means[m]:.1f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(order_m, rotation=30, fontsize=7.5); ax.set_ylabel("A_last (%)")
        ax.set_title(f"{dn} ({len(seeds)} seeds)", fontsize=10, fontweight="bold")
    fig.suptitle("EXP-CL-FULL: training-free rotation vs SDC vs affine drift compensation (aligned budget, multi-dataset)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(FIGP, "fig_CL-FULL.png"), dpi=200, bbox_inches="tight")
    # anchor sweep figure (cifar100 if present)
    for dn in dsets:
        s0 = list(allres[dn])[0]
        if any(k.startswith("rot@") for k in allres[dn][s0]["A_last"]):
            seeds = list(allres[dn])
            fig2, ax2 = plt.subplots(figsize=(7.5, 5))
            xs = [k for k in ANCHORS]
            for meth, col in [("rot", "#2E7D32"), ("affine", "#7A4FA3")]:
                ys = [np.mean([allres[dn][s]["A_last"][f"{meth}@{K}"] for s in seeds])*100 for K in ANCHORS]
                es = [np.std([allres[dn][s]["A_last"][f"{meth}@{K}"] for s in seeds])*100 for K in ANCHORS]
                ax2.errorbar([str(k) for k in xs], ys, yerr=es, fmt="-o", color=col, lw=2, capsize=3, label=meth)
            ax2.set_xlabel("# anchors used to fit compensation map"); ax2.set_ylabel("A_last (%)")
            ax2.set_title(f"Anchor efficiency ({dn}): orthogonal rotation vs unconstrained affine\n(rotation robust at few anchors; affine overfits)", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=9, frameon=False)
            fig2.tight_layout(); fig2.savefig(os.path.join(FIGP, f"fig_CL-ANCHOR_{dn}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig2)


if __name__ == "__main__":
    main()
