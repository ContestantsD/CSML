import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

_CT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CT)
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from meshmae_regressor import Mesh_regressor
from dataset_paired_offline import PairedOfflineFeatureDataset
from dataset_offline import compute_nan_stats

from captum.attr import IntegratedGradients


LH_DIR = None  # set from --lh-dir in main()
RH_DIR = None  # set from --rh-dir in main()
COHORT_DIR = None  # set from --cohort-dir in main()
PHENOTYPE = None  # set from --phenotype in main()

SEEDS = [1, 11, 16]
N_STEPS = 64
TOP_FRAC = 0.10


def build_bilateral_model(ckpt_path, device):
    model_kw = dict(channels=10, num_heads=6, encoder_depth=6, embed_dim=384,
                    patch_size=64, drop_path=0.0, path_mode="dual")
    net_l = Mesh_regressor(**model_kw)
    net_r = Mesh_regressor(**model_kw)
    net_l.head = nn.Identity()
    net_r.head = nn.Identity()

    fusion_mlp = nn.Sequential(
        nn.Linear(384 * 2, 128), nn.ReLU(), nn.Linear(128, 1)
    )

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_l.load_state_dict(sd["net0_state_dict"])
    net_r.load_state_dict(sd["net1_state_dict"])
    fusion_mlp.load_state_dict(sd["fusion_mlp_state_dict"])

    net_l.to(device).eval()
    net_r.to(device).eval()
    fusion_mlp.to(device).eval()
    return net_l, net_r, fusion_mlp


class BilateralIGWrapper(nn.Module):

    def __init__(self, net_l, net_r, fusion_mlp):
        super().__init__()
        self.net_l = net_l
        self.net_r = net_r
        self.fusion_mlp = fusion_mlp

    def forward(self, feats_l, feats_r):
        ctx = self._ctx
        z_l = self.net_l(ctx["faces_l"], feats_l, ctx["centers_l"],
                         ctx["coords_l"], ctx["hop_l"])
        z_r = self.net_r(ctx["faces_r"], feats_r, ctx["centers_r"],
                         ctx["coords_r"], ctx["hop_r"])
        return self.fusion_mlp(torch.cat([z_l, z_r], dim=1))


def load_splits(seed):
    df = pd.read_csv(f"{COHORT_DIR}/splits.csv", dtype={"subject_id": str})
    df["subject_id"] = df["subject_id"].str.zfill(6)
    out = {}
    for part in ("train", "val", "test"):
        out[part] = df[(df["seed"] == seed) & (df["partition"] == part)]["subject_id"].tolist()
    return out


def compute_nan_for_dir(feat_dir, train_subjs):
    sids_arr = np.load(f"{feat_dir}/subject_ids.npy", allow_pickle=True)
    sid2idx = {}
    for i, s in enumerate(sids_arr):
        sid2idx.setdefault(str(int(s)), i)
    train_idx = [sid2idx[s] for s in train_subjs if s in sid2idx]
    data_mmap = np.load(f"{feat_dir}/data.npy", mmap_mode="r")
    cmap = np.load(f"{feat_dir}/canonical_mapping.npy")
    mu, sigma = compute_nan_stats(data_mmap, cmap, train_idx)
    return {"mu": mu, "sigma": sigma}


def build_test_dataset(seed):
    splits = load_splits(seed)
    nan_l = compute_nan_for_dir(LH_DIR, splits["train"])
    nan_r = compute_nan_for_dir(RH_DIR, splits["train"])

    df = pd.read_csv(f"{COHORT_DIR}/cohort.csv", dtype={"Subject": str})
    df["Subject"] = df["Subject"].str.zfill(6)
    z = df[PHENOTYPE].astype(float)
    mu_p, sigma_p = float(z.mean()), float(z.std())
    if sigma_p < 1e-8:
        sigma_p = 1.0

    def make_z(subjs):
        sub_df = df[df["Subject"].isin(subjs)]
        return {r["Subject"]: (float(r[PHENOTYPE]) - mu_p) / sigma_p
                for _, r in sub_df.iterrows()}

    z_map = {**make_z(splits["train"]), **make_z(splits["val"]), **make_z(splits["test"])}

    ds = PairedOfflineFeatureDataset(
        LH_DIR, RH_DIR, splits["test"], PHENOTYPE,
        ram_cache=True, nan=True, dist="geodesic",
        nan_stats_l=nan_l, nan_stats_r=nan_r,
    )
    ds.z_map = z_map
    return ds


def compute_ig_for_subject(wrapper, ig_module, batch, device):
    (lf, lc, lcd, lf2, lFs, lhop, rf, rc, rcd, rf2, rFs, rhop, label, sid) = batch

    wrapper._ctx = {
        "faces_l": lf2.float().unsqueeze(0).to(device),
        "centers_l": lc.float().unsqueeze(0).to(device),
        "coords_l": lcd.float().unsqueeze(0).to(device),
        "hop_l": lhop.unsqueeze(0).to(device),
        "faces_r": rf2.float().unsqueeze(0).to(device),
        "centers_r": rc.float().unsqueeze(0).to(device),
        "coords_r": rcd.float().unsqueeze(0).to(device),
        "hop_r": rhop.unsqueeze(0).to(device),
    }

    feats_l = lf.float().unsqueeze(0).to(device).requires_grad_(True)
    feats_r = rf.float().unsqueeze(0).to(device).requires_grad_(True)
    baseline_l = torch.zeros_like(feats_l)
    baseline_r = torch.zeros_like(feats_r)

    attr_l, attr_r = ig_module.attribute(
        inputs=(feats_l, feats_r),
        baselines=(baseline_l, baseline_r),
        target=0,
        n_steps=N_STEPS,
    )

    imp_l = attr_l.squeeze(0).abs().sum(dim=(0, 2)).detach().cpu().numpy()
    imp_r = attr_r.squeeze(0).abs().sum(dim=(0, 2)).detach().cpu().numpy()
    return imp_l, imp_r


def run_seed(seed, ckpt_path, device, out_dir):
    print(f"\n=== seed {seed} ===")
    net_l, net_r, fusion_mlp = build_bilateral_model(ckpt_path, device)
    wrapper = BilateralIGWrapper(net_l, net_r, fusion_mlp).to(device).eval()
    ig_module = IntegratedGradients(wrapper)

    ds = build_test_dataset(seed)
    n_test = len(ds)
    print(f"[data] {n_test} test subjects")

    all_imp_l = []
    all_imp_r = []
    for k in range(n_test):
        batch = ds[k]
        imp_l, imp_r = compute_ig_for_subject(wrapper, ig_module, batch, device)
        total = imp_l.sum() + imp_r.sum()
        if total > 0:
            imp_l /= total
            imp_r /= total
        all_imp_l.append(imp_l)
        all_imp_r.append(imp_r)
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{n_test}]", flush=True)

    ig_l = np.mean(all_imp_l, axis=0)
    ig_r = np.mean(all_imp_r, axis=0)
    np.save(out_dir / f"ig_seed{seed}_LH.npy", ig_l.astype(np.float32))
    np.save(out_dir / f"ig_seed{seed}_RH.npy", ig_r.astype(np.float32))
    print(f"  saved ig_seed{seed}_{{LH,RH}}.npy  range L=[{ig_l.min():.4g},{ig_l.max():.4g}] "
          f"R=[{ig_r.min():.4g},{ig_r.max():.4g}]")
    return ig_l, ig_r


def stability_analysis(seed_results, out_dir):
    rows = []
    n_patches = len(seed_results[SEEDS[0]][0])
    top_k = max(1, int(n_patches * TOP_FRAC))

    for hemi in ("LH", "RH"):
        hi = 0 if hemi == "LH" else 1
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                sa, sb = SEEDS[i], SEEDS[j]
                va = seed_results[sa][hi]
                vb = seed_results[sb][hi]
                rho, p = spearmanr(va, vb)
                top_a = set(np.argsort(va)[-top_k:])
                top_b = set(np.argsort(vb)[-top_k:])
                jac = len(top_a & top_b) / len(top_a | top_b)
                rows.append({
                    "hemi": hemi, "seed_a": sa, "seed_b": sb,
                    "spearman": round(float(rho), 4), "p": float(p),
                    "top10pct_jaccard": round(float(jac), 4),
                    "common_top10pct": len(top_a & top_b),
                })

        tops = [set(np.argsort(seed_results[s][hi])[-top_k:]) for s in SEEDS]
        common = tops[0] & tops[1] & tops[2]
        rows.append({
            "hemi": hemi, "seed_a": "ALL3", "seed_b": "ALL3",
            "spearman": "", "p": "", "top10pct_jaccard": "",
            "common_top10pct": float(len(common)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ig_seed_stability.csv", index=False)
    print("\n=== IG cross-seed stability ===")
    print(df.to_string(index=False))
    return df


def save_top_patches(mean_l, mean_r, out_dir):
    n = len(mean_l)
    top_k = max(1, int(n * TOP_FRAC))
    rows = []
    for hemi, mean in [("LH", mean_l), ("RH", mean_r)]:
        idx = np.argsort(mean)[::-1]
        for rank, p in enumerate(idx[:top_k]):
            rows.append({"hemi": hemi, "patch": int(p), "rank": rank + 1,
                         "importance": float(mean[p])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ig_top_patches.csv", index=False)
    print(f"saved ig_top_patches.csv ({len(df)} patches)")


def main():
    ap = argparse.ArgumentParser(description="Integrated Gradients for CSML phenotype model")
    ap.add_argument("--lh-dir", required=True,
                    help="left-hemisphere offline feature directory")
    ap.add_argument("--rh-dir", required=True,
                    help="right-hemisphere offline feature directory")
    ap.add_argument("--cohort-dir", required=True,
                    help="cohort directory (cohort.csv / splits.csv)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--phenotype", required=True,
                    help="phenotype column the checkpoints were trained on")
    ap.add_argument("--ckpt-template", required=True,
                    help="checkpoint path template containing {seed}")
    args = ap.parse_args()
    global LH_DIR, RH_DIR, COHORT_DIR, PHENOTYPE
    LH_DIR, RH_DIR, COHORT_DIR = args.lh_dir, args.rh_dir, args.cohort_dir
    PHENOTYPE = args.phenotype

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    seed_results = {}
    for seed in SEEDS:
        if args.ckpt_template:
            ckpt = args.ckpt_template.format(seed=seed)
        if not os.path.exists(ckpt):
            print(f"[SKIP seed {seed}] checkpoint not found: {ckpt}")
            continue
        print(f"[ckpt] {ckpt}")
        ig_l, ig_r = run_seed(seed, ckpt, device, out_dir)
        seed_results[seed] = (ig_l, ig_r)

    if len(seed_results) < 2:
        print("[WARN] fewer than 2 seeds completed, skipping stability analysis")
        return

    mean_l = np.mean([v[0] for v in seed_results.values()], axis=0)
    mean_r = np.mean([v[1] for v in seed_results.values()], axis=0)
    np.save(out_dir / "ig_mean_LH.npy", mean_l.astype(np.float32))
    np.save(out_dir / "ig_mean_RH.npy", mean_r.astype(np.float32))
    print(f"\nsaved ig_mean_{{LH,RH}}.npy  (averaged over {len(seed_results)} seeds)")

    stability_analysis(seed_results, out_dir)

    save_top_patches(mean_l, mean_r, out_dir)

    with open(out_dir / "explainability_summary.md", "w") as f:
        f.write(f"# Explainability summary (Integrated Gradients, {args.phenotype})\n\n")
        f.write(f"- Model: bilateral (geodesic, NAN), "
                f"seeds {SEEDS}, 81 test subjects per seed\n")
        f.write(f"- Method: Integrated Gradients, {N_STEPS} steps, all-zero baseline, "
                f"LH/RH attributed jointly in a single forward pass\n")
        f.write(f"- Patch importance: sum of |IG| over 10 channels x 64 positions, "
                f"normalized per subject (LH+RH=1), then averaged\n")
        f.write(f"- Top {int(TOP_FRAC*100)}% = {int(len(mean_l)*TOP_FRAC)}/{len(mean_l)} patches\n\n")
        f.write("## Stability\n\n")
        stab = pd.read_csv(out_dir / "ig_seed_stability.csv")
        f.write(stab.to_markdown(index=False))
        f.write("\n\n> Interpretation: model prediction sensitivity to surface region inputs; "
                "not a causal claim.\n")
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
