import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import numpy as np
import torch
from sklearn.svm import SVR

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_offline import OfflineFeatureDataset, compute_nan_stats
from meshmae_regressor import Mesh_regressor

PCA_DIM = 50 
WHITEN_K = 38         
RIDGE_L = 1e-3        

_DECODER_PREFIX = ("decoder", "decoer_pos_embedding", "to_points", "to_features", "mask_token", "loss_func")


def check_ckpt_keys(net, sd, ckpt_path, tag):
    m, u = net.load_state_dict(sd, strict=False)
    enc_missing = [k for k in m if not k.startswith("head")]
    unex_ok = [k for k in u if k.startswith(_DECODER_PREFIX)]
    unex_other = [k for k in u if not k.startswith(_DECODER_PREFIX)]
    print(f"[ckpt:{tag}] missing(encoder)={len(enc_missing)} "
          f"unexpected(decoder-only)={len(unex_ok)} unexpected(other)={len(unex_other)}", flush=True)
    if enc_missing or unex_other:
        print(f"  [FAIL] missing={enc_missing[:5]} unexpected={unex_other[:5]}", flush=True)
        sys.exit(2)


def load_labels(label_dir):
    lab = np.load(f"{label_dir}/labels.npy")
    sids = np.load(f"{label_dir}/subject_ids.npy", allow_pickle=True)
    m = {str(s): float(v) for s, v in zip(sids, lab)}
    return m, [str(s) for s in sids]


def site_of(sid):
    return str(sid).split("_")[0]


@torch.no_grad()
def embed_hemisphere(feat_dir, ckpt, device, batch=16, nan_stats=None):
    ds = OfflineFeatureDataset(feat_dir, canonical=True, ram_cache=True, dist="geodesic", nan_stats=nan_stats)
    net = Mesh_regressor(channels=10, num_heads=6, encoder_depth=6,
                            embed_dim=384, patch_size=64, drop_path=0.1).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    check_ckpt_keys(net, sd, ckpt, tag=os.path.basename(os.path.dirname(ckpt)))
    net.eval()
    cap = {}
    hook = net.norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", out))
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    embs, sids = [], []
    for b in loader:
        feats, center, coord, faces, Fs, _, sid, hop = b
        net(faces.float().to(device), feats.float().to(device), center.float().to(device),
            coord.float().to(device), hop.float().to(device))
        x = cap["x"][:, 1:, :].mean(dim=1).cpu().numpy()   
        embs.append(x); sids.extend([str(s) for s in sid])
    hook.remove()
    del net
    torch.cuda.empty_cache()
    return np.concatenate(embs, 0).astype(np.float32), np.asarray(sids, dtype=object)


def subj_stats(data_path):
    d = np.load(data_path, mmap_mode="r"); n = d.shape[0]; o = np.zeros((n, 50), np.float32)
    for i in range(n):
        f = np.asarray(d[i], dtype=np.float32).reshape(10, -1)
        o[i] = np.concatenate([f.mean(1), f.std(1),
                               np.percentile(f, [10, 50, 90], axis=1).T.flatten()])
    return o


def fit_moment_alignment(Etr, Rtr, dim=PCA_DIM, ridge_l=RIDGE_L, whiten_k=WHITEN_K):
    muE = Etr.mean(0)
    Ec = Etr - muE
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    V = Vt[:dim].T
    Ep = Ec @ V
    muEp = Ep.mean(0)
    Epc = Ep - muEp
    covE = (Epc.T @ Epc) / len(Etr) + ridge_l * np.trace((Epc.T @ Epc) / len(Etr)) / dim * np.eye(dim)
    wE, VE = np.linalg.eigh(covE)
    wc = np.clip(wE, 1e-8, None)
    if whiten_k is not None and whiten_k < dim:
        wc[:-whiten_k] = 1.0
    WE = (VE @ np.diag(1.0 / np.sqrt(wc))) @ VE.T
    muR = Rtr.mean(0)
    Rc = Rtr - muR
    covR = (Rc.T @ Rc) / len(Rtr) + ridge_l * np.trace((Rc.T @ Rc) / len(Rtr)) / dim * np.eye(dim)
    wR, VR = np.linalg.eigh(covR)
    WR = (VR @ np.diag(np.sqrt(np.clip(wR, 0, None)))) @ VR.T

    def transform(X):
        return ((X - muE) @ V - muEp) @ WE @ WR + muR
    return transform


def metrics(p, t):
    p = p.astype(float); t = t.astype(float)
    pcc = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else 0.0
    mae = float(np.abs(p - t).mean()); ss = ((t - t.mean()) ** 2).sum()
    return pcc, mae, float(1 - ((p - t) ** 2).sum() / max(ss, 1e-9))


def standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-lh", required=True,
                    help="path to the pretrained left-hemisphere encoder checkpoint")
    ap.add_argument("--ckpt-rh", required=True,
                    help="path to the pretrained right-hemisphere encoder checkpoint")
    ap.add_argument("--lh-dir", required=True,
                    help="ADOS left-hemisphere offline feature directory")
    ap.add_argument("--rh-dir", required=True,
                    help="ADOS right-hemisphere offline feature directory")
    ap.add_argument("--label-dir", required=True,
                    help="label directory (labels.npy / subject_ids.npy)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--nan", action="store_true",
                    help="apply NAN normalization to encoder inputs (stats from each fold's training subjects)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
    for c in (args.ckpt_lh, args.ckpt_rh):
        if not os.path.exists(c):
            print(f"[FAIL] checkpoint not found: {c}", flush=True); sys.exit(2)

    EL, SL = embed_hemisphere(args.lh_dir, args.ckpt_lh, dev)
    ER, SR = embed_hemisphere(args.rh_dir, args.ckpt_rh, dev)
    print(f"[embed] LH {EL.shape} RH {ER.shape}", flush=True)

    lab_map, _ = load_labels(args.label_dir)
    rp = {s: i for i, s in enumerate(SR)}
    common = [s for s in SL if s in rp and s in lab_map]
    lp = {s: i for i, s in enumerate(SL)}
    EMB = np.concatenate([EL[[lp[s] for s in common]], ER[[rp[s] for s in common]]], 1).astype(np.float32)
    ls = [str(s) for s in np.load(f"{args.lh_dir}/subject_ids.npy", allow_pickle=True)]
    rs = [str(s) for s in np.load(f"{args.rh_dir}/subject_ids.npy", allow_pickle=True)]
    lq = {s: i for i, s in enumerate(ls)}; rq = {s: i for i, s in enumerate(rs)}
    RL = subj_stats(f"{args.lh_dir}/data.npy")
    RR = subj_stats(f"{args.rh_dir}/data.npy")
    RAW = np.concatenate([RL[[lq[s] for s in common]], RR[[rq[s] for s in common]]], 1)
    y = np.array([lab_map[s] for s in common], dtype=np.float32)
    sites = np.array([site_of(s) for s in common])
    if not np.isfinite(EMB).all() or not np.isfinite(RAW).all():
        print("[FAIL] embeddings or input statistics contain NaN/Inf", flush=True); sys.exit(2)
    uniq = [s for s in sorted(set(sites.tolist())) if int((sites == s).sum()) > 20]
    print(f"N={len(common)} n>20 sites={ {s: int((sites == s).sum()) for s in uniq} }", flush=True)

    ldir = args.lh_dir; rdir = args.rh_dir
    l_data = np.load(ldir + "/data.npy", mmap_mode="r")
    l_cmap = np.load(ldir + "/canonical_mapping.npy")
    r_data = np.load(rdir + "/data.npy", mmap_mode="r")
    r_cmap = np.load(rdir + "/canonical_mapping.npy")
    res = {}
    for s in uniq:
        tr = np.where(sites != s)[0]; te = np.where(sites == s)[0]
        if args.nan:
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, [lq[c] for c in common[tr]])
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, [rq[c] for c in common[tr]])
            ELf, SLf = embed_hemisphere(ldir, args.ckpt_lh, dev, nan_stats={"mu": mu_l, "sigma": sigma_l})
            ERf, SRf = embed_hemisphere(rdir, args.ckpt_rh, dev, nan_stats={"mu": mu_r, "sigma": sigma_r})
            lp2 = {str(x): i for i, x in enumerate(SLf)}; rp2 = {str(x): i for i, x in enumerate(SRf)}
            EMBf = np.concatenate([ELf[[lp2[c] for c in common]], ERf[[rp2[c] for c in common]]], 1).astype(np.float32)
            E_tr, E_te = EMBf[tr], EMBf[te]
        else:
            E_tr, E_te = EMB[tr], EMB[te]
        tL = fit_moment_alignment(E_tr[:, :384], RAW[tr][:, :50])
        tR = fit_moment_alignment(E_tr[:, 384:], RAW[tr][:, 50:])
        Xtr = np.concatenate([tL(E_tr[:, :384]), tR(E_tr[:, 384:])], 1)
        Xte = np.concatenate([tL(E_te[:, :384]), tR(E_te[:, 384:])], 1)
        Xtr, Xte = standardize(Xtr, Xte)
        p = SVR(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, y[tr]).predict(Xte)
        res[s] = metrics(p, y[te])

    print(f"\n{'site':<8}{'n':>4}{'PCC':>9}{'MAE':>8}{'COD':>9}", flush=True)
    for s in uniq:
        n = int((sites == s).sum())
        print(f"{s:<8}{n:>4}{res[s][0]:>+9.3f}{res[s][1]:>8.3f}{res[s][2]:>+9.3f}", flush=True)
    P = [res[s][0] for s in uniq]; M = [res[s][1] for s in uniq]; C = [res[s][2] for s in uniq]
    print(f"{'mean':<8}{'':>4}{np.mean(P):>+9.3f}{np.mean(M):>8.3f}{np.mean(C):>+9.3f}"
          f"   (PCC +/-{np.std(P):.3f})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "task": "ADOS calibrated severity, leave-one-site-out",
        "n_subjects": len(common), "sites": {s: int((sites == s).sum()) for s in uniq},
        "pca_dim": PCA_DIM, "whiten_k": WHITEN_K, "ridge_l": RIDGE_L,
        "svr": "rbf C=1.0 gamma=scale",
        "ckpt_lh": args.ckpt_lh, "ckpt_rh": args.ckpt_rh, "nan": args.nan,
        "per_site": {s: list(res[s]) for s in uniq},
        "summary": {"pcc_mean": float(np.mean(P)), "pcc_sd": float(np.std(P)),
                    "mae_mean": float(np.mean(M)), "cod_mean": float(np.mean(C))},
    }
    with open(f"{args.out_dir}/ados_loso.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_dir}/ados_loso.json", flush=True)


if __name__ == "__main__":
    main()
