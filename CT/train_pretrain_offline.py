import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import sys
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

_THIS_DIR = Path(__file__).resolve().parent
_CT_ROOT = os.environ.get("CSML_CT_ROOT", str(_THIS_DIR))
if _CT_ROOT not in sys.path:
    sys.path.insert(0, _CT_ROOT)

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from dataset_offline import OfflineFeatureDataset, compute_nan_stats


def build_model(args, device, num_patches):
    from meshmae_backbone import Mesh_mae as M
    extra = dict(path_mode=getattr(args, "path_mode", "dual"))
    net = M(masking_ratio=args.mask_ratio, channels=args.channels, patch_size=args.patch_size,
            num_patches=num_patches,
            embed_dim=args.embed_dim, encoder_depth=args.encoder_depth,
            num_heads=args.heads, decoder_depth=args.decoder_depth,
            decoder_embed_dim=args.decoder_dim, decoder_num_heads=args.decoder_num_heads,
            weight=args.feat_weight, **extra)
    return net.to(device)


def build_optimizer(net, args, num_training_steps):
    NO_DECAY = {"bias", "norm", "embed", "token", "gate"}
    decay, no_decay = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in name.lower() for k in NO_DECAY) else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    warmup = max(1, int(args.warmup_ratio * num_training_steps))
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, num_training_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched, warmup


def pretrain(net, loader, epochs, device, args, save_dir, tag="A"):
    accum = max(1, args.grad_accum)
    sched_epochs = args.cosine_tmax_epochs if args.cosine_tmax_epochs > 0 else epochs
    n_steps = sched_epochs * max(1, len(loader) // accum)
    opt, sched, warmup = build_optimizer(net, args, n_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print(f"[{tag}] pretrain {len(loader.dataset)} samples, {epochs} ep, {n_steps} opt-steps, "
          f"micro_batch={args.batch} accum={accum} eff_batch={args.batch*accum}, "
          f"warmup {warmup}, lr {args.lr}, wd {args.weight_decay}, amp {args.amp}", flush=True)

    best = float("inf")
    for ep in range(epochs):
        net.train()
        t0 = time.time(); run = 0.0; rn = 0
        for it, batch in enumerate(loader):
            feats, center, coord, faces, Fs, _, _, hop = batch
            feats = feats.float().to(device, non_blocking=True)
            center = center.float().to(device, non_blocking=True)
            coord = coord.float().to(device, non_blocking=True)
            faces = faces.float().to(device, non_blocking=True)
            Fs = Fs.to(device, non_blocking=True)
            hop = hop.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                loss, f_loss, s_loss = net(faces, feats, center, Fs, coord,
                                           hop_matrix=hop, ratio=args.mask_ratio)
            loss_val = loss.item()
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or (it + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            run += loss_val * feats.size(0); rn += feats.size(0)

            if it < 2 or (it + 1) % 100 == 0:
                cur_lr = opt.param_groups[0]["lr"]
                print(f"  [ep{ep} it{it+1}/{len(loader)}] loss={loss_val:.4f} "
                      f"(f={f_loss.item():.4f} s={s_loss.item():.4f}) lr={cur_lr:.2e}", flush=True)

        ep_loss = run / max(rn, 1)
        print(f"== [{tag}] ep{ep:02d} loss={ep_loss:.4f} t={time.time()-t0:.1f}s "
              f"lr={opt.param_groups[0]['lr']:.2e} ==", flush=True)

        if ep_loss < best:
            best = ep_loss
            torch.save(net.state_dict(), os.path.join(save_dir, f"{tag}_best.pkl"))
        if (ep + 1) in getattr(args, "save_epochs", []):
            torch.save(net.state_dict(), os.path.join(save_dir, f"{tag}_ep{ep+1}.pkl"))
            print(f"[{tag}] saved ckpt @ ep{ep+1}", flush=True)
        torch.save(net.state_dict(), os.path.join(save_dir, f"{tag}_last.pkl"))
    print(f"[{tag}] best epoch loss={best:.4f}, ckpt -> {save_dir}/{tag}_best.pkl", flush=True)


def select_frames_per_subject(sids, k):
    if k <= 0 or k >= 10:
        return None
    sids = np.asarray(sids)
    idx = []
    cur, start = sids[0], 0
    for j in range(1, len(sids) + 1):
        if j == len(sids) or sids[j] != cur:
            n = j - start
            positions = np.linspace(0, n - 1, k).round().astype(int)
            for p in positions:
                idx.append(start + int(p))
            if j < len(sids):
                cur, start = sids[j], j
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat-dirs', required=True,
                    help="comma-separated offline npy directories (with hop.npy)")
    ap.add_argument('--sid-list', default=None,
                    help="comma-separated subject-id list files; only listed subjects are used")
    ap.add_argument('--save-dir', required=True, help="checkpoint output directory")
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--n-worker', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--embed-dim', type=int, default=384)
    ap.add_argument('--encoder-depth', type=int, default=6)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--decoder-depth', type=int, default=6)
    ap.add_argument('--decoder-dim', type=int, default=512)
    ap.add_argument('--decoder-num-heads', type=int, default=8)
    ap.add_argument('--patch-size', type=int, default=64)
    ap.add_argument('--num-patches', type=int, default=0,
                    help="number of patch tokens; 0=auto-infer from first feat dir data.npy shape[2]")
    ap.add_argument('--channels', type=int, default=10)
    ap.add_argument('--mask-ratio', type=float, default=0.5)
    ap.add_argument('--feat-weight', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--warmup-ratio', type=float, default=0.05)
    ap.add_argument('--amp', action='store_true', help="mixed precision")
    ap.add_argument('--cosine-tmax-epochs', type=int, default=0,
                    help="cosine T_max in epochs (0=use --epochs)")
    ap.add_argument('--nan', action='store_true',
                    help="per-dataset NAN normalization (each dir computes its own mu_j/sigma_j)")
    ap.add_argument('--dist', choices=['hop', 'geodesic'], default='geodesic',
                    help="attention distance matrix: hop (BFS hops) | geodesic (surface mm, reads geodesic.npy)")
    ap.add_argument('--path-mode', choices=["dual", "local", "global"], default="dual",
                    help="dual=dual-path fusion (default); local=region_attn only; global=standard_attn only")
    ap.add_argument('--init-sigma', type=float, default=0.0,
                    help="bias_generator sigma init (>0 to activate); geodesic default 50mm, else hop default 2.0")
    ap.add_argument('--frames-per-subj', type=int, default=0,
                    help="deformed frame subsampling: k equally-spaced frames per subject (0=all, 3=subsample 3)")
    ap.add_argument('--no-ram-cache', action='store_true',
                    help="disable ram_cache (default on: load selected samples into RAM)")
    ap.add_argument('--grad-accum', type=int, default=1,
                    help="gradient accumulation steps (>1: micro-batch=--batch accumulated N steps)")
    ap.add_argument('--save-epochs', type=str, default="",
                    help="comma-separated epoch checkpoints (e.g. 50,100,150)")
    ap.add_argument('--resume', default=None,
                    help="resume from checkpoint (e.g. A_last.pkl); optimizer/sched restart from scratch")
    args = ap.parse_args()
    args.save_epochs = [int(x) for x in args.save_epochs.split(",") if x.strip()] if args.save_epochs else []

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    os.makedirs(args.save_dir, exist_ok=True)

    dirs = [d.strip() for d in args.feat_dirs.split(",") if d.strip()]
    allowed = None
    if args.sid_list:
        allowed = set()
        for p in [x.strip() for x in args.sid_list.split(",") if x.strip()]:
            with open(p) as f:
                allowed |= set(l.strip() for l in f if l.strip())
        print(f"[data] sid lists: {len(allowed)} subject ids")
    ram_cache = not args.no_ram_cache
    ds_list = []
    for d in dirs:
        subj_idx = None
        sids = None
        if allowed is not None or args.frames_per_subj > 0:
            sids = np.load(d + "/subject_ids.npy", allow_pickle=True)
        if allowed is not None:
            n_all = len(sids)
            subj_idx = [i for i, s in enumerate(sids) if str(s) in allowed]
            if not subj_idx:
                sys.exit(f"[FAIL] no sid-list subjects found in {d}")
            sids = np.asarray(sids)[subj_idx]
            print(f"[data] {d}: sid-list -> {len(subj_idx)} rows (of {n_all})")
        if args.frames_per_subj > 0:
            frame_idx = select_frames_per_subject(sids, args.frames_per_subj)
            if frame_idx is not None:
                subj_idx = [subj_idx[i] for i in frame_idx] if subj_idx is not None else list(frame_idx)
                print(f"[data] {d}: frames_per_subj={args.frames_per_subj} -> {len(subj_idx)} frames (of {len(sids)})")
        nan_stats = None
        if args.nan:
            d_data = np.load(d + "/data.npy", mmap_mode="r")
            d_cmap = np.load(d + "/canonical_mapping.npy")
            stat_idx = subj_idx if subj_idx is not None else list(range(d_data.shape[0]))
            mu_j, sigma_j = compute_nan_stats(d_data, d_cmap, stat_idx)
            nan_stats = {"mu": mu_j, "sigma": sigma_j}
            print(f"[data] {d}: NAN per-dataset stats on {len(stat_idx)} frames")
        ds = OfflineFeatureDataset(d, subject_indices=subj_idx, canonical=True, nan_stats=nan_stats, ram_cache=ram_cache, dist=args.dist)
        print(f"[data] {d}: {len(ds)} samples, ram_cache={ram_cache}, nan={'on' if nan_stats else 'off'}")
        ds_list.append(ds)
    full_ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
    print(f"[data] total {len(full_ds)} samples, canonical=True, hop loaded")

    loader = DataLoader(full_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.n_worker, pin_memory=True, drop_last=True)

    if args.num_patches > 0:
        num_patches = args.num_patches
    else:
        _d0 = np.load(dirs[0] + "/data.npy", mmap_mode="r")
        num_patches = _d0.shape[2]
    print(f"[model] num_patches = {num_patches}")
    torch.manual_seed(args.seed)
    net = build_model(args, device, num_patches)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        net.load_state_dict(sd)
        print(f"[resume] loaded model weights from {args.resume} (optimizer/sched fresh, new cosine)", flush=True)
    if args.dist == 'geodesic':
        sg = args.init_sigma if args.init_sigma > 0 else 50.0
        with torch.no_grad():
            net.region_assigner.bias_generator.log_sigma.fill_(math.log(sg))
        print(f"[dist] geodesic: bias_generator sigma init={sg:.1f}mm", flush=True)
    print(f"\n=== [A] pretrain {len(full_ds)} samples, {args.epochs}ep ===")
    print(f"=== [A] variant: path_mode={getattr(args, 'path_mode', 'dual')}, dist={args.dist}, nan={args.nan} ===", flush=True)
    pretrain(net, loader, args.epochs, device, args, args.save_dir, tag="A")


if __name__ == "__main__":
    main()
