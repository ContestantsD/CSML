import argparse
from pathlib import Path
import numpy as np
import scipy.io


def main():
    ap = argparse.ArgumentParser(
        description="build FC label directories (fc.npy / subject_ids.npy / global stats) "
                    "from parcel correlation matrices")
    ap.add_argument("--subject-ids", required=True,
                    help="offline subject_ids.npy defining the subject set")
    ap.add_argument("--fc-dir", action="append", required=True, metavar="SIZE=DIR",
                    help="parcel size and matrix directory, e.g. 100=/path/to/fc_matrices; repeatable")
    ap.add_argument("--out", required=True, help="label output root (writes OUT/FC<SIZE>/)")
    ap.add_argument("--pattern", default="{sid}_REST1_LR_Full.mat",
                    help=".mat filename template ({sid} substituted)")
    args = ap.parse_args()

    sids = [str(s).split(".")[0].split("_")[0]
            for s in np.load(args.subject_ids, allow_pickle=True)]
    print(f"offline subjects: {len(sids)}")

    for spec in args.fc_dir:
        size, sep, d = spec.partition("=")
        if not sep or not size.isdigit():
            raise SystemExit(f"--fc-dir expects SIZE=DIR, got: {spec}")
        size = int(size)
        d = Path(d)
        mats, fc_sids, missing = [], [], []
        for sid in sids:
            f = d / args.pattern.format(sid=sid)
            if not f.exists():
                missing.append(sid)
                continue
            m = scipy.io.loadmat(str(f))
            mats.append(np.array(m["connectivity_matrix"], dtype=np.float32))
            fc_sids.append(sid)
        mats = np.stack(mats)
        gmean, gstd = float(mats.mean()), float(mats.std())
        out = Path(args.out) / f"FC{size}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "fc.npy", mats)
        np.save(out / "subject_ids.npy", np.asarray(fc_sids, dtype=object))
        np.save(out / "global_mean.npy", np.float32(gmean))
        np.save(out / "global_std.npy", np.float32(gstd))
        print(f"FC{size}: N_fc={len(fc_sids)}/{len(sids)} (missing={len(missing)}) shape={mats.shape} "
              f"global_mean={gmean:.6f} global_std={gstd:.6f}")


if __name__ == "__main__":
    main()
