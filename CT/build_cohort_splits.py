import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PHENO_28 = [
    "MMSE_Score", "PSQI_Score", "PicSeq_Unadj", "CardSort_Unadj", "Flanker_Unadj",
    "PMAT24_A_SI", "ReadEng_Unadj", "PicVocab_Unadj", "ProcSpeed_Unadj", "DDisc_SV_6mo_40K",
    "VSPLOT_TC", "SCPT_SEN", "ListSort_Unadj", "CogFluidComp_Unadj", "CogEarlyComp_Unadj",
    "CogTotalComp_Unadj", "CogCrystalComp_Unadj", "Sadness_Unadj", "PercHostil_Unadj",
    "Emotion_Task_Acc", "Language_Task_Acc", "Relational_Task_Acc", "WM_Task_Acc",
    "Social_Task_Perc_Random", "Endurance_Unadj", "Strength_Unadj", "NEOFAC_A", "Noise_Comp",
]
SEEDS = [1, 11, 16]
N_TEST = 81
N_VAL = 73


def zfill6(s):
    return str(int(s)).zfill(6)


def main():
    ap = argparse.ArgumentParser(description="write the frozen cohort table and per-seed splits")
    ap.add_argument("--lh-dir", required=True,
                    help="left-hemisphere offline feature directory (subject_ids.npy)")
    ap.add_argument("--rh-dir", required=True,
                    help="right-hemisphere offline feature directory (subject_ids.npy)")
    ap.add_argument("--labels-dir", required=True,
                    help="labels directory (phenotypes.npy / phenotype_names.json / "
                         "brain_size_labels.csv / subject_ids.npy)")
    ap.add_argument("--out-dir", required=True,
                    help="output directory (cohort.csv / splits.csv / audit.json)")
    args = ap.parse_args()
    lh_dir, rh_dir, labels = args.lh_dir, args.rh_dir, args.labels_dir
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lh = sorted(zfill6(s) for s in np.load(f"{lh_dir}/subject_ids.npy", allow_pickle=True))
    rh = sorted(zfill6(s) for s in np.load(f"{rh_dir}/subject_ids.npy", allow_pickle=True))
    common = sorted(set(lh) & set(rh))

    arr = np.load(f"{labels}/phenotypes.npy")
    lab_sids = [zfill6(s) for s in np.load(f"{labels}/subject_ids.npy", allow_pickle=True)]
    names = json.loads((Path(labels) / "phenotype_names.json").read_text())
    assert lab_sids == lh, "labels/HCP/subject_ids.npy != LH mapped subject_ids.npy"
    i_age, i_sex = names.index("Age"), names.index("Gender")
    ip = [names.index(p) for p in PHENO_28]
    assert len(ip) == 28 and len(set(ip)) == 28
    sid2idx = {s: i for i, s in enumerate(lab_sids)}

    bdf = pd.read_csv(f"{labels}/brain_size_labels.csv", dtype={"subject_id": str})
    bdf["subject_id"] = bdf["subject_id"].str.zfill(6)
    etiv = {r["subject_id"]: float(r["eTIV"]) for _, r in bdf.iterrows()
            if pd.notna(r["eTIV"])}

    age_ok = lambda s: np.isfinite(arr[sid2idx[s], i_age])
    sex_ok = lambda s: np.isfinite(arr[sid2idx[s], i_sex])
    etiv_ok = lambda s: s in etiv
    pheno_ok = lambda s: all(np.isfinite(arr[sid2idx[s], j]) for j in ip)
    audit_stages = {
        "LH mapped": len(lh),
        "RH mapped": len(rh),
        "LH ∩ RH": len(common),
        "+ Age finite": sum(1 for s in common if age_ok(s)),
        "+ Gender finite": sum(1 for s in common if age_ok(s) and sex_ok(s)),
        "+ eTIV finite": sum(1 for s in common if age_ok(s) and sex_ok(s) and etiv_ok(s)),
        "+ 28 phenos all finite": sum(
            1 for s in common
            if age_ok(s) and sex_ok(s) and etiv_ok(s) and pheno_ok(s)
        ),
    }
    cohort = [s for s in common if age_ok(s) and sex_ok(s) and etiv_ok(s) and pheno_ok(s)]
    n_total = len(cohort)
    n_train = n_total - N_TEST - N_VAL
    assert n_train > 0, f"train<=0: total={n_total}"

    rows = []
    for s in cohort:
        i = sid2idx[s]
        rows.append({
            "Subject": s,
            "Age": float(arr[i, i_age]),
            "Gender": int(arr[i, i_sex]),
            "eTIV": float(etiv[s]),
            **{p: float(arr[i, names.index(p)]) for p in PHENO_28},
        })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out / "cohort.csv", index=False)

    split_rows = []
    splits_dict = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = cohort.copy()
        rng.shuffle(shuffled)
        te_list = shuffled[:N_TEST]
        va_list = shuffled[N_TEST:N_TEST + N_VAL]
        tr_list = shuffled[N_TEST + N_VAL:]
        assert len(va_list) == N_VAL and len(te_list) == N_TEST
        for s in tr_list: split_rows.append((s, seed, "train"))
        for s in va_list: split_rows.append((s, seed, "val"))
        for s in te_list: split_rows.append((s, seed, "test"))
        splits_dict[seed] = {"train": tr_list, "val": va_list, "test": te_list}
    pd.DataFrame(split_rows, columns=["subject_id", "seed", "partition"]).to_csv(
        out / "splits.csv", index=False
    )

    cohort_hash = hashlib.sha1(",".join(cohort).encode()).hexdigest()[:12]
    audit = {
        "frozen_at": pd.Timestamp.now().isoformat(),
        "cohort_sha1_first12": cohort_hash,
        "audit_stages": audit_stages,
        "n_cohort": n_total,
        "n_train": n_train,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "train_pct": round(n_train / n_total, 4),
        "val_pct": round(N_VAL / n_total, 4),
        "test_pct": round(N_TEST / n_total, 4),
        "seeds": SEEDS,
        "phenos": PHENO_28,
        "confound_regressors": ["Age", "Gender", "eTIV"],
        "sources": {
            "lh_dir": lh_dir, "rh_dir": rh_dir, "labels_dir": labels,
            "phenotypes_npy": f"{labels}/phenotypes.npy",
            "etiv_csv": f"{labels}/brain_size_labels.csv",
        },
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    test_sets = {sd: set(splits_dict[sd]["test"]) for sd in SEEDS}
    for i1 in range(len(SEEDS)):
        for i2 in range(i1 + 1, len(SEEDS)):
            overlap = len(test_sets[SEEDS[i1]] & test_sets[SEEDS[i2]])
            print(f"  test overlap seed{SEEDS[i1]} ∩ seed{SEEDS[i2]} = {overlap}/{N_TEST}")

    print(f"\n=== FROZEN cohort_v1 ===")
    for k, v in audit_stages.items():
        print(f"  {k:<26} {v}")
    print(f"  {'FINAL cohort':<26} {n_total}")
    print(f"  split: train={n_train} val={N_VAL} test={N_TEST}  (ratios "
          f"{n_train/n_total:.3f}/{N_VAL/n_total:.3f}/{N_TEST/n_total:.3f})")
    print(f"  sha1(12): {cohort_hash}")
    print(f"  artifact: {out}/")
    print(f"    cohort.csv   ({len(cdf)} rows x {len(cdf.columns)} cols)")
    print(f"    splits.csv   ({len(split_rows)} rows = {n_total} subj x {len(SEEDS)} seeds)")
    print(f"    audit.json")


if __name__ == "__main__":
    main()
