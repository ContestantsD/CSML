import argparse
import os
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def _parse_args():
    ap = argparse.ArgumentParser(description="paired bootstrap comparison of CSML and MeshMAE phenotype predictions")
    ap.add_argument("--csml-root", required=True, help="root directory of the CSML phenotype run outputs")
    ap.add_argument("--csml-run", required=True, help="CSML run name (prefix before __seed<seed>__)")
    ap.add_argument("--meshmae-root", required=True, help="root directory of the MeshMAE phenotype run outputs")
    ap.add_argument("--meshmae-run", required=True, help="MeshMAE run name")
    ap.add_argument("--out", required=True, help="output directory for the bootstrap tables")
    return ap.parse_args()


_A = _parse_args()
CSML, MM, OUT = _A.csml_root, _A.meshmae_root, _A.out
CSML_RUN, MESHMAE_RUN = _A.csml_run, _A.meshmae_run
PHENOS = [
    "MMSE_Score", "PSQI_Score", "PicSeq_Unadj", "CardSort_Unadj", "Flanker_Unadj",
    "PMAT24_A_SI", "ReadEng_Unadj", "PicVocab_Unadj", "ProcSpeed_Unadj", "DDisc_SV_6mo_40K",
    "VSPLOT_TC", "SCPT_SEN", "ListSort_Unadj", "CogFluidComp_Unadj", "CogEarlyComp_Unadj",
    "CogTotalComp_Unadj", "CogCrystalComp_Unadj", "Sadness_Unadj", "PercHostil_Unadj",
    "Emotion_Task_Acc", "Language_Task_Acc", "Relational_Task_Acc", "WM_Task_Acc",
    "Social_Task_Perc_Random", "Endurance_Unadj", "Strength_Unadj", "NEOFAC_A", "Noise_Comp",
]
SEEDS = [1, 11, 16]
NBOOT = 10000
RNG_SEED = 0
METRICS = ["pcc", "cod", "srmse"]
def metric_block(yt, yp):
    m = yt.mean(axis=1, keepdims=True)
    yc = yp - yp.mean(axis=1, keepdims=True)
    ytc = yt - m
    denom = np.sqrt((ytc ** 2).sum(1) * (yc ** 2).sum(1))
    pcc = (ytc * yc).sum(1) / np.maximum(denom, 1e-12)
    ss_tot = (ytc ** 2).sum(1)
    ss_res = ((yt - yp) ** 2).sum(1)
    cod = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    srmse = np.sqrt(((yt - yp) ** 2).mean(1))
    return pcc, cod, srmse

def load_seed(seed):
    yt_all, cs_all, mm_all = [], [], []
    for ph in PHENOS:
        a = pd.read_csv(f"{CSML}/{CSML_RUN}__seed{seed}__{ph}/predictions.csv")
        b = pd.read_csv(f"{MM}/{MESHMAE_RUN}__seed{seed}__{ph}/predictions.csv")
        a = a[a.partition == "test"].sort_values("subject_id").reset_index(drop=True)
        b = b[b.partition == "test"].sort_values("subject_id").reset_index(drop=True)
        assert len(a) == len(b) == 81
        assert (a.subject_id.astype(str).str.zfill(6).values == b.subject_id.astype(str).str.zfill(6).values).all()
        yt_all.append(a.y_true_z.values); cs_all.append(a.y_pred_z.values); mm_all.append(b.y_pred_z.values)
    return np.stack(yt_all), np.stack(cs_all), np.stack(mm_all)
def bootstrap_seed(seed, yt, cs, mm, nboot=NBOOT):
    n_subj = yt.shape[1]
    pcc_c, cod_c, srm_c = metric_block(yt, cs)
    pcc_m, cod_m, srm_m = metric_block(yt, mm)
    obs = {
        "pcc": (float(pcc_c.mean()), float(pcc_m.mean())),
        "cod": (float(cod_c.mean()), float(cod_m.mean())),
        "srmse": (float(srm_c.mean()), float(srm_m.mean())),
    }
    rng = np.random.default_rng(RNG_SEED)
    D = {m: np.zeros((nboot, 28)) for m in METRICS}
    for b in range(nboot):
        idx = rng.integers(0, n_subj, n_subj)
        ytb, csb, mmb = yt[:, idx], cs[:, idx], mm[:, idx]
        pcc_cb, cod_cb, srm_cb = metric_block(ytb, csb)
        pcc_mb, cod_mb, srm_mb = metric_block(ytb, mmb)
        D["pcc"][b] = pcc_cb - pcc_mb
        D["cod"][b] = cod_cb - cod_mb
        D["srmse"][b] = srm_cb - srm_mb
    return obs, D, dict(pcc=(pcc_c, pcc_m), cod=(cod_c, cod_m), srmse=(srm_c, srm_m))

def ci_p(diffs, obs_diff):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return lo, hi, p

def bootstrap_pooled(seed_data, nboot=NBOOT):
    rng = np.random.default_rng(RNG_SEED + 1)
    D = {m: np.zeros(nboot) for m in METRICS}
    n_seed = len(seed_data)
    for b in range(nboot):
        acc = {m: 0.0 for m in METRICS}
        for seed, (yt, cs, mm) in seed_data.items():
            idx = rng.integers(0, yt.shape[1], yt.shape[1])
            ytb, csb, mmb = yt[:, idx], cs[:, idx], mm[:, idx]
            pcc_cb, cod_cb, srm_cb = metric_block(ytb, csb)
            pcc_mb, cod_mb, srm_mb = metric_block(ytb, mmb)
            acc["pcc"] += pcc_cb.mean() - pcc_mb.mean()
            acc["cod"] += cod_cb.mean() - cod_mb.mean()
            acc["srmse"] += srm_cb.mean() - srm_mb.mean()
        for m in METRICS:
            D[m][b] = acc[m] / n_seed
    return D
os.makedirs(OUT, exist_ok=True)
SEED_DATA = {s: load_seed(s) for s in SEEDS}
macro_rows, pheno_rows = [], []
for seed in SEEDS:
    obs, D, obs_p = bootstrap_seed(seed, *SEED_DATA[seed])
    for m in METRICS:
        obs_c, obs_m = obs[m]
        obs_d = obs_c - obs_m
        macro_d = D[m].mean(axis=1)
        lo, hi, p = ci_p(macro_d, obs_d)
        macro_rows.append(dict(seed=seed, metric=m, csml=obs_c, meshmae=obs_m,
                               diff=obs_d, ci_lo=lo, ci_hi=hi, p_boot=p))
        oc, om = obs_p[m]
        for j, ph in enumerate(PHENOS):
            d = D[m][:, j]
            loj, hij, pj = ci_p(d, None)
            pheno_rows.append(dict(seed=seed, phenotype=ph, metric=m,
                                   csml=float(oc[j]), meshmae=float(om[j]),
                                   diff=float(oc[j] - om[j]),
                                   ci_lo=loj, ci_hi=hij, p_boot=pj))
    print(f"[seed {seed}] macro bootstrap done", flush=True)
dfm = pd.DataFrame(macro_rows)
dfp = pd.DataFrame(pheno_rows)
dfp["q_bh"] = np.nan
dfp["sig_q05"] = False
for (seed, m), g in dfp.groupby(["seed", "metric"]):
    q = multipletests(g.p_boot.values, method="fdr_bh")[1]
    dfp.loc[g.index, "q_bh"] = q
    dfp.loc[g.index, "sig_q05"] = q < 0.05

dfm.to_csv(f"{OUT}/macro_bootstrap_by_seed.csv", index=False)
dfp.to_csv(f"{OUT}/phenotype_bootstrap_all.csv", index=False)
dfp[dfp.metric == "pcc"].to_csv(f"{OUT}/phenotype_bootstrap_pcc.csv", index=False)

ov = []
for m in METRICS:
    g = dfm[dfm.metric == m]
    signs = [np.sign(x) for x in g["diff"]]
    ov.append(dict(metric=m, mean_csml=g.csml.mean(), sd_csml=g.csml.std(ddof=1),
                   mean_meshmae=g.meshmae.mean(), sd_meshmae=g.meshmae.std(ddof=1),
                   mean_diff=g["diff"].mean(), sd_diff=g["diff"].std(ddof=1),
                   dir_all_same=(len(set(signs)) == 1), n_seed=len(g)))
pd.DataFrame(ov).to_csv(f"{OUT}/macro_bootstrap_overall_summary.csv", index=False)
PD = bootstrap_pooled(SEED_DATA)
pooled_rows = []
for m in METRICS:
    g = dfm[dfm.metric == m]
    obs_c = float(g.csml.mean()); obs_m = float(g.meshmae.mean())
    diffs = PD[m]
    lo, hi, p = ci_p(diffs, None)
    pooled_rows.append(dict(metric=m, n_seeds=len(SEEDS), n_boot=len(diffs),
                            csml=obs_c, meshmae=obs_m, diff=obs_c - obs_m,
                            ci_lo=lo, ci_hi=hi, p_boot=p))
dfpooled = pd.DataFrame(pooled_rows)
dfpooled.to_csv(f"{OUT}/pooled_bootstrap_macro.csv", index=False)
r = dfpooled[dfpooled.metric == "pcc"].iloc[0]
print(f"[pooled over {len(SEEDS)} seeds] macro PCC: CSML {r.csml:.4f} vs MeshMAE {r.meshmae:.4f} "
      f"| diff {r['diff']:.4f} | 95% CI [{r.ci_lo:.4f}, {r.ci_hi:.4f}] | p={r.p_boot:.2e}", flush=True)
print("[save] csv done")
print("[macro]")
print(dfm.to_string(index=False))
print("[overall]")
print(pd.DataFrame(ov).to_string(index=False))
print("[sig phenotype PCC per seed]")
for seed in SEEDS:
    g = dfp[(dfp.metric == "pcc") & (dfp.seed == seed) & dfp.sig_q05]
    print(seed, list(g.phenotype))
