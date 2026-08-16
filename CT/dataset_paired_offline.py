import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_offline import OfflineFeatureDataset


class PairedOfflineFeatureDataset(Dataset):

    def __init__(self, l_dir, r_dir, subjects, phenotype, ram_cache=True, nan=False,
                 dist="hop", nan_stats_l=None, nan_stats_r=None):
        l_sids = np.load(l_dir + "/subject_ids.npy", allow_pickle=True)
        r_sids = np.load(r_dir + "/subject_ids.npy", allow_pickle=True)
        l_pos = {}
        for i, s in enumerate(l_sids):
            l_pos.setdefault(str(int(s)), i)
        r_pos = {}
        for i, s in enumerate(r_sids):
            r_pos.setdefault(str(int(s)), i)

        keep = [s for s in subjects if s in l_pos and s in r_pos]
        l_idx = [l_pos[s] for s in keep]
        r_idx = [r_pos[s] for s in keep]
        self.sids = keep
        self.phenotype = phenotype

        if nan and nan_stats_l is None:
            from dataset_offline import compute_nan_stats
            l_data = np.load(l_dir + "/data.npy", mmap_mode="r")
            l_cmap = np.load(l_dir + "/canonical_mapping.npy")
            r_data = np.load(r_dir + "/data.npy", mmap_mode="r")
            r_cmap = np.load(r_dir + "/canonical_mapping.npy")
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, l_idx)
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, r_idx)
            nan_stats_l = {"mu": mu_l, "sigma": sigma_l}
            nan_stats_r = {"mu": mu_r, "sigma": sigma_r}
            print(f"[paired-NAN fallback] L mu/sigma on {len(l_idx)} subj, R on {len(r_idx)} subj "
                  f"(WARN: per-partition, prefer pre-computed train stats)")

        self.l_ds = OfflineFeatureDataset(l_dir, subject_indices=l_idx, canonical=True,
                                          nan_stats=nan_stats_l, phenotype=phenotype,
                                          ram_cache=ram_cache, dist=dist)
        self.r_ds = OfflineFeatureDataset(r_dir, subject_indices=r_idx, canonical=True,
                                          nan_stats=nan_stats_r, phenotype=phenotype,
                                          ram_cache=ram_cache, dist=dist)
        print(f"[paired] {len(keep)} paired subjects dist={dist} nan={nan} "
              f"(L={l_dir.split('/')[-2]}, R={r_dir.split('/')[-2]})")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, k):
        lf, lc, lcd, lf2, lFs, label, lsid, lhop = self.l_ds[k]
        rf, rc, rcd, rf2, rFs, _, rsid, rhop = self.r_ds[k]
        return (lf, lc, lcd, lf2, lFs, lhop,
                rf, rc, rcd, rf2, rFs, rhop,
                label, self.sids[k])
