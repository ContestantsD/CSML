import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_offline import OfflineFeatureDataset


def _norm_sid(s):
    s = str(s)
    return s.split(".")[0].split("_")[0]


class PairedFCDataset(Dataset):

    def __init__(self, l_dir, r_dir, fc, fc_sids, subjects, ram_cache=True, dist="hop", nan=False, nan_stats=None):
        l_sids = [_norm_sid(s) for s in np.load(l_dir + "/subject_ids.npy", allow_pickle=True)]
        r_sids = [_norm_sid(s) for s in np.load(r_dir + "/subject_ids.npy", allow_pickle=True)]
        l_pos = {}
        for i, s in enumerate(l_sids):
            l_pos.setdefault(s, i)
        r_pos = {}
        for i, s in enumerate(r_sids):
            r_pos.setdefault(s, i)

        keep = [s for s in subjects if s in l_pos and s in r_pos and s in fc_sids]
        l_idx = [l_pos[s] for s in keep]
        r_idx = [r_pos[s] for s in keep]
        self.sids = keep

        self.size = fc.shape[-1]
        triu = np.triu_indices(self.size, k=1)
        self.triu = triu
        rows = [fc_sids[s] for s in keep]
        fc_vec = np.ascontiguousarray(fc[rows][:, triu[0], triu[1]]).astype(np.float32)
        self.n_out = fc_vec.shape[1]

        self.fc_mean = fc_vec.mean(axis=1, keepdims=True).astype(np.float32)
        self.fc_std = (fc_vec.std(axis=1, keepdims=True).astype(np.float32) + 1e-6)
        self.fc_target = ((fc_vec - self.fc_mean) / self.fc_std).astype(np.float32)

        self.l_ds = OfflineFeatureDataset(l_dir, subject_indices=l_idx, canonical=True, ram_cache=ram_cache, dist=dist)
        self.r_ds = OfflineFeatureDataset(r_dir, subject_indices=r_idx, canonical=True, ram_cache=ram_cache, dist=dist)
        if nan_stats is not None:
            self.l_ds = OfflineFeatureDataset(l_dir, subject_indices=l_idx, canonical=True, ram_cache=ram_cache, dist=dist,
                                              nan_stats=nan_stats["l"])
            self.r_ds = OfflineFeatureDataset(r_dir, subject_indices=r_idx, canonical=True, ram_cache=ram_cache, dist=dist,
                                              nan_stats=nan_stats["r"])
            print(f"[paired-FC NAN] using pre-computed train NAN stats")
        elif nan:
            from dataset_offline import compute_nan_stats
            l_data = np.load(l_dir + "/data.npy", mmap_mode="r")
            l_cmap = np.load(l_dir + "/canonical_mapping.npy")
            r_data = np.load(r_dir + "/data.npy", mmap_mode="r")
            r_cmap = np.load(r_dir + "/canonical_mapping.npy")
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, l_idx)
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, r_idx)
            self.l_ds = OfflineFeatureDataset(l_dir, subject_indices=l_idx, canonical=True, ram_cache=ram_cache, dist=dist,
                                              nan_stats={"mu": mu_l, "sigma": sigma_l})
            self.r_ds = OfflineFeatureDataset(r_dir, subject_indices=r_idx, canonical=True, ram_cache=ram_cache, dist=dist,
                                              nan_stats={"mu": mu_r, "sigma": sigma_r})
            self.nan_stats = {"l": {"mu": mu_l, "sigma": sigma_l},
                              "r": {"mu": mu_r, "sigma": sigma_r}}
            print(f"[paired-FC NAN] L mu/sigma on {len(l_idx)}, R mu/sigma on {len(r_idx)} (train-only)")
        self.nan_stats = getattr(self, "nan_stats", None)
        print(f"[paired-FC] {len(keep)} subjects, size={self.size}, n_out={self.n_out}, mode=sample_z", flush=True)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, k):
        lf, lc, _lcd, _lf2, _lFs, _label, _lsid, lhop = self.l_ds[k]
        rf, rc, _rcd, _rf2, _rFs, _label, _rsid, rhop = self.r_ds[k]
        return {
            "l_feats": lf, "l_center": lc, "l_hop": lhop,
            "r_feats": rf, "r_center": rc, "r_hop": rhop,
            "target": torch.from_numpy(self.fc_target[k]),
            "mean": torch.tensor(self.fc_mean[k, 0]),
            "std": torch.tensor(self.fc_std[k, 0]),
            "sid": self.sids[k],
        }
