import numpy as np
import torch
from torch.utils.data import Dataset
from dataset_offline import OfflineFeatureDataset


def _norm_sid(s):
    return str(s).split(".")[0].split("_")[0]


def _load_targets(prefix):
    return (np.load(prefix + "_targets_z.npy"),
            np.load(prefix + "_mean.npy"), np.load(prefix + "_std.npy"),
            [_norm_sid(s) for s in np.load(prefix + "_subject_ids.npy", allow_pickle=True)])


class PairedTfMRIDataset(Dataset):
    def __init__(self, lh_dir, rh_dir, l_pre, r_pre, subjects, ram_cache=True, dist="hop",
                 exclude_subtasks=None, nan=False, nan_stats=None, nan_subjects=None):
        l_tz, l_mean, l_std, l_sids = _load_targets(l_pre)
        r_tz, r_mean, r_std, r_sids = _load_targets(r_pre)
        l_pos = {s: i for i, s in enumerate(l_sids)}
        r_pos = {s: i for i, s in enumerate(r_sids)}
        off_l = [_norm_sid(s) for s in np.load(lh_dir + "/subject_ids.npy", allow_pickle=True)]
        off_r = [_norm_sid(s) for s in np.load(rh_dir + "/subject_ids.npy", allow_pickle=True)]
        ol = {}
        for i, s in enumerate(off_l):
            ol.setdefault(s, i)
        or_ = {}
        for i, s in enumerate(off_r):
            or_.setdefault(s, i)

        keep = [s for s in subjects if s in ol and s in or_ and s in l_pos and s in r_pos]
        self.sids = keep
        l_idx = [ol[s] for s in keep]; r_idx = [or_[s] for s in keep]
        lti = [l_pos[s] for s in keep]; rti = [r_pos[s] for s in keep]
        if exclude_subtasks:
            keep_idx = [i for i in range(l_tz.shape[1]) if i not in exclude_subtasks]
            l_tz = l_tz[:, keep_idx]
            l_mean = l_mean[:, keep_idx]; l_std = l_std[:, keep_idx]
            r_tz = r_tz[:, keep_idx]
            r_mean = r_mean[:, keep_idx]; r_std = r_std[:, keep_idx]
            print(f"[paired-tfmri] excluded subtasks {exclude_subtasks}, remaining {len(keep_idx)}", flush=True)
        self._dirs = (lh_dir, rh_dir)
        self._ram_cache = ram_cache
        self._dist = dist
        self._ol, self._or_ = ol, or_
        nan_l = nan_r = None
        if nan_stats is not None:
            nan_l, nan_r = nan_stats["l"], nan_stats["r"]
            print("[paired-tfmri NAN] using pre-computed train NAN stats", flush=True)
        elif nan:
            from dataset_offline import compute_nan_stats
            stat_sids = nan_subjects if nan_subjects is not None else keep
            l_sidx = [ol[s] for s in stat_sids if s in ol]
            r_sidx = [or_[s] for s in stat_sids if s in or_]
            l_data = np.load(lh_dir + "/data.npy", mmap_mode="r")
            l_cmap = np.load(lh_dir + "/canonical_mapping.npy")
            r_data = np.load(rh_dir + "/data.npy", mmap_mode="r")
            r_cmap = np.load(rh_dir + "/canonical_mapping.npy")
            mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, l_sidx)
            mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, r_sidx)
            nan_l = {"mu": mu_l, "sigma": sigma_l}
            nan_r = {"mu": mu_r, "sigma": sigma_r}
            self.nan_stats = {"l": nan_l, "r": nan_r}
            print(f"[paired-tfmri NAN] L mu/sigma on {len(l_sidx)}, R mu/sigma on {len(r_sidx)} (train-only)", flush=True)
        self.l_ds = OfflineFeatureDataset(lh_dir, subject_indices=l_idx, canonical=True,
                                          ram_cache=ram_cache, dist=dist, nan_stats=nan_l)
        self.r_ds = OfflineFeatureDataset(rh_dir, subject_indices=r_idx, canonical=True,
                                          ram_cache=ram_cache, dist=dist, nan_stats=nan_r)
        self.l_tz = torch.from_numpy(l_tz[lti].astype(np.float32))
        self.l_mean = torch.from_numpy(l_mean[lti].astype(np.float32))
        self.l_std = torch.from_numpy(l_std[lti].astype(np.float32))
        self.r_tz = torch.from_numpy(r_tz[rti].astype(np.float32))
        self.r_mean = torch.from_numpy(r_mean[rti].astype(np.float32))
        self.r_std = torch.from_numpy(r_std[rti].astype(np.float32))
        self.n_sub = l_tz.shape[1]; self.V = l_tz.shape[2]
        print(f"[paired-tfmri] {len(keep)} paired subjects, n_sub={self.n_sub} V={self.V}", flush=True)

    def refit_nan(self, train_sids):
        from dataset_offline import compute_nan_stats
        lh_dir, rh_dir = self._dirs
        l_sidx = [self._ol[s] for s in train_sids if s in self._ol]
        r_sidx = [self._or_[s] for s in train_sids if s in self._or_]
        l_data = np.load(lh_dir + "/data.npy", mmap_mode="r")
        l_cmap = np.load(lh_dir + "/canonical_mapping.npy")
        r_data = np.load(rh_dir + "/data.npy", mmap_mode="r")
        r_cmap = np.load(rh_dir + "/canonical_mapping.npy")
        mu_l, sigma_l = compute_nan_stats(l_data, l_cmap, l_sidx)
        mu_r, sigma_r = compute_nan_stats(r_data, r_cmap, r_sidx)
        self.l_ds = OfflineFeatureDataset(lh_dir, subject_indices=[self._ol[s] for s in self.sids],
                                          canonical=True, ram_cache=self._ram_cache, dist=self._dist,
                                          nan_stats={"mu": mu_l, "sigma": sigma_l})
        self.r_ds = OfflineFeatureDataset(rh_dir, subject_indices=[self._or_[s] for s in self.sids],
                                          canonical=True, ram_cache=self._ram_cache, dist=self._dist,
                                          nan_stats={"mu": mu_r, "sigma": sigma_r})
        self.nan_stats = {"l": {"mu": mu_l, "sigma": sigma_l}, "r": {"mu": mu_r, "sigma": sigma_r}}
        print(f"[paired-tfmri NAN] refit L on {len(l_sidx)}, R on {len(r_sidx)} (train-only)", flush=True)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, k):
        lf, lc, _lcd, _lf2, _lFs, _lab, _lsid, lhop = self.l_ds[k]
        rf, rc, _rcd, _rf2, _rFs, _rab, _rsid, rhop = self.r_ds[k]
        return {"lf": lf, "lc": lc, "lhop": lhop, "rf": rf, "rc": rc, "rhop": rhop,
                "l_tz": self.l_tz[k], "l_mean": self.l_mean[k], "l_std": self.l_std[k],
                "r_tz": self.r_tz[k], "r_mean": self.r_mean[k], "r_std": self.r_std[k],
                "idx": k}
