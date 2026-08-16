import numpy as np
import torch
from torch.utils.data import Dataset

N_PATCHES = 2048
PATCH_SIZE = 64


def compute_nan_stats(data_mmap, canon_map, train_indices):
    acc = []
    for i in train_indices:
        f = np.asarray(data_mmap[i], dtype=np.float32)
        inv = np.argsort(canon_map[i])
        f = f[:, inv, :]
        acc.append(f.mean(axis=-1))
    acc = np.stack(acc, 0)
    mu = acc.mean(0)
    sigma = np.maximum(acc.std(0), 1e-3)
    return mu.astype(np.float32), sigma.astype(np.float32)


class OfflineFeatureDataset(Dataset):

    def __init__(self, feat_dir, subject_indices=None, canonical=True,
                 nan_stats=None, phenotype=None, ram_cache=False, dist="hop"):
        from pathlib import Path
        self.dir = Path(feat_dir)
        self.dist = dist
        self.data = np.load(self.dir / "data.npy", mmap_mode="r")
        self.centers = np.load(self.dir / "centers.npy", mmap_mode="r")
        self.coords = np.load(self.dir / "coordinates.npy", mmap_mode="r")
        self.faces = np.load(self.dir / "faces.npy", mmap_mode="r")
        self.canon_map = np.load(self.dir / "canonical_mapping.npy")
        _dist_name = "geodesic.npy" if dist == "geodesic" else "hop.npy"
        hop_path = self.dir / _dist_name
        self.hop = np.load(hop_path, mmap_mode="r") if hop_path.exists() else None
        self.sids = [str(s) for s in np.load(self.dir / "subject_ids.npy", allow_pickle=True)]
        self.canonical = canonical
        self.phenotype = phenotype

        if subject_indices is None:
            subject_indices = list(range(self.data.shape[0]))
        self.indices = list(subject_indices)

        self.mu = self.sigma = None
        if nan_stats is not None:
            self.mu = nan_stats["mu"]
            self.sigma = nan_stats["sigma"]

        self._ram = None
        if ram_cache:
            self._build_ram_cache()

    def _hop_arr(self, i):
        if self.hop is None:
            return None
        if self.dist == "geodesic":
            return np.asarray(self.hop[i], dtype=np.float32)
        return np.asarray(self.hop[i])

    def _build_ram_cache(self):
        data, centers, coords, faces, hops, sids = [], [], [], [], [], []
        for k, i in enumerate(self.indices):
            f = np.asarray(self.data[i], dtype=np.float32)
            c = np.asarray(self.centers[i], dtype=np.float16)
            cd = np.asarray(self.coords[i], dtype=np.float16)
            fa = np.asarray(self.faces[i]).astype(np.float32)
            h = np.asarray(self.hop[i]) if self.hop is not None else None
            if self.canonical:
                inv = np.argsort(self.canon_map[i])
                f = f[:, inv, :]; c = c[inv]; cd = cd[inv]; fa = fa[inv]
                if h is not None:
                    h = h[inv][:, inv]
            if self.mu is not None:
                f = (f - self.mu[:, :, None]) / self.sigma[:, :, None]
            data.append(np.ascontiguousarray(f.astype(np.float16))); centers.append(np.ascontiguousarray(c))
            coords.append(np.ascontiguousarray(cd)); faces.append(np.ascontiguousarray(fa))
            if h is not None:
                hops.append(np.ascontiguousarray(h))
            sids.append(self.sids[i])
            if (k + 1) % 1000 == 0:
                print(f"  [ram_cache] {k+1}/{len(self.indices)} gathered", flush=True)
        self._ram = {"data": data, "centers": centers, "coords": coords,
                     "faces": faces, "hops": hops, "sids": sids}
        print(f"[ram_cache] {len(data)} samples cached in RAM (~{sum(x.nbytes for x in data)/1e9:.1f}GB data)", flush=True)

    def __getitem__(self, k):
        i = self.indices[k]
        sid = self.sids[i]
        label = self.phenotype[sid] if self.phenotype else 0

        if self._ram is not None:
            f = self._ram["data"][k].astype(np.float32)
            c = self._ram["centers"][k].astype(np.float32)
            cd = self._ram["coords"][k].astype(np.float32)
            fa = self._ram["faces"][k]
            hop = self._ram["hops"][k]
            if hop is not None and self.dist == "geodesic":
                hop = hop.astype(np.float32)
        else:
            f = np.asarray(self.data[i], dtype=np.float32)
            c = np.asarray(self.centers[i], dtype=np.float32)
            cd = np.asarray(self.coords[i], dtype=np.float32)
            fa = np.asarray(self.faces[i]).astype(np.float32)
            if self.canonical:
                inv = np.argsort(self.canon_map[i])
                f = f[:, inv, :]; c = c[inv]; cd = cd[inv]; fa = fa[inv]
            if self.mu is not None:
                f = (f - self.mu[:, :, None]) / self.sigma[:, :, None]
            if self.hop is None:
                raise RuntimeError(f"hop.npy missing in {self.dir}")
            hop = self._hop_arr(i)
            if self.canonical:
                inv = np.argsort(self.canon_map[i])
                hop = hop[inv][:, inv]

        Fs = np.array(f.shape[1] * f.shape[2], dtype=np.int64)
        return (torch.from_numpy(np.ascontiguousarray(f)),
                torch.from_numpy(np.ascontiguousarray(c)),
                torch.from_numpy(np.ascontiguousarray(cd)),
                torch.from_numpy(np.ascontiguousarray(fa)),
                torch.from_numpy(np.asarray(Fs)),
                label, sid,
                torch.from_numpy(np.ascontiguousarray(hop)))

    def __len__(self):
        return len(self.indices)
