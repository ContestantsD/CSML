import torch


def _pairwise_nn_sq(xyz1, xyz2):
    diff = xyz1[:, :, None, :] - xyz2[:, None, :, :]
    d2 = (diff ** 2).sum(-1)
    return d2.min(2)[0], d2.min(1)[0]


class ChamferDistanceL1(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = _pairwise_nn_sq(xyz1, xyz2)
        return (torch.mean(torch.sqrt(dist1 + 1e-12)) +
                torch.mean(torch.sqrt(dist2 + 1e-12))) / 2


class ChamferDistanceL2(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = _pairwise_nn_sq(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)


class ChamferDistanceL2_split(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        dist1, dist2 = _pairwise_nn_sq(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)
