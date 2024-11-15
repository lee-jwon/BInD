import math
import os
import sys
from math import pi as PI

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_min

from .model.layer import full_edge_to_half_edge, half_edge_to_full_edge


class BondDistanceGuidance(nn.Module):
    def __init__(self, distance_min_max=[1.2, 1.9], epsilon1=0.1, epsilon2=0.1):
        super().__init__()

        self.distance_min, self.distance_max = distance_min_max
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def forward(self, x, e_type, e_index):
        e_mask = e_type == 0
        d = torch.norm(x[e_index[0]] - x[e_index[1]], dim=1)
        left = (d - self.distance_max).clip(min=0)
        right = (self.distance_min - d).clip(min=0)
        drift = self.epsilon1 * left + self.epsilon2 * right
        drift[e_mask] = 0
        return drift.sum()


class TwoHopDistanceGuidance(nn.Module):
    def __init__(
        self, distance_min_max=[1.2, 1.9], angle_min=90, epsilon1=0.1, epsilon2=0.1
    ):
        super().__init__()

        self.distance_min, self.distance_max = distance_min_max
        self.angle_min = angle_min

        self.distance_min = self.distance_min * 2 * math.sin(PI * angle_min / 360)
        self.distance_max = self.distance_max * 2

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def forward(self, x, e_type, e_index):
        e_mask = e_type != 0
        e_mask_idx = e_mask.nonzero().squeeze(-1)
        e_full_idx, _ = half_edge_to_full_edge(
            e_index[:, e_mask_idx], e_type[e_mask_idx]
        )
        adj = to_dense_adj(e_full_idx, max_num_nodes=x.shape[0])[0, :, :]
        adj2 = adj @ adj
        trace_zero = (1 - torch.eye(x.shape[0])).to(adj2.device)
        adj2 = adj2 * trace_zero  # remove self-loop
        e_two_hop_index = adj2.nonzero().contiguous().T
        d = torch.norm(x[e_two_hop_index[0]] - x[e_two_hop_index[1]], dim=1)
        right = (self.distance_min - d).clip(min=0)
        left = (d - self.distance_max).clip(min=0)
        drift = self.epsilon1 * left + self.epsilon2 * right
        return drift.sum()


class InterBondDistanceGuidance(nn.Module):
    def __init__(self, distance_min_max=[2.0, 8.0], epsilon1=0.1, epsilon2=0.1):
        super().__init__()

        self.distance_min, self.distance_max = distance_min_max
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def forward(self, x1, x2, e12_type, e12_index):
        e12_mask = e12_type == 0
        d = torch.norm(x1[e12_index[0]] - x2[e12_index[1]], dim=1)
        right = (self.distance_min - d).clip(min=0)
        left = (d - self.distance_max).clip(min=0)
        drift = self.epsilon1 * left + self.epsilon2 * right
        drift[e12_mask] = 0
        return drift.sum()


class SeparatedInterBondDistanceGuidance(nn.Module):
    def __init__(self, epsilon1=0.1, epsilon2=0.1):
        super().__init__()

        self.n_types = 6
        """self.dmin_maxs = [
            [2.0, 5.5], # SB
            [2.0, 5.5], # SB
            [2.0, 4.1], # HB 
            [2.0, 4.1], # HB 
            [2.0, 4.0], # HI
            [2.0, 5.5], # PP
        ]"""
        # min max from data distribution (reference)
        """self.dmin_maxs = [
            [2.8, 7.5], # SB
            [2.0, 5.5], # SB
            [2.4, 4.1], # HB 
            [2.4, 4.1], # HB 
            [3.0, 4.0], # HI
            [3.0, 7.0], # PP
        ]"""
        self.dmin_maxs = [
            [2.8, 7.5],  # SB
            [2.8, 7.5],  # SB
            [2.4, 4.1],  # HB
            [2.4, 4.1],  # HB
            [2.0, 4.0],  # HI
            [3.0, 7.0],  # PP
        ]

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    def forward(self, x1, x2, e12_type, e12_index):
        drift_merged = torch.zeros_like(e12_type).float()
        for i in range(self.n_types):
            dmin, dmax = self.dmin_maxs[i]
            is_inter = e12_type == (i + 1)
            if is_inter.sum() == 0:
                continue
            e12_index_sparse = e12_index.T[is_inter].T
            d_sparse = torch.norm(
                x1[e12_index_sparse[0]] - x2[e12_index_sparse[1]], dim=1
            )
            right = (dmin - d_sparse).clip(min=0)
            left = (d_sparse - dmax).clip(min=0)
            drift = self.epsilon1 * left + self.epsilon2 * right
            drift_merged[is_inter] = drift
        return drift_merged.sum()


class StericClashGuidance(nn.Module):
    def __init__(self, distance_min=0.5, epsilon=0.1, mode="every"):  # nearest, every
        super().__init__()

        self.distance_min = distance_min
        self.epsilon = epsilon
        self.mode = mode

        assert self.mode in ["nearest", "every"]

    def forward(self, x1, x2, e12_index):
        d = torch.norm(x1[e12_index[0]] - x2[e12_index[1]], dim=1)
        if self.mode == "nearest":
            out = x1.new_zeros((x1.shape[0],))
            nearest_d, _ = scatter_min(d, e12_index[1], out=out)
            drift = (self.distance_min - nearest_d).clip(min=0)
        elif self.mode == "every":
            drift = (self.distance_min - d).clip(min=0)
        return drift.sum() * self.epsilon


class BondAngleGuidance(nn.Module):  # TODO: too slow
    def __init__(self, scale):
        super().__init__()

        self.angle_min = 100.0  # degree
        self.scale = scale

    def calc_angle(self, point1, point2, point3):
        v1 = point1 - point2
        v2 = point3 - point2
        dotp = torch.dot(v1, v2)
        normp = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
        cos_theta = dotp / normp
        angle_rad = torch.arccos(cos_theta.clip(min=-1.0, max=1.0))
        angle_degree = torch.rad2deg(angle_rad)
        return angle_degree

    def forward(self, x, e_type, e_index):
        e_mask = e_type != 0
        e_mask_idx = e_mask.nonzero().squeeze(-1)
        e_full_idx, e_full_type = half_edge_to_full_edge(
            e_index[:, e_mask_idx], e_type[e_mask_idx]
        )
        num_half_edge = e_mask_idx.shape[0]
        angle_index, angle_drift = [], []
        for i in range(num_half_edge):
            idx1, idx2 = e_full_idx[:, i]
            x1, x2 = x[idx1], x[idx2]

            # 1. x1 is center
            idx3s = e_full_idx[:, (e_full_idx[0] == idx1).nonzero()][1].squeeze()
            idx3s = idx3s[idx3s != idx2]
            for idx3 in idx3s:
                if (idx2, idx1, idx3) in angle_index:
                    continue
                # print(idx1, idx2, idx3)
                x3 = x[idx3]
                angle = self.calc_angle(x2, x1, x3)
                angle_index += [(idx2, idx1, idx3), (idx3, idx1, idx2)]
                angle_drift += [self.scale * (self.angle_min - angle).clip(min=0)]

            # 2. x2 is center (is necessary?)
            idx3s = e_full_idx[:, (e_full_idx[0] == idx2).nonzero()][1].squeeze()
            idx3s = idx3s[idx3s != idx1]
            for idx3 in idx3s:
                if (idx1, idx2, idx3) in angle_index:
                    continue
                # print(idx1, idx2, idx3)
                x3 = x[idx3]
                angle = self.calc_angle(x1, x2, x3)
                angle_index += [(idx1, idx2, idx3), (idx3, idx2, idx1)]
                angle_drift += [self.scale * (self.angle_min - angle).clip(min=0)]

        if len(angle_drift) == 0:
            return 0.0
        angle_drift = torch.stack(angle_drift)
        return angle_drift.sum()


if __name__ == "__main__":
    pass
