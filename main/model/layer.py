import math
from math import pi as PI
from tkinter import NONE
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch_scatter import scatter

# Settings
H_INIT_METHOD = "default"  # default, he
X_INIT_METHOD = "default"
E_INIT_METHOD = "default"
GLOBAL_INIT_MEHTOD = "default"
ACTIVATION = "silu"


def half_edge_to_full_edge(edge_index, edge_attr):
    """
    Change directed edge to undirected edge
    """
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return edge_index, edge_attr


def full_edge_to_half_edge(edge_index, edge_attr):  # [2, 2 * n_edge], [2 * n_edge, *]
    """
    Change undirected edge to directed edge (first-half)
    """
    n_edge = edge_attr.size(0)
    edge_index = edge_index[:, : n_edge // 2]
    edge_attr = edge_attr[: n_edge // 2, :] + edge_attr[n_edge // 2 :, :]
    return edge_index, edge_attr


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        dropout=0.0,
        layer_norm=False,
        activation="relu",
        last_activation="none",
        init_method="default",
        last_layer_xavier_small=False,
    ):
        super(MLP, self).__init__()
        assert len(dims) > 1  # more than two dims (in out)
        assert activation in [
            "none",
            "relu",
            "silu",
            "leaky_relu",
            "softplus",
        ]
        assert last_activation in [
            "none",
            "relu",
            "silu",
            "leaky_relu",
            "softplus",
            "sigmoid",
            "tanh",
        ]
        assert init_method in [
            "default",
            "xavier",
            "he",
        ]

        n_layer = len(dims)
        layers = []
        for i in range(n_layer - 1):  # 0, 1, ..., n_layer - 2
            in_dim, out_dim = dims[i], dims[i + 1]

            # Parameter initialization
            if init_method == "default":
                init_func = None  # Use PyTorch default initialization
            elif init_method == "xavier":
                init_func = nn.init.xavier_uniform_
            elif init_method == "he":
                init_func = nn.init.kaiming_uniform_
            else:
                raise ValueError
            bias_init_func = nn.init.zeros_

            # Linear layer
            linear_layer = nn.Linear(in_dim, out_dim)
            if init_func is not None:
                init_func(linear_layer.weight)
                bias_init_func(linear_layer.bias)
            layers.append(linear_layer)

            if i < n_layer - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

                if activation == "none":
                    pass
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif activation == "softplus":
                    layers.append(nn.Softplus())
                else:
                    raise NotImplementedError

        if last_layer_xavier_small:
            torch.nn.init.xavier_uniform_(layers[-1].weight, gain=0.001)

        if last_activation == "none":
            pass
        elif last_activation == "relu":
            layers.append(nn.ReLU())
        elif last_activation == "silu":
            layers.append(nn.SiLU())
        elif last_activation == "leaky_relu":
            layers.append(nn.LeakyReLU())
        elif last_activation == "softplus":
            layers.append(nn.Softplus())
        elif last_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise NotImplementedError

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SoftEncoding(nn.Module):
    def __init__(self, n_step, min_val=0.0, max_val=10.0, gamma=10.0):
        super().__init__()

        self.n_step = n_step
        self.min_val = min_val
        self.max_val = max_val
        self.gamma = gamma

    def forward(self, r):
        c = torch.Tensor(
            [
                self.min_val * (self.n_step - i - 1) / (self.n_step - 1)
                + self.max_val * i / (self.n_step - 1)
                for i in range(self.n_step)
            ]
        )  # [n_step]
        c = c.unsqueeze(0).to(r.device)  # [1, n_step]
        r = r.unsqueeze(1).repeat(1, self.n_step)  # [n_edge, n_step]
        r = torch.exp(-self.gamma * torch.pow(r - c, 2))
        return r


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        device = x.device
        return self.encoding[x, :].to(device)


class EGNNInteractionLayer(nn.Module):  # EGNN
    def __init__(
        self,
        h1_dim: int,
        e11_dim: int,
        h2_dim: int,
        e22_dim: int,
        e12_dim: int,
        hid_dim: int,
        dropout: float,
        n_step: int,  # number of distance soft one-hot steps (100 ~ 300)
        dist_min_max_gamma: Tuple[
            float, float, float
        ],  # distance minimum and maximum ([2, 5])
        position_reduction: str,
        message_reduction: str,
        update_edge: bool,
        use_tanh: bool,
        m_attention: str,
        m_attention_coef=None,
    ):
        """
        followings are done:
        ligand atom is indexed as 2
        only x2 position is updated not x1!
        """
        super().__init__()
        assert position_reduction in ["sum", "mean"]
        assert message_reduction in ["sum", "mean", "max", "mul"]

        self.h1_dim, self.h2_dim = h1_dim, h2_dim
        self.e11_dim, self.e22_dim = e11_dim, e22_dim
        self.e12_dim = e12_dim

        self.hid_dim = hid_dim
        self.dist_min_max_gamma = dist_min_max_gamma
        self.position_reduction = position_reduction
        self.message_reduction = message_reduction
        self.update_edge = update_edge
        self.use_tanh = use_tanh
        self.m_attention = m_attention
        self.m_attention_coef = m_attention_coef

        if self.dist_min_max_gamma is None:
            self.embd_distance = nn.Linear(1, n_step)
        else:
            self.embd_distance = SoftEncoding(n_step, *dist_min_max_gamma)

        m_init_method = E_INIT_METHOD
        m_activation = ACTIVATION
        fc_m11_dims = [h1_dim * 2 + e11_dim + n_step, hid_dim, hid_dim]
        self.fc_m11 = MLP(
            fc_m11_dims,
            dropout=dropout,
            layer_norm=True,
            activation=m_activation,
            last_activation=m_activation,
            init_method=m_init_method,
        )
        if self.m_attention == "mlp":
            fc_m11_att_dims = [hid_dim, 1]
            self.fc_m11_att = MLP(
                fc_m11_att_dims,
                dropout=dropout,
                layer_norm=True,
                activation="none",
                last_activation="sigmoid",
                init_method=m_init_method,
            )
        fc_m22_dims = [h2_dim * 2 + e22_dim + n_step, hid_dim, hid_dim]
        self.fc_m22 = MLP(
            fc_m22_dims,
            dropout=dropout,
            layer_norm=True,
            activation=m_activation,
            last_activation=m_activation,
            init_method=m_init_method,
        )
        if self.m_attention == "mlp":
            fc_m22_att_dims = [hid_dim, 1]
            self.fc_m22_att = MLP(
                fc_m22_att_dims,
                dropout=dropout,
                layer_norm=True,
                activation="none",
                last_activation="sigmoid",
                init_method=m_init_method,
            )
        fc_m12_dims = [h1_dim + h2_dim + e12_dim + n_step, hid_dim, hid_dim]
        self.fc_m12 = MLP(
            fc_m12_dims,
            dropout=dropout,
            layer_norm=True,
            activation=m_activation,
            last_activation=m_activation,
            init_method=m_init_method,
        )
        if self.m_attention == "mlp":
            fc_m12_att_dims = [hid_dim, 1]
            self.fc_m12_att = MLP(
                fc_m12_att_dims,
                dropout=dropout,
                layer_norm=True,
                activation="none",
                last_activation="sigmoid",
                init_method=m_init_method,
            )

        if update_edge:
            e_init_method = E_INIT_METHOD
            e_activation = ACTIVATION
            fc_e11_dims = [hid_dim, e11_dim]
            self.fc_e11 = MLP(
                fc_e11_dims,
                dropout=dropout,
                layer_norm=True,
                activation=e_activation,
                last_activation=e_activation,
                init_method=e_init_method,
            )
            fc_e22_dims = [hid_dim, e11_dim]
            self.fc_e22 = MLP(
                fc_e22_dims,
                dropout=dropout,
                layer_norm=True,
                activation=e_activation,
                last_activation=e_activation,
                init_method=e_init_method,
            )
            fc_e12_dims = [hid_dim, e12_dim]
            self.fc_e12 = MLP(
                fc_e12_dims,
                dropout=dropout,
                layer_norm=True,
                activation=e_activation,
                last_activation=e_activation,
                init_method=e_init_method,
            )

        x_init_method = X_INIT_METHOD
        x_activation = ACTIVATION
        if use_tanh:
            t = "tanh"
        else:
            t = "none"
        fc_x22_dims = [hid_dim, hid_dim, 1]
        self.fc_x22 = MLP(
            fc_x22_dims,
            dropout=dropout,
            layer_norm=False,
            activation=x_activation,
            last_activation=t,
            init_method=x_init_method,
            last_layer_xavier_small=True,
        )
        fc_x12_dims = [hid_dim, hid_dim, 1]
        self.fc_x12 = MLP(
            fc_x12_dims,
            dropout=dropout,
            layer_norm=False,
            activation=x_activation,
            last_activation=t,
            init_method=x_init_method,
            last_layer_xavier_small=True,
        )

        h_init_method = H_INIT_METHOD
        h_activation = ACTIVATION
        fc_h1_dims = [h1_dim + hid_dim + hid_dim + h1_dim + h2_dim, hid_dim, h1_dim]
        self.fc_h1 = MLP(
            fc_h1_dims,
            dropout=dropout,
            layer_norm=False,
            activation=h_activation,
            last_activation=h_activation,
            init_method=h_init_method,
        )
        fc_h2_dims = [h2_dim + hid_dim + hid_dim + h1_dim + h2_dim, hid_dim, h2_dim]
        self.fc_h2 = MLP(
            fc_h2_dims,
            dropout=dropout,
            layer_norm=False,
            activation=h_activation,
            last_activation=h_activation,
            init_method=h_init_method,
        )

        # global updataers
        self.fc_global1 = MLP(
            [h1_dim, h1_dim, h1_dim],
            dropout=dropout,
            layer_norm=True,
            activation=h_activation,
            last_activation=h_activation,
            init_method=GLOBAL_INIT_MEHTOD,
        )
        self.fc_global2 = MLP(
            [h2_dim, h2_dim, h2_dim],
            dropout=dropout,
            layer_norm=True,
            activation=h_activation,
            last_activation=h_activation,
            init_method=GLOBAL_INIT_MEHTOD,
        )

    def forward(
        self,
        h1,
        x1,
        e11_index,
        e11,
        batch1,
        h2,
        x2,
        e22_index,
        e22,
        batch2,
        e12_index,
        e12,
    ):
        # make message
        m11 = self.make_intra_m(h1, x1, e11_index, e11, "11")
        m22 = self.make_intra_m(h2, x2, e22_index, e22, "22")
        m12 = self.make_inter_m(h1, x1, h2, x2, e12_index, e12)

        # update position (only 2)
        x2 = self.update_x2(x1, x2, h2, e22_index, batch2, e12_index, m22, m12)

        # update node feature
        h1, h2 = self.update_h1_h2(
            h1,
            e11_index,
            batch1,
            h2,
            e22_index,
            batch2,
            e12_index,
            m11,
            m22,
            m12,
        )

        # update edge with nn
        if self.update_edge:
            e11 = e11 + self.fc_e11(m11)
            e22 = e22 + self.fc_e22(m22)
            e12 = e12 + self.fc_e12(m12)
            return h1, x1, e11, h2, x2, e22, e12
        else:
            return h1, x1, m11, h2, x2, m22, m12

    def make_intra_m(
        self, h, x, e_index, e, layer_idx
    ):  # [N_node, hid_dim] # [2, N_edge]
        h_i, h_j = h[e_index[1]], h[e_index[0]]  #  [N_edge, hid_dim]
        x_i, x_j = x[e_index[1]], x[e_index[0]]
        r = (x_i - x_j).norm(dim=1)  # distance, not distance square
        if self.dist_min_max_gamma is None:
            r_emb = self.embd_distance(r.unsqueeze(1))  # [n_edge, n_step]
        else:
            r_emb = self.embd_distance(r)  # [n_edge, n_step]

        m = torch.cat([h_i, h_j, r_emb, e], dim=1)  # [n_edge, *]
        if layer_idx == "11":
            m = self.fc_m11(m)
            if self.m_attention == "none":
                m = m
            elif self.m_attention == "mlp":
                m = m * self.fc_m11_att(m)
            elif self.m_attention == "cosine":
                m = m * self.cosine_filter_by_distance(r, self.m_attention_coef)
            elif self.m_attention == "sigmoid":
                m = m * self.sigmoid_filter_by_distance(r, self.m_attention_coef)
            else:
                raise NotImplementedError
        elif layer_idx == "22":
            m = self.fc_m22(m)
            if self.m_attention == "none":
                m = m
            elif self.m_attention == "mlp":
                m = m * self.fc_m22_att(m)
            elif self.m_attention == "cosine":
                m = m * self.cosine_filter_by_distance(r, self.m_attention_coef)
            elif self.m_attention == "sigmoid":
                m = m * self.sigmoid_filter_by_distance(r, self.m_attention_coef)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return m

    def make_inter_m(
        self, h1, x1, h2, x2, e12_index, e12
    ):  # [N_node, hid_dim] # [2, N_edge]
        h_i, h_j = h1[e12_index[0]], h2[e12_index[1]]
        x_i, x_j = x1[e12_index[0]], x2[e12_index[1]]
        r = (x_i - x_j).norm(dim=1)  # distance, not distance square
        if self.dist_min_max_gamma is None:
            r_emb = self.embd_distance(r.unsqueeze(1))  # [n_edge, n_step]
        else:
            r_emb = self.embd_distance(r)  # [n_edge, n_step]
        m = torch.cat([h_i, h_j, r_emb, e12], dim=1)  # [n_edge, *]
        m = self.fc_m12(m)
        if self.m_attention == "none":
            m = m
        elif self.m_attention == "mlp":
            m = m * self.fc_m12_att(m)
        elif self.m_attention == "cosine":
            m = m * self.cosine_filter_by_distance(r, self.m_attention_coef)
        elif self.m_attention == "sigmoid":
            m = m * self.sigmoid_filter_by_distance(r, self.m_attention_coef)
        else:
            raise NotImplementedError
        return m

    def update_x2(self, x1, x2, h2, e22_index, batch2, e12_index, m22, m12):
        v22 = x2[e22_index[0]] - x2[e22_index[1]]
        v12 = x1[e12_index[0]] - x2[e12_index[1]]
        v22 = v22 / (v22.norm(dim=1, keepdim=True) + 1e-10)
        v12 = v12 / (v12.norm(dim=1, keepdim=True) + 1e-10)
        v22 = v22 * self.fc_x22(m22)
        v12 = v12 * self.fc_x12(m12)

        # intra update
        x2_intra_update = scatter(
            v22,
            e22_index[1],
            dim=0,
            dim_size=x2.size(0),
            reduce=self.position_reduction,
        )

        # inter update
        x2_inter_update = scatter(
            v12,
            e12_index[1],
            dim=0,
            dim_size=x2.size(0),
            reduce=self.position_reduction,
        )

        x2 = x2 + x2_intra_update + x2_inter_update
        return x2

    def update_h1_h2(
        self,
        h1,
        e11_index,
        batch1,
        h2,
        e22_index,
        batch2,
        e12_index,
        m11,
        m22,
        m12,
    ):
        m11_aggr = scatter(
            m11, e11_index[1], dim=0, dim_size=h1.size(0), reduce=self.message_reduction
        )
        m22_aggr = scatter(
            m22, e22_index[1], dim=0, dim_size=h2.size(0), reduce=self.message_reduction
        )
        m12_1_aggr = scatter(
            m12, e12_index[0], dim=0, dim_size=h1.size(0), reduce=self.message_reduction
        )
        m12_2_aggr = scatter(
            m12, e12_index[1], dim=0, dim_size=h2.size(0), reduce=self.message_reduction
        )

        h1_aggr = scatter(h1, batch1, dim=0, dim_size=batch1.max() + 1, reduce="sum")
        z1 = self.fc_global1(h1_aggr)
        h2_aggr = scatter(h2, batch2, dim=0, dim_size=batch2.max() + 1, reduce="sum")
        z2 = self.fc_global2(h2_aggr)

        # update h
        h1 = h1 + self.fc_h1(
            torch.cat([h1, m11_aggr, m12_1_aggr, z1[batch1], z2[batch1]], dim=1)
        )
        h2 = h2 + self.fc_h2(
            torch.cat([h2, m22_aggr, m12_2_aggr, z1[batch2], z2[batch2]], dim=1)
        )
        # h1 = h1 + self.fc_h1(torch.cat([h1, m11_aggr, m12_1_aggr], dim=1))
        # h2 = h2 + self.fc_h2(torch.cat([h2, m22_aggr, m12_2_aggr], dim=1))
        return h1, h2


class IGNNLayer(nn.Module):
    def __init__(
        self,
        h_dim,
        e_dim,
        dropout=0.0,
        layer_norm=True,
        activation="silu",
        init_method="he",
        message_reduction="mean",
        update_edge=True,
    ):
        super(IGNNLayer, self).__init__()

        self.h_dim = h_dim
        self.e_dim = e_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.activation = activation
        self.init_method = init_method
        self.message_reduction = message_reduction
        self.update_edge = update_edge

        self.dist_emb = nn.Linear(1, h_dim)

        fc1_dims = [h_dim * 3 + e_dim, h_dim, h_dim]
        self.fc1 = MLP(
            fc1_dims,
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=activation,
            init_method=init_method,
        )

        # fc2_dims = [h_dim, h_dim, 1]
        # self.fc2 = MLP(
        #    fc2_dims,
        #    dropout=dropout,
        #    layer_norm=layer_norm,
        #    activation=activation,
        #    last_activation=activation,
        #    init_method=init_method,
        # )

        fc3_dims = [h_dim * 2, h_dim, h_dim]
        self.fc3 = MLP(
            fc3_dims,
            dropout=dropout,
            layer_norm=layer_norm,
            activation=activation,
            last_activation=activation,
            init_method=init_method,
        )

        if update_edge:
            fc4_dims = [e_dim * 2, e_dim, e_dim]
            self.fc4 = MLP(
                fc4_dims,
                dropout=dropout,
                layer_norm=layer_norm,
                activation=activation,
                last_activation=activation,
                init_method=init_method,
            )

    def forward(self, h, x, e, e_index):
        h_i, h_j = h[e_index[0]], h[e_index[1]]
        x_i, x_j = x[e_index[0]], x[e_index[1]]
        d = (x_i - x_j).norm(dim=-1, keepdim=True)  # distance, not distance square
        d_emb = self.dist_emb(d)
        h_cat = torch.cat([h_i, h_j, d_emb, e], dim=1)  # [n_edge, *]
        msg = self.fc1(h_cat)
        h_aggr = scatter(
            msg, e_index[1], dim=0, dim_size=h.size(0), reduce=self.message_reduction
        )
        h_update = h + self.fc3(torch.cat([h, h_aggr], dim=-1))
        if self.update_edge:
            e_update = e + self.fc4(torch.cat([e, msg], dim=-1))
        else:
            e_update = e
        return h_update, x, e_update
