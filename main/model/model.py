import math
from copy import deepcopy
from itertools import dropwhile
from math import pi as PI

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch_scatter import scatter

from main.model.function import *
from main.model.layer import *


class GenDiff(nn.Module):
    def __init__(self, confs):
        super().__init__()

        self.all_confs = confs
        n_timestep = confs["n_timestep"]
        self.confs = confs["model"]
        confs = confs["model"]

        node_embd_dim = confs["node_embd_dim"]
        edge_embd_dim = confs["edge_embd_dim"]
        hid_dim = confs["hid_dim"]
        self.node_embd_dim, self.edge_embd_dim, self.hid_dim = (
            node_embd_dim,
            edge_embd_dim,
            hid_dim,
        )
        self.n_timestep = n_timestep

        # timestep embedder
        self.pe_timestep = PositionalEncoding(confs["timestep_embd_dim"], n_timestep)
        self.embd_timestep = MLP(
            [
                confs["timestep_embd_dim"],
                confs["timestep_embd_dim"] * 4,
                confs["timestep_embd_dim"],
            ],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            init_method="he",
        )

        # embedding layers
        self.embd_rec_h = nn.Linear(confs["rec_h_dim"], node_embd_dim, bias=False)
        self.embd_rec_e = nn.Embedding(confs["rec_e_dim"], edge_embd_dim)
        self.embd_lig_h = nn.Embedding(confs["lig_h_dim"], node_embd_dim)
        self.embd_lig_e = nn.Embedding(confs["lig_e_dim"], edge_embd_dim)
        self.embd_inter_e = nn.Embedding(confs["inter_e_dim"], edge_embd_dim)

        # timestep merging layers
        self.merge_lig_h_timestep = MLP(
            [node_embd_dim + confs["timestep_embd_dim"], node_embd_dim, node_embd_dim],
            layer_norm=True,
            activation="silu",
            init_method="he",
        )
        self.merge_lig_e_timestep = MLP(
            [edge_embd_dim + confs["timestep_embd_dim"], edge_embd_dim, edge_embd_dim],
            layer_norm=True,
            activation="silu",
            init_method="he",
        )
        self.merge_inter_e_timestep = MLP(
            [edge_embd_dim + confs["timestep_embd_dim"], edge_embd_dim, edge_embd_dim],
            layer_norm=True,
            activation="silu",
            init_method="he",
        )

        if confs["dist_min"] != None:
            dist_min_max_gamma = [confs["dist_min"], confs["dist_max"], confs["gamma"]]
        else:
            dist_min_max_gamma = None

        # inter layers
        inter_layers = []
        for _ in range(confs["n_layer"]):
            inter_layers.append(
                EGNNInteractionLayer(
                    h1_dim=node_embd_dim,
                    e11_dim=edge_embd_dim,
                    h2_dim=node_embd_dim,
                    e22_dim=edge_embd_dim,
                    e12_dim=edge_embd_dim,
                    hid_dim=hid_dim,
                    dropout=0.0,
                    n_step=confs["n_step"],
                    dist_min_max_gamma=dist_min_max_gamma,
                    position_reduction=confs["position_reduction"],
                    message_reduction=confs["message_reduction"],
                    update_edge=True,
                    use_tanh=confs["use_tanh"],
                    m_attention=confs["message_attention_mode"],
                    m_attention_coef=confs["message_attention_coef"],
                )
            )
        self.inter_layers = nn.ModuleList(inter_layers)

        # readout feature layers (for direct prediction)
        self.fc_readout_lig_h = MLP(
            [
                confs["node_embd_dim"],
                confs["node_embd_dim"],
                confs["lig_h_dim"],
                confs["lig_h_dim"],
            ],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            last_activation="none",
            init_method="he",
        )

        # readout
        self.fc_readout_rec_h_binary = MLP(
            [confs["node_embd_dim"], confs["node_embd_dim"], confs["lig_h_dim"], 2],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            last_activation="none",
            init_method="he",
        )

        # readout layers
        self.fc_readout_lig_e = MLP(
            [
                confs["edge_embd_dim"],
                confs["edge_embd_dim"],
                confs["lig_e_dim"],
                confs["lig_e_dim"],
            ],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            last_activation="none",
            init_method="he",
        )
        self.fc_readout_inter_e = MLP(
            [
                confs["edge_embd_dim"],
                confs["edge_embd_dim"],
                confs["inter_e_dim"],
                confs["inter_e_dim"],
            ],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            last_activation="none",
            init_method="he",
        )

        self.fc_readout_inter_e_active = MLP(
            [
                confs["node_embd_dim"] + confs["node_embd_dim"],
                confs["node_embd_dim"],
                confs["inter_e_dim"],
                confs["inter_e_dim"],
            ],
            dropout=confs["dropout"],
            layer_norm=confs["layer_norm"],
            activation="silu",
            last_activation="none",
            init_method="he",
        )

    def propagate(
        self,
        rec_h,
        rec_x,
        rec_e_index,
        rec_e_type,
        rec_batch,
        lig_h_type,
        lig_x,
        lig_e_index,
        lig_e_type,
        lig_batch,
        timestep,
        inter_e_index=None,
        inter_e_type=None,
    ):
        # get
        device = rec_h.device

        # embd timestep
        timestep_embedded = self.pe_timestep(timestep).to(device)
        timestep_embedded = self.embd_timestep(timestep_embedded)

        # embd nodes
        rec_h = self.embd_rec_h(rec_h)
        lig_h = self.embd_lig_h(lig_h_type)

        # embd edges
        if rec_e_type != None:
            rec_e = self.embd_rec_e(rec_e_type)
        else:
            rec_e = (
                torch.zeros(rec_e_index.size(1), self.edge_embd_dim).detach().to(device)
            )
        lig_e = self.embd_lig_e(lig_e_type)

        # embd inter edges
        if inter_e_index != None:
            inter_e = self.embd_inter_e(inter_e_type)
        else:
            inter_e_index = self.make_rec_to_lig_edge_index(rec_batch, lig_batch)
            inter_e = (
                torch.zeros(inter_e_index.size(1), self.edge_embd_dim)
                .detach()
                .to(device)
            )

        # make it to full edge (interaction edge is done in layer)
        full_rec_e_index, full_rec_e = half_edge_to_full_edge(rec_e_index, rec_e)
        full_lig_e_index, full_lig_e = half_edge_to_full_edge(lig_e_index, lig_e)

        # merge timestep to features
        timestep_embedded_lig_h = timestep_embedded[lig_batch]
        timestep_embedded_lig_e = timestep_embedded[lig_batch[full_lig_e_index[1]]]
        timestep_embedded_inter_e = timestep_embedded[lig_batch[inter_e_index[1]]]
        lig_h = self.merge_lig_h_timestep(
            torch.cat([lig_h, timestep_embedded_lig_h], dim=1)
        )
        full_lig_e = self.merge_lig_e_timestep(
            torch.cat([full_lig_e, timestep_embedded_lig_e], dim=1)
        )
        inter_e = self.merge_inter_e_timestep(
            torch.cat([inter_e, timestep_embedded_inter_e], dim=1)
        )

        if self.confs["rec_edge_cutoff"] != None:
            if type(self.confs["rec_edge_cutoff"]) == list:  # max min
                ths_by_timestep = self.calc_cutoff_radius(
                    timestep,
                    self.confs["rec_edge_cutoff"][0],
                    self.confs["rec_edge_cutoff"][1],
                )
                (
                    sparse_full_rec_e_index,
                    sparse_full_rec_e,
                    rec_e_mask,
                ) = sparsify_edge_by_distance_each_batch(
                    full_rec_e_index, full_rec_e, rec_x, rec_batch, ths_by_timestep
                )
            else:
                ths = self.confs["rec_edge_cutoff"]
                (
                    sparse_full_rec_e_index,
                    sparse_full_rec_e,
                    rec_e_mask,
                ) = sparsify_edge_by_distance(
                    full_rec_e_index,
                    full_rec_e,
                    rec_x,
                    ths=ths,
                )
        else:
            sparse_full_rec_e_index = full_rec_e_index
            sparse_full_rec_e = full_rec_e

        # interaction layer
        for layer_idx in range(self.confs["n_layer"]):
            # sparsify edge (lig) if required
            if self.confs["lig_edge_cutoff"] != None:
                if type(self.confs["lig_edge_cutoff"]) == list:  # max min
                    ths_by_timestep = self.calc_cutoff_radius(
                        timestep,
                        self.confs["lig_edge_cutoff"][0],
                        self.confs["lig_edge_cutoff"][1],
                    )
                    (
                        sparse_full_lig_e_index,
                        sparse_full_lig_e,
                        lig_e_mask,
                    ) = sparsify_edge_by_distance_each_batch(
                        full_lig_e_index, full_lig_e, lig_x, lig_batch, ths_by_timestep
                    )
                else:
                    ths = self.confs["lig_edge_cutoff"]
                    (
                        sparse_full_lig_e_index,
                        sparse_full_lig_e,
                        lig_e_mask,
                    ) = sparsify_edge_by_distance(
                        full_lig_e_index,
                        full_lig_e,
                        lig_x,
                        ths=ths,
                    )
            else:
                sparse_full_lig_e_index = full_lig_e_index
                sparse_full_lig_e = full_lig_e

            # sparsify edge (inter)
            if self.confs["inter_edge_cutoff"] != None:
                if type(self.confs["inter_edge_cutoff"]) == list:  # max min
                    ths_by_timestep = self.calc_cutoff_radius(
                        timestep,
                        self.confs["inter_edge_cutoff"][0],
                        self.confs["inter_edge_cutoff"][1],
                    )
                    (
                        sparse_inter_e_index,
                        sparse_inter_e,
                        inter_e_mask,
                    ) = sparsify_inter_edge_by_distance_each_batch(
                        inter_e_index,
                        inter_e,
                        rec_x,
                        lig_x,
                        rec_batch,
                        ths=ths_by_timestep,
                    )
                else:
                    ths = self.confs["inter_edge_cutoff"]
                    (
                        sparse_inter_e_index,
                        sparse_inter_e,
                        inter_e_mask,
                    ) = sparsify_inter_edge_by_distance(
                        inter_e_index,
                        inter_e,
                        rec_x,
                        lig_x,
                        ths=ths,
                    )
            else:
                sparse_inter_e_index = inter_e_index
                sparse_inter_e = inter_e

            # heterogeneous interation layer
            rec_h, _, rec_e_, lig_h, lig_x, lig_e_, inter_e_ = self.inter_layers[
                layer_idx
            ](
                rec_h,
                rec_x,
                sparse_full_rec_e_index,
                sparse_full_rec_e,
                rec_batch,
                lig_h,
                lig_x,
                sparse_full_lig_e_index,
                sparse_full_lig_e,
                lig_batch,
                sparse_inter_e_index,
                sparse_inter_e,
            )

            # update only selected edges
            if self.confs["rec_edge_cutoff"] != None:
                full_rec_e[rec_e_mask == 1] = rec_e_
            else:
                full_rec_e = rec_e_

            if self.confs["lig_edge_cutoff"] != None:
                full_lig_e[lig_e_mask == 1] = lig_e_
            else:
                full_lig_e = lig_e_

            if self.confs["inter_edge_cutoff"] != None:
                inter_e[inter_e_mask == 1] = inter_e_
            else:
                inter_e = inter_e_

        # merge edges to half
        lig_e_index, lig_e = full_edge_to_half_edge(full_lig_e_index, full_lig_e)

        return rec_h, lig_h, lig_x, lig_e, inter_e

    def pred_lig_h_from_embded(self, embded_lig_h):
        return self.fc_readout_lig_h(embded_lig_h)

    def pred_lig_e_from_embded(self, embded_lig_e):
        return self.fc_readout_lig_e(embded_lig_e)

    def pred_inter_e_from_embded(self, embded_inter_e):
        return self.fc_readout_inter_e(embded_inter_e)

    def pred_rec_h_binary_from_embded(self, embded_rec_h):
        x = self.fc_readout_rec_h_binary(embded_rec_h).squeeze(1)
        return x

    def pred_inter_e_active(self, embded_rec_h, embded_lig_h, inter_e_index):
        rec_h = embded_rec_h[inter_e_index[0]]
        lig_h = embded_lig_h[inter_e_index[1]]
        inter_e = torch.cat([rec_h, lig_h], dim=1)
        inter_e = self.fc_readout_inter_e_active(inter_e)
        return inter_e

    def make_rec_to_lig_edge_index(self, rec_batch, lig_batch):
        # make edge from c (conditon) to i (interest)
        rec_batch_ = rec_batch.unsqueeze(-1).expand(-1, len(lig_batch))
        lig_batch_ = lig_batch.unsqueeze(0).expand(len(rec_batch), -1)
        edge_index = torch.where(rec_batch_ == lig_batch_)
        return torch.stack(edge_index, dim=0)

    def calc_cutoff_radius(self, timestep, ths_max, ths_min):
        # larger the timestep (p(xt|x0)), the radius should be larger to capture the overall structure !!!
        d_ths = ths_min + (ths_max - ths_min) * timestep / self.n_timestep
        return d_ths

    """def calc_cutoff_radius_cosine(self, timestep, ths_max, ths_min):
        # larger the timestep (p(xt|x0)), the radius should be larger to capture overall structure !!!
        d_ths = 0.5 * (
            ths_max
            + ths_min
            - (ths_max - ths_min) * torch.cos(PI * timestep / self.n_timestep)
        )
        return d_ths"""


class NCIVAE(nn.Module):
    r"""
    VAE for generating NCI pattern
    """

    def __init__(self, confs):
        super().__init__()
        self.confs = confs

        self.embd_h = nn.Linear(confs["rec_h_dim"], confs["node_embd_dim"], bias=False)
        self.embd_e = nn.Embedding(confs["rec_e_dim"], confs["edge_embd_dim"])
        self.embd_i = nn.Linear(confs["rec_i_dim"], confs["node_embd_dim"], bias=False)
        self.emdb_nl = nn.Linear(1, confs["node_embd_dim"])

        self.fc_enc = nn.Linear(confs["node_embd_dim"] * 3, confs["node_embd_dim"])
        self.encoder = []
        for _ in range(confs["num_layers"]):
            self.encoder.append(
                IGNNLayer(
                    confs["node_embd_dim"],
                    confs["edge_embd_dim"],
                    dropout=confs["dropout"],
                    update_edge=confs["update_edge"],
                )
            )
        self.encoder = nn.ModuleList(self.encoder)
        self.readout = nn.Linear(confs["node_embd_dim"], confs["node_embd_dim"])
        self.vae = VariationalEncoder(confs)

        self.fc_dec = nn.Linear(
            confs["node_embd_dim"] * 2 + confs["latent_embd_dim"],
            confs["node_embd_dim"],
        )
        self.decoder = []
        for _ in range(confs["num_layers"]):
            self.decoder.append(
                IGNNLayer(
                    confs["node_embd_dim"],
                    confs["edge_embd_dim"],
                    dropout=confs["dropout"],
                    update_edge=confs["update_edge"],
                )
            )
        self.decoder = nn.ModuleList(self.decoder)
        self.readout_final = nn.Sequential(
            nn.Linear(confs["node_embd_dim"], confs["rec_i_dim"]), nn.Sigmoid()
        )

        self.recon_loss = nn.BCELoss(reduction="none")
        self.reg_loss_coeff = 1.0

    def forward(self, graph):
        h = graph.h
        x = graph.x
        e = graph.e
        e_index = graph.e_index
        i = graph.i
        nl = graph.nl
        N = graph.num_graphs

        h_embd = self.embd_h(h)
        e_embd = self.embd_e(e)
        i_embd = self.embd_i(i)
        nl_embd = self.emdb_nl(nl.unsqueeze(1))[graph.batch]

        # 1. Encoder q(z | h, x, e, nl, i)
        h_embd = self.fc_enc(torch.cat([h_embd, i_embd, nl_embd], -1))
        for layer in self.encoder:
            h_embd, x, e_embd = layer(h_embd, x, e_embd, e_index)
        g = scatter(self.readout(h_embd), graph.batch, dim=0, reduce="mean")

        z, reg_loss = self.vae(g)
        z_expand = z[graph.batch]

        # 2. Decoder p(i | h, x, e, nl, z)
        h_embd = self.embd_h(h)
        h_embd = self.fc_dec(torch.cat([h_embd, z_expand, nl_embd], -1))
        e_embd = self.embd_e(e)
        for layer in self.decoder:
            h_embd, x, e_embd = layer(h_embd, x, e_embd, e_index)

        i_pred = self.readout_final(h_embd)
        recon_loss = self.recon_loss(i_pred, i)

        reg_loss = self.reg_loss_coeff * reg_loss.mean()
        recon_loss = recon_loss.mean()
        total_loss = reg_loss + recon_loss
        return total_loss, reg_loss, recon_loss, i_pred, i

    def sample(self, graph):
        h = graph.h.cuda()
        x = graph.x.cuda()
        e = graph.e_type.cuda()
        e_index = graph.e_index.cuda()
        nl = graph.nl.cuda()
        N = graph.num_graphs
        batch = graph.batch.cuda()

        z = torch.randn((N, self.confs["latent_embd_dim"]), device=h.device)
        z_expand = z[batch]

        nl_embd = self.emdb_nl(nl.unsqueeze(1).float())[batch]
        h_embd = self.embd_h(h)
        h_embd = self.fc_dec(torch.cat([h_embd, z_expand, nl_embd], -1))
        e_embd = self.embd_e(e)
        for layer in self.decoder:
            h_embd, x, e_embd = layer(h_embd, x, e_embd, e_index)

        i_pred = self.readout_final(h_embd)
        i_pred = i_pred.round()

        total_e_type = []
        for k in range(N):  # batch index
            one_pred_i = i_pred[batch == k]
            num_rec_node = int(one_pred_i.shape[0])
            num_lig_node = int(nl[k])
            num_inter_edge = num_rec_node * num_lig_node
            one_pred_i[:, 0] = 0
            ncis = one_pred_i.nonzero()

            lig_idx = 0
            pred_e_type = [0] * num_inter_edge
            for l, m in ncis:  # if exceed, cut and proceed to next sample
                if lig_idx >= num_lig_node:
                    break
                pred_e_type[l * num_lig_node + lig_idx] = m
                lig_idx += 1
            total_e_type.append(torch.LongTensor(pred_e_type))

        return torch.concat(total_e_type), i_pred

    def reconstruct(self, graph):
        h = graph.h
        x = graph.x
        e = graph.e
        e_index = graph.e_index
        i = graph.i
        nl = graph.nl
        N = graph.num_graphs

        h_embd = self.embd_h(h)
        e_embd = self.embd_e(e)
        i_embd = self.embd_i(i)
        nl_embd = self.emdb_nl(nl.unsqueeze(1))[graph.batch]

        # 1. Encoder q(z | h, x, e, nl, i)
        h_embd = self.fc_enc(torch.cat([h_embd, i_embd, nl_embd], -1))
        for layer in self.encoder:
            h_embd, x, e_embd = layer(h_embd, x, e_embd, e_index)
        g = scatter(self.readout(h_embd), graph.batch, dim=0, reduce="mean")

        z, _ = self.vae(g)
        z_expand = z[graph.batch]

        # 2. Decoder p(i | h, x, e, nl, z)
        h_embd = self.embd_h(h)
        h_embd = self.fc_dec(torch.cat([h_embd, z_expand, nl_embd], -1))
        e_embd = self.embd_e(e)
        for layer in self.decoder:
            h_embd, x, e_embd = layer(h_embd, x, e_embd, e_index)

        i_pred = self.readout_final(h_embd)
        return i_pred.round()


class VariationalEncoder(nn.Module):
    r"""
    Skeleton code for encoder of VAE
    """

    def __init__(self, confs):
        super().__init__()
        self.confs = confs

        self.mean = nn.Linear(confs["node_embd_dim"], confs["latent_embd_dim"])
        self.logvar = nn.Linear(confs["node_embd_dim"], confs["latent_embd_dim"])

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device)
        return eps * std + mean

    def vae_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

    def forward(self, g):
        mean = self.mean(g)
        logvar = self.logvar(g)
        latent = self.reparameterize(mean, logvar)
        vae_loss = self.vae_loss(mean, logvar)
        return latent, vae_loss
