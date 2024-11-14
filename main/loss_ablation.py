import math
from copy import deepcopy
from itertools import dropwhile
from math import pi as PI

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch_geometric.utils import bipartite_subgraph
from torch_scatter import scatter

from main.diffusion.utils import bernoulli_kl, categorical_kl
from main.model.function import *
from main.model.layer import *


class BaselineLoss(nn.Module):
    def __init__(self, confs, model, transitions):
        super().__init__()

        self.confs = confs
        self.model = model
        self.lig_h_transition = transitions[0]
        self.lig_x_transition = transitions[1]
        self.lig_e_transition = transitions[2]
        self.inter_e_transition = transitions[3]

    def forward(
        self, rec_graph, lig_graph, inter_e_index, inter_e_type, timestep, is_train=None
    ):
        # get batch
        lig_h_batch = lig_graph.batch
        lig_e_batch = lig_graph.batch[lig_graph.e_index[0]]
        inter_e_batch = lig_graph.batch[inter_e_index[1]]

        # apply noise
        _, pert_lig_h_type = self.lig_h_transition.sample_xt_from_x0(
            lig_graph.h_type, timestep, lig_h_batch
        )
        _, _, epsilon_lig_x, pert_lig_x = self.lig_x_transition.xt_from_x0(
            lig_graph.x, timestep, lig_h_batch
        )
        _, pert_lig_e_type = self.lig_e_transition.sample_xt_from_x0(
            lig_graph.e_type, timestep, lig_e_batch
        )
        if not self.confs["abl_igen"]:
            _, pert_inter_e_type = self.inter_e_transition.sample_xt_from_x0(
                inter_e_type, timestep, inter_e_batch
            )
        else:
            inter_e_index = None
            pert_inter_e_type = None
        # make model predictions
        (
            _,
            embded_lig_h,
            embded_lig_x,
            embded_lig_e,
            embded_inter_e,
        ) = self.model.propagate(
            rec_graph.h,
            rec_graph.x,
            rec_graph.e_index,
            rec_graph.e_type,
            rec_graph.batch,
            pert_lig_h_type,
            pert_lig_x,
            lig_graph.e_index,
            pert_lig_e_type,
            lig_graph.batch,
            timestep,
            inter_e_index,  # None when abl_igen
            pert_inter_e_type,  # None when abl_igen
        )
        pred_lig_h = self.model.pred_lig_h_from_embded(embded_lig_h)
        pred_lig_h_logprob = torch.log_softmax(pred_lig_h, dim=1)
        pred_lig_x = embded_lig_x
        pred_lig_e = self.model.pred_lig_e_from_embded(embded_lig_e)
        pred_lig_e_logprob = torch.log_softmax(pred_lig_e, dim=1)
        pred_inter_e = self.model.pred_inter_e_from_embded(embded_inter_e)
        pred_inter_e_logprob = torch.log_softmax(pred_inter_e, dim=1)

        # atom type loss
        lig_h_logprob = self.lig_h_transition.idx_to_logprob(lig_graph.h_type)
        prev_lig_h_logprob, _ = self.lig_h_transition.calc_posterior_and_sample(
            lig_h_logprob, pert_lig_h_type, timestep, lig_h_batch
        )
        pred_prev_lig_h_logprob, _ = self.lig_h_transition.calc_posterior_and_sample(
            pred_lig_h_logprob, pert_lig_h_type, timestep, lig_h_batch
        )
        lig_h_loss = self.lig_h_transition.calc_loss(
            pred_prev_lig_h_logprob,
            prev_lig_h_logprob,
            lig_graph.h_type,
            timestep,
            lig_h_batch,
        )
        lig_h_loss = lig_h_loss.mean()
        lig_h_ce_loss = categorical_kl(lig_h_logprob, pred_lig_h_logprob).mean()

        # atom position loss
        if self.confs["mse_train_objective"] == "noise":
            lig_x_loss = F.mse_loss(
                pred_lig_x, epsilon_lig_x.detach(), reduction="mean"
            )
        elif self.confs["mse_train_objective"] == "data":
            lig_x_loss = F.mse_loss(pred_lig_x, lig_graph.x.detach(), reduction="mean")
        else:
            raise NotImplementedError

        # bond type loss
        lig_e_logprob = self.lig_e_transition.idx_to_logprob(lig_graph.e_type)
        prev_lig_e_logprob, _ = self.lig_e_transition.calc_posterior_and_sample(
            lig_e_logprob, pert_lig_e_type, timestep, lig_e_batch
        )
        pred_prev_lig_e_logprob, _ = self.lig_e_transition.calc_posterior_and_sample(
            pred_lig_e_logprob, pert_lig_e_type, timestep, lig_e_batch
        )
        lig_e_loss = self.lig_e_transition.calc_loss(
            pred_prev_lig_e_logprob,
            prev_lig_e_logprob,
            lig_graph.e_type,
            timestep,
            lig_e_batch,
        )
        lig_e_loss = lig_e_loss.mean()
        lig_e_ce_loss = categorical_kl(lig_e_logprob, pred_lig_e_logprob).mean()

        # interaction type loss
        if not self.confs["abl_igen"]:
            inter_e_logprob = self.inter_e_transition.idx_to_logprob(inter_e_type)
            (
                prev_inter_e_logprob,
                _,
            ) = self.inter_e_transition.calc_posterior_and_sample(
                inter_e_logprob, pert_inter_e_type, timestep, inter_e_batch
            )
            (
                pred_prev_inter_e_logprob,
                _,
            ) = self.inter_e_transition.calc_posterior_and_sample(
                pred_inter_e_logprob, pert_inter_e_type, timestep, inter_e_batch
            )
            inter_e_loss = self.inter_e_transition.calc_loss(
                pred_prev_inter_e_logprob,
                prev_inter_e_logprob,
                inter_e_type,
                timestep,
                inter_e_batch,
            )
            inter_e_loss = inter_e_loss.mean()
            inter_e_ce_loss = categorical_kl(
                inter_e_logprob, pred_inter_e_logprob
            ).mean()
        else:
            inter_e_loss = 0.0

        lig_h_loss = lig_h_loss
        lig_x_loss = lig_x_loss
        lig_e_loss = lig_e_loss
        inter_e_loss = inter_e_loss

        # merge all losses
        loss_dict = {}
        loss = (
            lig_h_loss * self.confs["lig_h_loss"]
            + lig_x_loss * self.confs["lig_x_loss"]
            + lig_e_loss * self.confs["lig_e_loss"]
            + inter_e_loss * self.confs["inter_e_loss"]
            + lig_h_ce_loss
            * self.confs["lig_h_loss"]
            * self.confs["categorical_ce_loss_ratio"]
            + lig_e_ce_loss
            * self.confs["lig_e_loss"]
            * self.confs["categorical_ce_loss_ratio"]
            # + inter_e_ce_loss
            * self.confs["inter_e_loss"]
            * self.confs["categorical_ce_loss_ratio"]
        )
        loss_dict["lig_h_loss"] = lig_h_loss
        loss_dict["lig_x_loss"] = lig_x_loss
        loss_dict["lig_e_loss"] = lig_e_loss
        loss_dict["inter_e_loss"] = inter_e_loss
        loss_dict["lig_h_ce_loss"] = lig_h_ce_loss
        loss_dict["lig_e_ce_loss"] = lig_e_ce_loss
        loss_dict["inter_e_ce_loss"] = 0.0

        return loss, loss_dict


class InterEDGEDirectLoss(nn.Module):
    def __init__(self, confs, model, transitions):
        super().__init__()

        self.confs = confs
        self.model = model
        self.lig_h_transition = transitions[0]
        self.lig_x_transition = transitions[1]
        self.lig_e_transition = transitions[2]
        self.inter_e_transition = transitions[3]

    def forward(
        self, rec_graph, lig_graph, inter_e_index, inter_e_type, timestep, is_train=None
    ):
        # get batch
        rec_h_batch = rec_graph.batch
        lig_h_batch = lig_graph.batch
        lig_e_batch = lig_graph.batch[lig_graph.e_index[0]]
        inter_e_batch = lig_graph.batch[inter_e_index[1]]

        # apply noise
        _, pert_lig_h_type = self.lig_h_transition.sample_xt_from_x0(
            lig_graph.h_type, timestep, lig_h_batch
        )
        _, _, epsilon_lig_x, pert_lig_x = self.lig_x_transition.xt_from_x0(
            lig_graph.x, timestep, lig_h_batch
        )
        _, pert_lig_e_type = self.lig_e_transition.sample_xt_from_x0(
            lig_graph.e_type, timestep, lig_e_batch
        )
        if not self.confs["abl_igen"]:
            _, pert_inter_e_type = self.inter_e_transition.sample_xt_from_x0(
                inter_e_type, timestep, inter_e_batch
            )
        else:
            inter_e_index = None
            pert_inter_e_type = None

        # get real edges from full edges
        """(
            real_pert_inter_e_index,
            real_pert_inter_e_type,
            real_pert_inter_e_mask,
        ) = complete_edge_to_existing_edge(inter_e_index, pert_inter_e_type)"""

        # make model predictions
        (
            embded_rec_h,
            embded_lig_h,
            embded_lig_x,
            embded_lig_e,
            _,
        ) = self.model.propagate(
            rec_graph.h,
            rec_graph.x,
            rec_graph.e_index,
            rec_graph.e_type,
            rec_graph.batch,
            pert_lig_h_type,
            pert_lig_x,
            lig_graph.e_index,
            pert_lig_e_type,
            lig_graph.batch,
            timestep,
            inter_e_index,  # None when abl_igen
            pert_inter_e_type,  # None when abl_igen
        )
        pred_lig_h = self.model.pred_lig_h_from_embded(embded_lig_h)
        pred_lig_h_logprob = torch.log_softmax(pred_lig_h, dim=1)
        pred_lig_x = embded_lig_x
        pred_lig_e = self.model.pred_lig_e_from_embded(embded_lig_e)
        pred_lig_e_logprob = torch.log_softmax(pred_lig_e, dim=1)
        # pred_inter_e = self.model.pred_inter_e_from_embded(embded_inter_e)
        # pred_inter_e_logprob = torch.log_softmax(pred_inter_e, dim=1)

        # atom type loss
        lig_h_logprob = self.lig_h_transition.idx_to_logprob(lig_graph.h_type)
        prev_lig_h_logprob, _ = self.lig_h_transition.calc_posterior_and_sample(
            lig_h_logprob, pert_lig_h_type, timestep, lig_h_batch
        )
        pred_prev_lig_h_logprob, _ = self.lig_h_transition.calc_posterior_and_sample(
            pred_lig_h_logprob, pert_lig_h_type, timestep, lig_h_batch
        )
        lig_h_loss = self.lig_h_transition.calc_loss(
            pred_prev_lig_h_logprob,
            prev_lig_h_logprob,
            lig_graph.h_type,
            timestep,
            lig_h_batch,
        )
        lig_h_loss = lig_h_loss.mean()

        # atom position loss
        if self.confs["mse_train_objective"] == "noise":
            lig_x_loss = F.mse_loss(
                pred_lig_x, epsilon_lig_x.detach(), reduction="mean"
            )
        elif self.confs["mse_train_objective"] == "data":
            lig_x_loss = F.mse_loss(pred_lig_x, lig_graph.x.detach(), reduction="mean")
        else:
            raise NotImplementedError

        # bond type loss
        lig_e_logprob = self.lig_e_transition.idx_to_logprob(lig_graph.e_type)
        prev_lig_e_logprob, _ = self.lig_e_transition.calc_posterior_and_sample(
            lig_e_logprob, pert_lig_e_type, timestep, lig_e_batch
        )
        pred_prev_lig_e_logprob, _ = self.lig_e_transition.calc_posterior_and_sample(
            pred_lig_e_logprob, pert_lig_e_type, timestep, lig_e_batch
        )
        lig_e_loss = self.lig_e_transition.calc_loss(
            pred_prev_lig_e_logprob,
            prev_lig_e_logprob,
            lig_graph.e_type,
            timestep,
            lig_e_batch,
        )
        lig_e_loss = lig_e_loss.mean()

        # interaction activity loss 1) copmpute degree 2) compute activity 3) bernoulli KL
        # calc degree and active node bernprob
        rec_d_for_inter_e, _ = compute_degree_for_inter_edge(
            inter_e_index, inter_e_type
        )
        pert_rec_d_for_inter_e, _ = compute_degree_for_inter_edge(
            inter_e_index, pert_inter_e_type
        )
        delta_rec_d_for_inter_e = rec_d_for_inter_e - pert_rec_d_for_inter_e
        inter_e_gamma = self.inter_e_transition.get_gammas(timestep, rec_h_batch)
        active_rec_h_for_inter_e_prob = 1 - (1 - inter_e_gamma).pow(
            delta_rec_d_for_inter_e
        )
        active_rec_h_for_inter_e_logprob = torch.log(
            torch.stack(
                [
                    active_rec_h_for_inter_e_prob + 1e-30,
                    1.0 - active_rec_h_for_inter_e_prob,
                ],
                dim=1,
            )
        )
        pred_active_rec_h_for_inter_e_logprob = torch.log_softmax(
            self.model.pred_rec_h_binary_from_embded(embded_rec_h), dim=1
        )
        active_rec_h_for_inter_e_loss = categorical_kl(
            active_rec_h_for_inter_e_logprob, pred_active_rec_h_for_inter_e_logprob
        )
        active_rec_h_for_inter_e_loss = active_rec_h_for_inter_e_loss.mean()

        # calc onestep posterior and get actives
        inter_e_logprob = self.inter_e_transition.idx_to_logprob(inter_e_type)
        (
            prev_inter_e_logprob,
            prev_inter_e_type,  # sampled
        ) = self.inter_e_transition.calc_posterior_and_sample(
            inter_e_logprob, pert_inter_e_type, timestep, inter_e_batch
        )
        (
            active_rec_h_mask_for_inter_e,
            active_rec_h_index_for_inter_e,
            _,
        ) = compute_active_node_for_inter_edge(
            inter_e_index, prev_inter_e_type, pert_inter_e_type
        )

        if active_rec_h_mask_for_inter_e.sum() > 0:
            # interaction type loss
            sub_inter_e_index, _, sub_inter_e_mask = bipartite_subgraph(
                [
                    active_rec_h_index_for_inter_e,
                    torch.arange(0, len(lig_graph.batch), 1).to(inter_e_index.device),
                ],
                inter_e_index,
                return_edge_mask=True,
            )
            pred_sub_inter_e = self.model.pred_inter_e_active(
                embded_rec_h, embded_lig_e, sub_inter_e_index
            )
            pred_sub_inter_e_logprob = torch.log_softmax(pred_sub_inter_e, dim=1)
            (
                pred_prev_sub_inter_e_logprob,
                _,
            ) = self.inter_e_transition.calc_posterior_and_sample(
                pred_sub_inter_e_logprob,
                pert_inter_e_type[sub_inter_e_mask],
                timestep,
                inter_e_batch[sub_inter_e_mask],
            )
            inter_e_loss = self.inter_e_transition.calc_loss(
                pred_prev_sub_inter_e_logprob,
                prev_inter_e_logprob[sub_inter_e_mask],
                inter_e_type[sub_inter_e_mask],
                timestep,
                inter_e_batch[sub_inter_e_mask],
            )
            inter_e_loss = inter_e_loss.mean()
        else:
            inter_e_loss = torch.Tensor([0.0]).squeeze()

        lig_h_loss = lig_h_loss
        lig_x_loss = lig_x_loss
        lig_e_loss = lig_e_loss
        inter_e_loss = inter_e_loss

        # merge all losses
        loss = (
            lig_h_loss * self.confs["lig_h_loss"]
            + lig_x_loss * self.confs["lig_x_loss"]
            + lig_e_loss * self.confs["lig_e_loss"]
            + inter_e_loss * self.confs["inter_e_loss"]
            + active_rec_h_for_inter_e_loss
            * self.confs["active_rec_h_for_inter_e_loss"]
        )

        return (
            loss,
            lig_h_loss,
            lig_x_loss,
            lig_e_loss,
            inter_e_loss,
            active_rec_h_for_inter_e_loss,
        )
