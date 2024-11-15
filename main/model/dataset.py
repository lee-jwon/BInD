import os
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_scatter import scatter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_pickle_file(fn):
    try:
        with open(fn, "rb") as f:
            return pickle.load(f)
    except:
        print(fn)
        return None


def load_data_in_parallel(fns):
    num_workers = 8
    with Pool(num_workers) as pool:
        data = list(tqdm(pool.imap(load_pickle_file, fns), total=len(fns)))
    data = [x for x in data if x != None]
    return data


class RecLigDataset(Dataset):
    def __init__(
        self,
        fns,
        center_to="none",
        rec_noise=0.0,
        mode="train",
        povme_test_fn=None,
        pre_load=False,
        is_dict=False,
    ):
        super().__init__()
        self.fns = fns
        assert center_to in ["lig", "rec", "none"]
        assert mode in ["train", "ref", "povme", "extracted_interaction"]
        self.center_to = center_to
        self.rec_noise = rec_noise
        self.mode = mode
        self.extracted_interactions = None
        self.is_dict = is_dict

        self.pre_load = pre_load
        if pre_load:
            self.data = load_data_in_parallel(fns)
        if is_dict:
            self.data = self.fns

        if mode == "povme":
            if povme_test_fn is not None:
                df = pd.read_csv(povme_test_fn)
                self.my_key_to_v = dict(zip(df["my_key"], df["v"]))
            else:
                # assert self.__len__() == 1  # need to be single data
                self.my_key_to_v = dict(zip([i for i in range(self.__len__())], [self.data[0]["v"] for _ in range(self.__len__())]))

    def __len__(self):
        return len(self.fns)

    def append_sampler(self, x):
        self.prior_atom_sampler = x

    def append_extracted_interactions(self, extracted_interactions):  # list of list
        self.extracted_interactions = extracted_interactions
        assert len(self.extracted_interactions) == len(self.data)
        self.mode = "extracted_interaction"

    def __getitem__(self, idx):
        if self.pre_load or self.is_dict:
            sample = deepcopy(self.data[idx])
        else:
            sample = load_pickle_file(self.fns[idx])

        # gaussian noise
        if self.rec_noise > 0.0:
            gn = np.random.normal(0, self.rec_noise, size=sample["rec"]["x"].shape)
            sample["rec"]["x"] = sample["rec"]["x"] + gn

        if self.center_to == "lig":
            sample["lig"]["x"] = sample["lig"]["x"] - sample["lig"]["centroid"]
            sample["rec"]["x"] = sample["rec"]["x"] - sample["lig"]["centroid"]
        elif self.center_to == "rec":
            sample["lig"]["x"] = sample["lig"]["x"] - sample["rec"]["centroid"]
            sample["rec"]["x"] = sample["rec"]["x"] - sample["rec"]["centroid"]
        elif self.center_to == "none":
            pass
        else:
            raise NotImplementedError

        # merge receptor properties and merge
        rec_h_aa = F.one_hot(torch.Tensor(sample["rec"]["h_aa"]).long(), num_classes=21)
        rec_h_type = F.one_hot(
            torch.Tensor(sample["rec"]["h_type"]).long(), num_classes=5
        )
        rec_h_charge = F.one_hot(
            torch.Tensor(sample["rec"]["h_charge"]).long(), num_classes=5
        )
        rec_h_numh = F.one_hot(
            torch.Tensor(sample["rec"]["h_numH"]).long(), num_classes=7
        )
        rec_h_ca = F.one_hot(
            torch.Tensor(sample["rec"]["h_isCA"]).long(), num_classes=2
        )
        # rec_h_surf = torch.Tensor(sample["rec"]["h_surf"]).float().unsqueeze(1)
        # rec_h_interable = torch.Tensor(sample["rec"]["h_interable"]).float()
        rec_h = torch.cat(
            [
                rec_h_type,
                rec_h_aa,
                rec_h_charge,
                rec_h_numh,
                rec_h_ca,
            ],
            dim=1,
        )  # 21 + 5 + 5 + 7 + 2 + 1 = 41

        # get inter edge (direction is rec to lig)
        inter_e_index = torch.Tensor(sample["inter"]["rec_to_lig_index"]).long()
        inter_e_type = torch.Tensor(sample["inter"]["rec_to_lig_type"]).long()

        # get rec graph
        rec_graph = Data(
            h=rec_h.float(),
            h_type=torch.Tensor(sample["rec"]["h_type"]).long(),
            x=torch.Tensor(sample["rec"]["x"]).float(),
            e_index=torch.Tensor(sample["rec"]["e_index"]).long(),
            e_type=torch.Tensor(sample["rec"]["e_type"]).long(),
            centroid=torch.Tensor(sample["rec"]["centroid"]).unsqueeze(0),
            fn=sample["rec_fn"],
        )

        # get lig graph
        lig_graph = Data(
            h_type=torch.Tensor(sample["lig"]["h_type"]).long(),
            x=torch.Tensor(sample["lig"]["x"]).float(),
            e_index=torch.Tensor(sample["lig"]["e_index"]).long(),
            e_type=torch.Tensor(sample["lig"]["e_type"]).long(),
            e_hop=torch.Tensor(sample["lig"]["e_hop"]).long(),
            centroid=torch.Tensor(sample["lig"]["centroid"]).unsqueeze(0),
            fn=sample["lig_fn"],
            my_key=sample["my_key"],
        )

        if self.mode == "train":
            pass
        elif self.mode == "ref":
            pass
        elif self.mode == "povme":
            povme_v = self.my_key_to_v[lig_graph.my_key]
            povme_sampled_n = self.prior_atom_sampler.sample(povme_v)
            cur_n = lig_graph.x.size(0)
            if povme_sampled_n < cur_n:
                for _ in range(cur_n - povme_sampled_n):
                    non_inter_lig_node_idx = get_non_inter_lig_idx(
                        inter_e_index, inter_e_type
                    )
                    if len(non_inter_lig_node_idx) >= 1:
                        (
                            lig_graph,
                            inter_e_index,
                            inter_e_type,
                        ) = remove_lig_node_given_idx(
                            lig_graph,
                            inter_e_index,
                            inter_e_type,
                            non_inter_lig_node_idx[0],
                        )
            elif povme_sampled_n > cur_n:
                for _ in range(povme_sampled_n - cur_n):
                    lig_graph, inter_e_index, inter_e_type = add_lig_node_last(
                        lig_graph, inter_e_index, inter_e_type
                    )
            else:
                pass
        elif self.mode == "extracted_interaction":
            if True:  # opt with same number of atom
                # randomly choose NCIs from extracted NCIs
                ext_e_index, ext_e_type = random.choice(
                    self.extractced_interactions[idx]
                )
                ext_lig_n = ext_e_index[1].max() + 1
                cur_n = lig_graph.x.size(0)
                assert ext_e_index[0].max() + 1 == rec_graph.x.size(0)
                inter_e_type = torch.zeros_like(inter_e_type)
                # add or remove nodes when mismatch # will be used only for sampling
                if ext_lig_n < cur_n:
                    for _ in range(cur_n - ext_lig_n):
                        non_inter_lig_node_idx = get_non_inter_lig_idx(
                            inter_e_index, inter_e_type
                        )
                        if len(non_inter_lig_node_idx) >= 1:
                            (
                                lig_graph,
                                _,
                                _,
                            ) = remove_lig_node_given_idx(
                                lig_graph,
                                inter_e_index,
                                inter_e_type,
                                non_inter_lig_node_idx[0],
                            )
                elif ext_lig_n > cur_n:
                    for _ in range(ext_lig_n - cur_n):
                        lig_graph, _, _ = add_lig_node_last(
                            lig_graph, inter_e_index, inter_e_type
                        )
                else:
                    pass
                inter_e_index = torch.Tensor(ext_e_index)
                inter_e_type = torch.Tensor(ext_e_type)
            else:  # opt with new sampling from POVME
                povme_v = self.my_key_to_v[lig_graph.my_key]
                povme_sampled_n = self.prior_atom_sampler.sample(povme_v)
                ext_e_index, ext_e_type = random.choice(
                    self.extractced_interactions[idx]
                )
                ext_e_index = torch.Tensor(ext_e_index).long()
                ext_e_type = torch.Tensor(ext_e_type).long()
                ext_lig_n = ext_e_index[1].max() + 1  # extracted n
                assert inter_e_index[0].max() + 1 == rec_graph.x.size(0)
                # match lig_graph to ext (this will be always done)
                inter_e_type = torch.zeros_like(inter_e_type)
                # convert lig_graph to extracted
                lig_graph, _, _ = edit_graph_and_inter_to_desired_n(
                    lig_graph, inter_e_index, inter_e_type, ext_lig_n
                )
                # convert all to povme_sampled n
                lig_graph, ext_e_index, ext_e_type = edit_graph_and_inter_to_desired_n(
                    lig_graph, ext_e_index, ext_e_type, povme_sampled_n
                )
                inter_e_index = ext_e_index.long()
                inter_e_type = ext_e_type.long()

        return rec_graph, lig_graph, inter_e_index.long(), inter_e_type.long()


def edit_graph_and_inter_to_desired_n(lig_graph, inter_e_index, inter_e_type, target_n):
    "this function works in tensor"
    current_n = lig_graph.x.size(0)
    assert current_n == inter_e_index[1].max() + 1
    if target_n < current_n:
        for _ in range(current_n - target_n):
            non_inter_lig_node_idx = get_non_inter_lig_idx(inter_e_index, inter_e_type)
            if len(non_inter_lig_node_idx) >= 1:
                (
                    lig_graph,
                    inter_e_index,
                    inter_e_type,
                ) = remove_lig_node_given_idx(
                    lig_graph,
                    inter_e_index,
                    inter_e_type,
                    non_inter_lig_node_idx[0],
                )
    elif target_n > current_n:
        for _ in range(target_n - current_n):
            lig_graph, inter_e_index, inter_e_type = add_lig_node_last(
                lig_graph, inter_e_index, inter_e_type
            )
    else:
        pass
    return lig_graph, inter_e_index, inter_e_type


class RecDataset(Dataset):
    def __init__(self, fns, center_to="none", rec_noise=0.0, mode="train"):
        super().__init__()
        self.fns = fns
        assert center_to in ["rec", "none"]
        assert mode in ["train", "ref", "extracted_interaction"]
        self.center_to = center_to
        self.rec_noise = rec_noise
        self.mode = mode
        self.extractced_interactions = None

    def __len__(self):
        return len(self.fns)

    def append_sampler(self, x):
        self.prior_atom_sampler = x

    def __getitem__(self, idx):
        with open(self.fns[idx], "rb") as f:
            data = pickle.load(f)
        sample = deepcopy(data)

        # gaussian noise
        if self.rec_noise > 0.0:
            gn = np.random.normal(0, self.rec_noise, size=sample["rec"]["x"].shape)
            sample["rec"]["x"] = sample["rec"]["x"] + gn

        if self.center_to == "rec":
            sample["rec"]["x"] = sample["rec"]["x"] - sample["rec"]["centroid"]
        elif self.center_to == "none":
            pass
        else:
            raise NotImplementedError

        # merge receptor properties and merge
        rec_h_aa = F.one_hot(torch.Tensor(sample["rec"]["h_aa"]).long(), num_classes=21)
        rec_h_type = F.one_hot(
            torch.Tensor(sample["rec"]["h_type"]).long(), num_classes=5
        )
        rec_h_charge = F.one_hot(
            torch.Tensor(sample["rec"]["h_charge"]).long(), num_classes=5
        )
        rec_h_numh = F.one_hot(
            torch.Tensor(sample["rec"]["h_numH"]).long(), num_classes=7
        )
        rec_h_ca = F.one_hot(
            torch.Tensor(sample["rec"]["h_isCA"]).long(), num_classes=2
        )
        rec_h = torch.cat(
            [
                rec_h_type,
                rec_h_aa,
                rec_h_charge,
                rec_h_numh,
                rec_h_ca,
            ],
            dim=1,
        )  # 40

        # get inter edge (direction is rec to lig)
        inter_e_index = torch.Tensor(sample["inter"]["rec_to_lig_index"]).long()
        inter_e_type = torch.Tensor(sample["inter"]["rec_to_lig_type"]).long()
        inter_r_type = inter_e_type.reshape(sample["rec"]["x"].shape[0], -1)
        inter_r_type = F.one_hot(inter_r_type, num_classes=7).sum(dim=1)

        inter_e_mask = inter_e_type == 0
        for idx in inter_e_index[0][~inter_e_mask]:
            inter_r_type[idx][0] = 0
        inter_r_type[inter_r_type != 0] = 1

        # get rec graph
        rec_graph = Data(
            h=rec_h.float(),
            h_type=torch.Tensor(sample["rec"]["h_type"]).long(),
            x=torch.Tensor(sample["rec"]["x"]).float(),
            e_index=torch.Tensor(sample["rec"]["e_index"]).long(),
            e=torch.Tensor(sample["rec"]["e_type"]).long(),
            i=inter_r_type.float(),
            nl=torch.Tensor([sample["lig"]["x"].shape[0]]),
            centroid=torch.Tensor(sample["rec"]["centroid"]).unsqueeze(0),
            fn=sample["rec_fn"],
        )
        return rec_graph


def rec_lig_collate_fn(batch_list):
    bs = len(batch_list)
    rec_graph_list = [x[0] for x in batch_list]
    lig_graph_list = [x[1] for x in batch_list]
    inter_e_index_list = [x[2] for x in batch_list]  # list of [2, n_inter_edge]
    inter_e_type_list = [x[3] for x in batch_list]  # list of [n_inter_edge]

    # get n of lig and rec atoms
    n_rec_node_list = [graph.x.size(0) for graph in rec_graph_list]
    n_lig_node_list = [graph.x.size(0) for graph in lig_graph_list]

    # shift inter idxs
    shifted_inter_e_index_list = []
    n_rec_add, n_lig_add = 0, 0
    for i in range(bs):
        inter_e_index = inter_e_index_list[i]
        inter_e_index[0] += n_rec_add
        inter_e_index[1] += n_lig_add
        n_rec_add += n_rec_node_list[i]
        n_lig_add += n_lig_node_list[i]
        shifted_inter_e_index_list.append(inter_e_index)
    inter_e_index = torch.cat(shifted_inter_e_index_list, dim=1)
    inter_e_type = torch.cat(inter_e_type_list, dim=0)

    # batch graphs
    rec_graph = Batch.from_data_list(rec_graph_list)
    lig_graph = Batch.from_data_list(lig_graph_list)

    return rec_graph, lig_graph, inter_e_index, inter_e_type


def remove_lig_node_given_idx(graph, inter_e_index, inter_e, idx):
    """
    from data, remove the designated idx (ligand atom index)
    """
    new_graph = deepcopy(graph)
    new_inter_e_index = deepcopy(inter_e_index)
    new_inter_e = deepcopy(inter_e)

    # node feature
    node_mask = torch.arange(graph.h_type.size(0)) != idx
    new_graph.h_type = new_graph.h_type[node_mask]
    new_graph.x = new_graph.x[node_mask]

    # edge feature
    e_mask = (graph.e_index[0] != idx) & (graph.e_index[1] != idx)
    new_graph.e_index = graph.e_index[:, e_mask]
    new_graph.e_type = new_graph.e_type[e_mask]
    e_index_change_mask = new_graph.e_index > idx  # larger than deleted
    new_graph.e_index[e_index_change_mask] -= 1

    # inter feature
    inter_e_mask = inter_e_index[1] != idx
    new_inter_e_index = inter_e_index[:, inter_e_mask]
    new_inter_e = new_inter_e[inter_e_mask]
    inter_e_index_change_mask = new_inter_e_index[1] > idx
    new_inter_e_index[1, inter_e_index_change_mask] -= 1
    return new_graph, new_inter_e_index.long(), new_inter_e.long()


def add_lig_node_last(graph, inter_e_index, inter_e):
    new_graph = deepcopy(graph)
    new_inter_e_index = deepcopy(inter_e_index)
    new_inter_e = deepcopy(inter_e)

    # node feature
    new_graph.h_type = torch.cat([new_graph.h_type, torch.Tensor([0])], dim=0)
    new_graph.x = torch.cat([new_graph.x, torch.Tensor([[0, 0, 0]])], dim=0)

    # edge feature
    add_edge_index_1 = torch.Tensor(
        [graph.h_type.size(0) for _ in range(graph.h_type.size(0))]
    )
    add_edge_index_2 = torch.arange(graph.h_type.size(0))
    add_e_index = torch.stack([add_edge_index_1, add_edge_index_2], dim=0).long()
    add_e = torch.zeros(add_e_index.size(1)).long()
    new_graph.e_index = torch.cat([new_graph.e_index, add_e_index], dim=1)
    new_graph.e_type = torch.cat([new_graph.e_type, add_e], dim=0)

    # inter feature
    n_rec_node = inter_e_index[0].max() + 1
    add_inter_edge_index_1 = torch.arange(n_rec_node)
    add_inter_edge_index_2 = torch.Tensor(
        [graph.h_type.size(0) for _ in range(n_rec_node)]
    )
    add_inter_e_index = torch.stack(
        [add_inter_edge_index_1, add_inter_edge_index_2], dim=0
    ).long()
    add_inter_e = torch.zeros(add_inter_e_index.size(1)).long()
    new_inter_e_index = torch.cat([new_inter_e_index, add_inter_e_index], dim=1).long()
    new_inter_e = torch.cat([new_inter_e, add_inter_e], dim=0).long()
    return new_graph, new_inter_e_index, new_inter_e


def get_non_inter_lig_idx(inter_e_index, inter_e):
    """
    get ligand atom index which does not has any NCIs
    """
    inter_e_index = deepcopy(inter_e_index)
    n_lig_node = inter_e_index[1].max() + 1
    inter_e = deepcopy(inter_e)
    inter_e_mask = inter_e != 0  # 0th is empty type
    inter_e_index_lig = inter_e_index[1][
        inter_e_mask
    ].tolist()  # node idx with real inters
    inter_e_index_lig = set(inter_e_index_lig)
    non_inter_lig_idx = set(range(n_lig_node)) - inter_e_index_lig
    return list(non_inter_lig_idx)
