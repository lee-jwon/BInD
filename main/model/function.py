import torch
from torch.nn import functional as F
from torch_geometric.utils import degree, to_dense_adj


def sparsify_edge_by_distance(e_index, e, x, ths):
    d = torch.norm(x[e_index[0]] - x[e_index[1]], dim=1)
    e_mask = d < ths
    sparse_e_index = e_index[:, e_mask]
    sparse_e = e[e_mask]
    e_mask = e_mask
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_edge_by_distance_each_batch(e_index, e, x, batch, ths):
    # batch will be node batch! not edge batch
    ths_per_e = ths[batch[e_index[0]]]
    d = torch.norm(x[e_index[0]] - x[e_index[1]], dim=1)
    e_mask = d < ths_per_e
    sparse_e_index = e_index[:, e_mask]
    sparse_e = e[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_inter_edge_by_distance(e12_index, e12, x1, x2, ths):
    d = torch.norm(x1[e12_index[0]] - x2[e12_index[1]], dim=1)
    e_mask = d < ths
    sparse_e_index = e12_index[:, e_mask]
    sparse_e = e12[e_mask]
    e_mask = e_mask
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_inter_edge_by_distance_each_batch(e12_index, e12, x1, x2, batch1, ths):
    # batch will be node batch! not edge batch
    ths_per_e = ths[batch1[e12_index[0]]]
    d = torch.norm(x1[e12_index[0]] - x2[e12_index[1]], dim=1)
    e_mask = d < ths_per_e
    sparse_e_index = e12_index[:, e_mask]
    sparse_e = e12[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_inter_edge_by_knn(e12_index, e12, x1, x2, k=8):
    d = torch.cdist(x2, x1)  # [L, R]
    knn_index = torch.topk(d, k=k, dim=1, largest=False)[1]  # [L, k]
    knn_mask = torch.zeros_like(d).bool()
    for i in range(d.shape[0]):
        knn_mask[i, knn_index[i]] = True
    e_mask = torch.zeros(e12.shape[0]).bool()
    for i in range(e12.shape[0]):
        e_mask[i] = knn_mask[e12_index[1][i], e12_index[0][i]]
    sparse_e_index = e12_index[:, e_mask]
    sparse_e = e12[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_edge_by_knn(e_index, e, x, k=8):
    d = torch.cdist(x, x)  # [L, L]
    knn_index = torch.topk(d, k=k, dim=1, largest=False)[1]  # [L, k]
    knn_mask = torch.zeros_like(d).bool()
    for i in range(d.shape[0]):
        knn_mask[i, knn_index[i]] = True
    e_mask = torch.zeros(e.shape[0]).bool()
    for i in range(e.shape[0]):
        e_mask[i] = knn_mask[e_index[1][i], e_index[0][i]]
    sparse_e_index = e_index[:, e_mask]
    sparse_e = e[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_inter_edge_by_knn_compress(e12_index, e12, x1, x2, k=8):
    d = torch.cdist(x2, x1)  # [L, R]
    index = torch.arange(1, e12.shape[0] + 1).to(x1.device)
    knn_index = torch.topk(d, k=k, dim=1, largest=False)[1]  # [L, k]
    repeat = (
        (torch.arange(0, x2.shape[0]) * x1.shape[0]).unsqueeze(1).repeat(1, k)
    ).to(
        x1.device
    )  # [L, k]
    knn_index_expand = knn_index.reshape(-1) + repeat.reshape(-1)
    knn_mask = torch.zeros(x1.shape[0] * x2.shape[0]).to(x1.device)
    knn_mask[knn_index_expand] = 1
    index_to_adj = to_dense_adj(e12_index, edge_attr=index, max_num_nodes=x1.shape[0])[
        0, :, : x2.shape[0]
    ].T
    index_to_adj_expand = index_to_adj.reshape(-1)
    e_mask_expand = index_to_adj_expand * knn_mask
    e_mask = index_to_adj_expand[e_mask_expand.nonzero()].squeeze(-1) - 1
    sparse_e_index = e12_index[:, e_mask]
    sparse_e = e12[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def sparsify_edge_by_knn_compress(e_index, e, x, k=8):
    d = torch.cdist(x, x)  # [L, L]
    index = torch.arange(1, e.shape[0] + 1).to(x.device)
    knn_index = torch.topk(d, k=k, dim=1, largest=False)[1]  # [L, k]
    repeat = ((torch.arange(0, x.shape[0]) * x.shape[0]).unsqueeze(1).repeat(1, k)).to(
        x.device
    )  # [L, k]
    knn_index_expand = knn_index.reshape(-1) + repeat.reshape(-1)
    knn_mask = torch.zeros(x.shape[0] ** 2).to(x.device)
    knn_mask[knn_index_expand] = 1
    index_to_adj = to_dense_adj(e_index, edge_attr=index, max_num_nodes=x.shape[0])[0].T
    index_to_adj_expand = index_to_adj.reshape(-1)
    e_mask_expand = index_to_adj_expand * knn_mask
    e_mask = index_to_adj_expand[e_mask_expand.nonzero()].squeeze(-1) - 1
    sparse_e_index = e_index[:, e_mask]
    sparse_e = e[e_mask]
    return sparse_e_index, sparse_e, e_mask.long()


def complete_edge_to_existing_edge(e_index, e_type):
    """
    0th index will be the nonetype edge!
    """
    e_real_edge_mask = e_type != 0
    e_index = e_index.transpose(1, 0)[e_real_edge_mask].transpose(1, 0)
    e_type = e_type[e_real_edge_mask]
    return e_index, e_type, e_real_edge_mask


def compute_active_node_for_inter_edge(e12_index, e12_1, e12_2):
    """
    returns activet edges, and mask matrix
    """
    active_edge_mask = e12_1 != e12_2  # if changed, true
    active_e_index = e12_index.transpose(1, 0)[active_edge_mask].transpose(1, 0)
    active_node_index_1 = torch.unique(active_e_index[0,])
    active_node_index_2 = torch.unique(active_e_index[1,])
    return active_edge_mask, active_node_index_1, active_node_index_2


def compute_degree_for_inter_edge(e12_index, e12, n1=None, n2=None):
    if n1 is None:
        n1 = e12_index[0, :].max() + 1
    if n2 is None:
        n2 = e12_index[1, :].max() + 1
    e_real_edge_mask = e12 != 0  # if True, real edge
    e_index = e12_index.transpose(1, 0)[e_real_edge_mask].transpose(1, 0)
    e1_index = e_index[0, :]
    e2_index = e_index[1, :]
    d1 = degree(e1_index.flatten(), num_nodes=n1)
    d2 = degree(e2_index.flatten(), num_nodes=n2)
    return d1.long(), d2.long()
