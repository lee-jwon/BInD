import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def extract(coef, t, batch, ndim=2):
    if batch is None:  # for debugging
        out = coef[t]
    else:
        out = coef[t][batch]
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError("ndim > 3")


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_substract_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + ((a - maximum).exp() - (b - maximum).exp()).log()


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)  # G := -log(-log(U))
    sample_index = (gumbel_noise + logits).argmax(dim=-1)  # Gumbel max
    return sample_index


def categorical_kl(log_prob1, log_prob2):
    assert log_prob1.size() == log_prob2.size()
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


def bernoulli_kl(p1, p2):
    assert p1.size() == p2.size()
    eps = 1e-30
    kl = p1 * torch.log(p1 / (p2 + eps) + eps) + (1 - p1) * torch.log(
        (1 - p1) / (1 - p2 + eps) + eps
    )
    return kl


def bernoulli_kl_logprob(p1, p2):
    eps = 1e-30
    kl = p1.exp() * (p1 - p2) + (1 - p1.exp()) * (
        torch.log(1 - p1.exp() + eps) - torch.log(1 - p2.exp() + eps)
    )
    return kl
