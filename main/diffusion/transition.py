import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import *


class ContinuousTransition(nn.Module):
    def __init__(self, betas):
        super().__init__()

        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        prev_alpha_bars = np.concatenate([[1.0], alpha_bars[:-1]])

        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alpha_bars = to_torch_const(alpha_bars)
        self.prev_alpha_bars = to_torch_const(prev_alpha_bars)

    def xt_from_x0(self, x0, timestep, batch):
        """
        q(x_t | x_0) = N(x_t | sqrt(alpha_bar_t) * x_0, alpha_bar_t * I)
        """
        alpha_bar = extract(self.alpha_bars, timestep, batch)
        epsilon = torch.randn_like(x0)
        mu = alpha_bar.sqrt() * x0
        sigma = (1 - alpha_bar).sqrt()
        xt = mu + sigma * epsilon
        return mu, sigma, epsilon, xt

    def xtaft_from_xt(self, xt, timestep, batch):
        """
        q(x_t+1 | x_t) = N(x_t+1 | sqrt(alpha_bar_t) * x_0, alpha_bar_t * I)
        """
        beta = extract(self.betas, timestep, batch)
        alpha_bar = extract(self.alpha_bars, timestep, batch)
        epsilon = torch.randn_like(xt)
        mu = (1.0 - beta).sqrt() * xt
        sigma = (beta).sqrt()
        xt = mu + sigma * epsilon
        return mu, sigma, epsilon, xt

    def sample_init(self, shape):
        return torch.randn(shape).to(self.betas.device)

    def xtprev_from_xt_x0(self, xt, x0, timestep, batch):
        """
        q(x_t-1 | x_t, x_0) = N(x_t-1 | mu, sigma^2)
        mu = [sqrt(alpha_bar_t-1) * beta_t / (1 - alpha_bar_t)] * x_0 +
             [sqrt(alpha_t) * (1 - alpha_bar_t-1) / (1 - alpha_bar_t)] * x_t
        sigma^2 = [(1 - alpha_bar_t-1) / (1 - alpha_bar_t)] * beta_t
        """
        prev_alpha_bar = extract(self.prev_alpha_bars, timestep, batch)
        beta = extract(self.betas, timestep, batch)
        alpha_bar = extract(self.alpha_bars, timestep, batch)
        alpha = extract(self.alphas, timestep, batch)
        z = torch.randn_like(x0)

        time_is_zero = timestep[batch] == 0  # q(x_0 | x_1)
        time_is_zero = time_is_zero.unsqueeze(1)

        c1 = prev_alpha_bar.sqrt() * beta / (1 - alpha_bar)
        c2 = alpha.sqrt() * (1 - prev_alpha_bar) / (1 - alpha_bar)
        mu = c1 * x0 + c2 * xt
        sigma = ((1 - prev_alpha_bar) * beta / (1 - alpha_bar)).sqrt()
        xtprev = mu + sigma * z

        xtprev = torch.where(time_is_zero, mu, xtprev)
        return mu, sigma, z, xtprev

    def xtprev_from_xt_x0_ode(self, xt, x0, timestep, batch):  # s smaller than t always
        """
        q(x_t-1 | x_t, x_0) = N(x_t-1 | mu, sigma^2)
        mu = [sqrt(alpha_bar_t-1) * beta_t / (1 - alpha_bar_t)] * x_0 +
             [sqrt(alpha_t) * (1 - alpha_bar_t-1) / (1 - alpha_bar_t)] * x_t
        sigma^2 = [(1 - alpha_bar_t-1) / (1 - alpha_bar_t)] * beta_t
        """
        # assert (timestep_s < timestep_t).sum() == len(timestep_s), "timestep_s should be less than timestep_t"

        prev_alpha_bar = extract(self.prev_alpha_bars, timestep, batch)
        beta = extract(self.betas, timestep, batch)
        alpha_bar = extract(self.alpha_bars, timestep, batch)
        alpha = extract(self.alphas, timestep, batch)

        time_is_zero = timestep[batch] == 0  # q(x_0 | x_1)
        time_is_zero = time_is_zero.unsqueeze(1)

        pred_epsilon = (xt - (alpha_bar).sqrt() * x0) / (1 - alpha_bar).sqrt()
        xtprev = prev_alpha_bar.sqrt() * x0 + (1 - prev_alpha_bar).sqrt() * pred_epsilon
        # xtprev = torch.where(time_is_zero, mu, xtprev)
        return None, None, None, xtprev

    def xtprev_from_xt_epsilon(self, xt, epsilon, timestep, batch):
        """
        q(x_t-1 | x_t, e) = N(x_t-1 | mu, sigma^2)
        mu = 1 / sqrt(alpha) * [x_t - (beta_t / sqrt(1 - alpha_bar_t)) * e]
        sigma^2 = [(1 - alpha_bar_t-1) / (1 - alpha_bar_t)] * beta_t
        """
        prev_alpha_bar = extract(self.prev_alpha_bars, timestep, batch)
        beta = extract(self.betas, timestep, batch)
        alpha_bar = extract(self.alpha_bars, timestep, batch)
        alpha = extract(self.alphas, timestep, batch)
        z = torch.randn_like(xt)

        time_is_zero = timestep[batch] == 0
        time_is_zero = time_is_zero.unsqueeze(1)

        c1 = 1 / alpha.sqrt()
        c2 = (1 - alpha) / (1 - alpha_bar).sqrt()
        sigma = ((1 - prev_alpha_bar) * beta / (1 - alpha_bar)).sqrt()
        mu = c1 * (xt - c2 * epsilon)
        xtprev = mu + sigma * z
        xtprev = torch.where(time_is_zero, mu, xtprev)
        return mu, sigma, z, xtprev


class CategoricalTransition(nn.Module):
    def __init__(self, betas, n_class, init_prob=None):
        super().__init__()
        self.eps = 1e-30
        self.log_eps = -30
        num_classes = n_class
        self.num_classes = num_classes

        if isinstance(init_prob, str):
            if init_prob == "absorb_001":  # absorb all states into the first one
                init_prob = 0.01 * np.ones(num_classes)
                init_prob[0] = 1.0
                self.init_prob = init_prob / np.sum(init_prob)
            elif init_prob == "absorb_0001":  # absorb all states into the first one
                init_prob = 0.001 * np.ones(num_classes)
                init_prob[0] = 1.0
                self.init_prob = init_prob / np.sum(init_prob)
            elif init_prob == "absorb_00001":  # absorb all states into the first one
                init_prob = 0.0001 * np.ones(num_classes)
                init_prob[0] = 1.0
                self.init_prob = init_prob / np.sum(init_prob)
            elif init_prob == "absorb_absolute":  # absorb all states into the first one
                init_prob = 0.0 * np.ones(num_classes)
                init_prob[0] = 1.0
                self.init_prob = init_prob / np.sum(init_prob)
            elif (
                init_prob == "tomask"
            ):  # absorb all states into the the mask type (last one)
                init_prob = 0.001 * np.ones(num_classes)
                init_prob[-1] = 1.0
                self.init_prob = init_prob / np.sum(init_prob)
            elif init_prob == "uniform":
                self.init_prob = np.ones(num_classes) / num_classes
            else:
                raise NotImplementedError
        else:
            self.init_prob = init_prob / np.sum(init_prob)

        self.betas = betas
        self.num_timesteps = len(betas)

        # Construct transition matrices for q(x_t | x_{t-1})
        q_one_step_mats = [
            self._get_transition_mat(t) for t in range(0, self.num_timesteps)
        ]
        q_one_step_mats = np.stack(q_one_step_mats, axis=0)  # (T, K, K)
        self.q_onestep_mats = to_torch_const(q_one_step_mats)

        # Construct transition matrices for q(x_t | x_0)
        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats = np.stack(q_mats, axis=0)

        transpose_q_onestep_mats = np.transpose(q_one_step_mats, axes=[0, 2, 1])

        self.q_mats = to_torch_const(q_mats)
        self.transpose_q_onestep_mats = to_torch_const(transpose_q_onestep_mats)

        self.init_prob = to_torch_const(self.init_prob)
        self.init_logprob = torch.log(self.init_prob + self.eps).clamp_min(self.log_eps)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        prev_alpha_bars = np.concatenate([[1.0], alpha_bars[:-1]])
        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alpha_bars = to_torch_const(alpha_bars)
        self.prev_alpha_bars = to_torch_const(prev_alpha_bars)

    def _get_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        if self.init_prob is None:
            mat = np.full(
                shape=(self.num_classes, self.num_classes),
                fill_value=beta_t / float(self.num_classes),
                dtype=np.float64,
            )
            diag_indices = np.diag_indices_from(mat)
            diag_val = 1.0 - beta_t * (self.num_classes - 1.0) / self.num_classes
            mat[diag_indices] = diag_val
        else:
            mat = np.repeat(np.expand_dims(self.init_prob, 0), self.num_classes, axis=0)
            mat = beta_t * mat
            mat_diag = np.eye(self.num_classes) * (1.0 - beta_t)
            mat = mat + mat_diag
        return mat

    def q_vt_pred(self, log_v0, t, batch):
        # compute q(vt | v0) // actually represent v_{t+1}
        qt_mat = extract(self.q_mats, t, batch, ndim=1)
        # index_class = log_v0.argmax(dim=-1)
        # q_vt = qt_mat[torch.arange(len(index_class)), index_class]
        q_vt = torch.einsum("...i,...ij->...j", log_v0.exp(), qt_mat)
        return torch.log(q_vt + self.eps).clamp_min(self.log_eps)

    def sample_init(self, n_sample):
        init_logprob = self.init_logprob.unsqueeze(0).repeat(n_sample, 1)
        init_idx = self.sample_from_logprob(init_logprob)
        return init_idx

    def idx_to_logprob(self, x_idx):
        assert x_idx.max().item() < self.num_classes
        x_onehot = F.one_hot(x_idx, self.num_classes)
        log_x = torch.log(x_onehot.float().clamp(min=self.eps))
        return log_x

    def idx_to_prob(self, x_idx):
        return F.one_hot(x_idx, self.n_class)

    def sample_from_logprob(self, x_logprob):
        uniform = torch.rand_like(x_logprob)
        gumbel_noise = -torch.log(-torch.log(uniform + self.eps) + self.eps)
        sample_idx = (gumbel_noise + x_logprob).argmax(dim=-1)
        return sample_idx

    def sample_xt_from_x0(self, x0, timestep, batch):
        """
        q(x_t | x_0) =
            alpha_bar_t * x_0 + (1 - alpha_bar_t) * init_prob
        """
        # sample from q(vt | v0)
        log_v0 = self.idx_to_logprob(x0)
        log_q_vt_v0 = self.q_vt_pred(log_v0, timestep, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        return log_q_vt_v0, sample_class

    def sample_xtaft_from_xt(self, xt, timestep, batch):
        # single step after
        log_xt = self.idx_to_logprob(xt)
        qt = extract(self.q_onestep_mats, timestep, batch, ndim=1)  # similar q_vt_pred
        q_xtaft = torch.einsum(
            "...i,...ij->...j", log_xt.exp(), qt
        )  # similar q_vt_pred
        log_q_xtaft = torch.log(q_xtaft + self.eps).clamp_min(
            self.log_eps
        )  # similar q_vt_pred
        sample_class = log_sample_categorical(log_q_xtaft)
        return log_q_xtaft, sample_class

    def calc_posterior_and_sample(self, x0_logprob, xt, timestep, batch):
        # logprob of x0, xt (current), timestep, batch
        # get previous from x0 prediction and xt
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)

        timeiszero = timestep[batch] == 0
        xt_logprob = self.idx_to_logprob(xt)
        q_t_to_t_minus_1 = extract(
            self.transpose_q_onestep_mats, timestep, batch, ndim=1
        )
        q_0_to_t_minus_1 = extract(self.q_mats, timestep - 1, batch, ndim=1)
        fact1 = torch.einsum(
            "bj,bjk->bk", torch.exp(xt_logprob), q_t_to_t_minus_1
        )  # (batch, N)
        fact2 = torch.einsum(
            "bj,bjk->bk", torch.exp(x0_logprob), q_0_to_t_minus_1
        )  # (batch, N)

        out = torch.log(fact1 + self.eps).clamp_min(self.log_eps) + torch.log(
            fact2 + self.eps
        ).clamp_min(self.log_eps)
        out = out - torch.logsumexp(out, dim=1, keepdim=True)
        out = torch.where(
            timeiszero.unsqueeze(1).repeat(1, self.num_classes), x0_logprob, out
        )
        xtprev = self.sample_from_logprob(out)
        x0_argmax = x0_logprob.argmax(dim=1)
        xtprev = torch.where(timeiszero, x0_argmax, xtprev)
        return out, xtprev

    def get_gammas(self, t, batch):
        beta = extract(self.betas, t, batch)
        prev_alpha_bar = extract(self.prev_alpha_bars, t, batch)
        alpha_bar = extract(self.alpha_bars, t, batch)
        gamma = beta * prev_alpha_bar / (1 - alpha_bar)
        return gamma.squeeze(1)

    def calc_loss(self, pred_logprob, true_logprob, x0, timestep, batch):
        x0_logprob = self.idx_to_logprob(x0)
        kl_loss = categorical_kl(true_logprob, pred_logprob)
        decoder_nll = -log_categorical(
            x0_logprob, pred_logprob
        )  # for zeroth likelihood term
        mask = (timestep == 0).float()[batch]
        loss = mask * decoder_nll + (1 - mask) * kl_loss
        return loss

    """def calc_focal_loss(
        self, pred_logprob, true_logprob, x0, timestep, batch, alpha=0.01
    ):
        x0_logprob = self.idx_to_logprob(x0)
        loss_nonzero = focal_loss(true_logprob, pred_logprob, alpha)
        loss_zero = focal_loss(x0_logprob, pred_logprob, alpha)
        mask = (timestep == 0).float()[batch]
        loss = mask * loss_zero + (1 - mask) * loss_nonzero
        return loss"""

    def sample_xtsubseq_from_xt(self, xt, timestep, batch):
        xt_logprob = None


# Deprecated
class _CategoricalTransition(nn.Module):
    def __init__(self, betas, n_class, init_prob="uniform"):
        super().__init__()
        self.n_class = n_class
        self.eps = 1e-30
        self.log_eps = -30
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas, axis=0)
        prev_alpha_bars = np.concatenate([[1.0], alpha_bars[:-1]])
        log_alphas = np.log(alphas + self.eps)
        log_1_min_alphas = np.log(1.0 - alphas + self.eps)
        log_alpha_bars = np.log(alpha_bars + self.eps)
        log_1_min_alpha_bars = np.log(1.0 - alpha_bars + self.eps)
        log_prev_alpha_bars = np.log(prev_alpha_bars + self.eps)
        log_1_min_prev_alpha_bars = np.log(1.0 - prev_alpha_bars + self.eps)

        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alpha_bars = to_torch_const(alpha_bars)
        self.prev_alpha_bars = to_torch_const(prev_alpha_bars)
        self.log_alphas = to_torch_const(log_alphas)
        self.log_1_min_alphas = to_torch_const(log_1_min_alphas)
        self.log_alpha_bars = to_torch_const(log_alpha_bars)
        self.log_1_min_alpha_bars = to_torch_const(log_1_min_alpha_bars)
        self.log_prev_alpha_bars = to_torch_const(log_prev_alpha_bars)
        self.log_1_min_prev_alpha_bars = to_torch_const(log_1_min_prev_alpha_bars)

        if init_prob == "uniform":
            self.init_prob = to_torch_const(np.ones(n_class) / n_class)
        else:
            raise NotImplementedError
        self.init_logprob = torch.log(self.init_prob + self.eps).clamp_min(self.log_eps)

    def sample_init(self, n_sample):
        init_logprob = self.init_logprob.unsqueeze(0).repeat(n_sample, 1)
        init_idx = self.sample_from_logprob(init_logprob)
        return init_idx

    def idx_to_logprob(self, x_idx):
        assert x_idx.max().item() < self.n_class
        x_onehot = F.one_hot(x_idx, self.n_class)
        log_x = torch.log(x_onehot.float().clamp(min=self.eps))
        return log_x

    def idx_to_prob(self, x_idx):
        return F.one_hot(x_idx, self.n_class)

    def sample_from_logprob(self, x_logprob):
        uniform = torch.rand_like(x_logprob)
        gumbel_noise = -torch.log(-torch.log(uniform + self.eps) + self.eps)
        sample_idx = (gumbel_noise + x_logprob).argmax(dim=-1)
        return sample_idx

    def sample_xt_from_x0(self, x0, timestep, batch):
        """
        q(x_t | x_0) =
            alpha_bar_t * x_0 + (1 - alpha_bar_t) * init_prob
        """
        device = x0.device
        n_sample = x0.size(0)
        log_alpha_bar = extract(self.log_alpha_bars, timestep, batch)
        log_1_min_alpha_bar = extract(self.log_1_min_alpha_bars, timestep, batch)
        x0_logprob = self.idx_to_logprob(x0)
        init_logprob = self.init_logprob.unsqueeze(0).repeat(n_sample, 1).to(device)
        xt_logprob = log_add_exp(
            log_alpha_bar + x0_logprob, log_1_min_alpha_bar + init_logprob
        )
        xt = self.sample_from_logprob(xt_logprob)
        return xt_logprob, xt

    def calc_posterior_and_sample(self, x0_logprob, xt, timestep, batch):
        """
        q(x_t-1 | x_t, x_0) =
            [alpha_t * x_t + (1 - alpha_t) * init_prob] *
            [alpha_bar_t-1 * x_0 + (1 - alpha_bar_t-1) * init_prob]
        """
        device = x0_logprob.device
        n_sample = x0_logprob.size(0)
        log_alpha = extract(self.log_alphas, timestep, batch)
        log_1_min_alpha = extract(self.log_1_min_alphas, timestep, batch)
        log_prev_alpha_bar = extract(self.log_prev_alpha_bars, timestep, batch)
        log_1_min_prev_alpha_bar = extract(
            self.log_1_min_prev_alpha_bars, timestep, batch
        )

        timeiszero = timestep[batch] == 0

        xt_logprob = self.idx_to_logprob(xt)
        init_logprob = self.init_logprob.unsqueeze(0).repeat(n_sample, 1).to(device)

        left_term = log_add_exp(xt_logprob + log_alpha, log_1_min_alpha + init_logprob)
        right_term = log_add_exp(
            log_prev_alpha_bar + x0_logprob, log_1_min_prev_alpha_bar + init_logprob
        )

        xtprev_logprob = left_term + right_term
        xtprev_logprob = xtprev_logprob - torch.logsumexp(
            xtprev_logprob, dim=-1, keepdim=True
        )
        xtprev_logprob = torch.where(timeiszero, x0_logprob, xtprev_logprob)
        xtprev = self.sample_from_logprob(xtprev_logprob)
        x0_argmax = x0_logprob.argmax(dim=1)
        xtprev = torch.where(timeiszero, x0_argmax, xtprev)
        return xtprev_logprob, xtprev

    def calc_loss(self, pred_logprob, true_logprob, x0, timestep, batch):
        x0_logprob = self.idx_to_logprob(x0)
        kl_loss = categorical_kl(true_logprob, pred_logprob)
        decoder_nll = -log_categorical(x0_logprob, pred_logprob)
        mask = (timestep == 0).float()[batch]
        loss = mask * decoder_nll + (1 - mask) * kl_loss
        return loss
