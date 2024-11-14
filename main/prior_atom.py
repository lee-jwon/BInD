import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class POVMESampler:
    def __init__(self, povme_train_fn, v_sigma, n_sigma):
        self.povme_train_fn = povme_train_fn
        self.v_sigma = v_sigma
        self.n_sigma = n_sigma

        df = pd.read_csv(povme_train_fn)
        hist, xedges, yedges = np.histogram2d(
            df["v"],
            df["n"],
            bins=[1700, 60],
            range=[[0.5, 1700.5], [0.5, 60.5]],
        )
        smeared_hist = gaussian_filter(hist, sigma=[v_sigma, n_sigma])
        self.smeared_hist = smeared_hist
        self.xedges = xedges

    def sample(self, v):
        if v > 1699:
            v = 1699
        x_value = v
        x_bin_index = (
            np.digitize([x_value], self.xedges) - 1
        )  # Subtract 1 to convert to 0-based index
        marginal_distribution_y = self.smeared_hist[x_bin_index[0], :]
        y = marginal_distribution_y
        y = y / np.sum(y)
        n_sampled = np.random.choice(len(y), p=y) + 1
        if n_sampled < 7:
            n_sampled = 7
        if n_sampled > 60:
            n_sampled = 60
        return n_sampled
