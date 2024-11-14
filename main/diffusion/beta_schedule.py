import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0 - A1) / (sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(-k * x) + b
    # print(y)

    alphas_cumprod = y
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod


def segment_schedule(timesteps, time_segment, segment_diff):
    assert np.sum(time_segment) == timesteps
    alphas_cumprod = []
    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])
    alphas_cumprod = np.array(alphas_cumprod)

    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    return betas


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def get_beta_schedule(noise_confs, n_timestep):
    beta_schedule = noise_confs["schedule"]

    if beta_schedule == "linear":
        beta_start = noise_confs["beta_start"]
        beta_end = noise_confs["beta_end"]
        betas = np.linspace(beta_start, beta_end, n_timestep, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        beta_start = noise_confs["beta_start"]
        beta_end = noise_confs["beta_end"]
        s = 6
        betas = np.linspace(-s, s, n_timestep)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "advanced":
        betas = advance_schedule(
            timesteps=n_timestep,
            scale_start=noise_confs["start"],
            scale_end=noise_confs["end"],
            width=noise_confs["width"],
            return_alphas_bar=False,
        )
    elif beta_schedule == "cosine":
        s = noise_confs["s"]
        betas = cosine_beta_schedule(n_timestep, s=s)
    elif beta_schedule == "mix":
        betas_1 = get_beta_schedule(
            noise_confs["noise_1"], noise_confs["change_timestep"]
        )
        betas_2 = get_beta_schedule(
            noise_confs["noise_2"], n_timestep - noise_confs["change_timestep"]
        )
        betas = np.concatenate([betas_1, betas_2], axis=0)
    else:
        raise NotImplementedError

    assert betas.shape == (n_timestep,)
    return betas
