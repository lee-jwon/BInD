import glob
import logging
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pprint import pformat, pprint

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import yaml
from easydict import EasyDict
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter, scatter_mean
from tqdm import tqdm

from main.dataset import RecLigDataset, rec_lig_collate_fn
from main.diffusion.beta_schedule import get_beta_schedule
from main.diffusion.transition import CategoricalTransition, ContinuousTransition
from main.loss import BaselineLoss
from main.model.model import GenDiff
from main.utils.file import recreate_directory
from main.utils.system import (
    find_free_port,
    fix_seed,
    get_cuda_visible_devices,
    text_filling,
)


def sample_time(num_graphs, num_timesteps, device):
    # sample time in balacned mode
    time_step = torch.randint(
        0, num_timesteps, size=(num_graphs // 2 + 1,), device=device
    )
    time_step = torch.cat([time_step, num_timesteps - time_step - 1], dim=0)[
        :num_graphs
    ]
    # time_step = torch.randint(0, num_timesteps, size=(num_graphs,), device=device)
    # pt = torch.ones_like(time_step).float() / num_timesteps
    return time_step, None


def process(model, data_loader, confs, optimizer=None, scaler=None):
    device = model.device
    if optimizer != None:
        is_train = True
    else:
        is_train = False

    result = {}
    st = time.time()

    n_graph = 0
    total_loss = 0.0
    (
        total_lig_h_loss,
        total_lig_x_loss,
        total_lig_e_loss,
        total_inter_e_loss,
        total_lig_h_ce_loss,
        total_lig_e_ce_loss,
        total_inter_e_ce_loss,
    ) = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    for rec_graph, lig_graph, inter_e_index, inter_e_type in tqdm(
        data_loader, disable=False
    ):
        # get batch and infos
        n_graph += rec_graph.batch.max().item() + 1
        rec_graph.to(device)
        lig_graph.to(device)
        inter_e_index = inter_e_index.to(device)
        inter_e_type = inter_e_type.to(device)

        # timestep sample
        timestep, _ = sample_time(
            n_graph, num_timesteps=confs["n_timestep"], device=device
        )

        loss, loss_dict = model(
            rec_graph, lig_graph, inter_e_index, inter_e_type, timestep, is_train
        )

        # update parameters
        if optimizer != None:  # train
            optimizer.zero_grad()
            if confs["autocast"]:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), confs["clip_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if confs["clip_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), confs["clip_grad_norm"]
                    )
                optimizer.step()

        total_lig_h_loss += loss_dict["lig_h_loss"].item()
        total_lig_x_loss += loss_dict["lig_x_loss"].item()
        total_lig_e_loss += loss_dict["lig_e_loss"].item()
        total_inter_e_loss += loss_dict["inter_e_loss"].item()
        total_lig_h_ce_loss += loss_dict["lig_h_ce_loss"].item()
        total_lig_e_ce_loss += loss_dict["lig_e_ce_loss"].item()
        total_inter_e_ce_loss += loss_dict["inter_e_ce_loss"].item()
        total_loss += loss.item()

    result["n_graph"] = n_graph
    result["lig_h_loss"] = round(total_lig_h_loss / len(data_loader), 8)
    result["lig_x_loss"] = round(total_lig_x_loss / len(data_loader), 8)
    result["lig_e_loss"] = round(total_lig_e_loss / len(data_loader), 8)
    result["inter_e_loss"] = round(total_inter_e_loss / len(data_loader), 8)
    result["lig_h_ce_loss"] = round(total_lig_h_ce_loss / len(data_loader), 8)
    result["lig_e_ce_loss"] = round(total_lig_e_ce_loss / len(data_loader), 8)
    result["inter_e_ce_loss"] = round(total_inter_e_ce_loss / len(data_loader), 8)
    result["loss"] = round(total_loss / len(data_loader), 8)
    result["n_iter"] = len(data_loader)
    result["time"] = round(time.time() - st, 2)

    return result


def main_worker(gpu, ngpus_per_node, confs):
    # set path for logging
    log_file_path = os.path.join(confs["save_dirn"], "log.log")
    # err_file_path = os.path.join(confs["save_dirn"], "err.err")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s\n%(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    # sys.stderr = open(err_file_path, "w")
    time.sleep(2)

    # log confs
    if gpu == 0:
        logging.info("train_ddp.py")
        logging.info(pformat(confs))

    # set tensorboard
    layout = {
        "model peformance": {
            "all_loss": [
                "Multiline",
                ["train/all_loss", "valid/all_loss", "test/all_loss"],
            ],
            "atom_type_loss": [
                "Multiline",
                ["train/atom_type_loss", "valid/atom_type_loss", "test/atom_type_loss"],
            ],
            "atom_type_CE_loss": [
                "Multiline",
                [
                    "train/atom_type_CE_loss",
                    "valid/atom_type_CE_loss",
                    "test/atom_type_CE_loss",
                ],
            ],
            "atom_position_loss": [
                "Multiline",
                [
                    "train/atom_position_loss",
                    "valid/atom_position_loss",
                    "test/atom_position_loss",
                ],
            ],
            "bond_type_loss": [
                "Multiline",
                ["train/bond_type_loss", "valid/bond_type_loss", "test/bond_type_loss"],
            ],
            "bond_type_CE_loss": [
                "Multiline",
                [
                    "train/bond_type_CE_loss",
                    "valid/bond_type_CE_loss",
                    "test/bond_type_CE_loss",
                ],
            ],
            "interaction_type_loss": [
                "Multiline",
                [
                    "train/interaction_type_loss",
                    "valid/interaction_type_loss",
                    "test/interaction_type_loss",
                ],
            ],
            "interaction_type_CE_loss": [
                "Multiline",
                [
                    "train/interaction_type_CE_loss",
                    "valid/interaction_type_CE_loss",
                    "test/interaction_type_CE_loss",
                ],
            ],
            "local_geometry_loss": [
                "Multiline",
                [
                    "train/local_geometry_loss",
                    "valid/local_geometry_loss",
                    "test/local_geometry_loss",
                ],
            ],
        },
        "training parameters": {"lr": ["Multiline", ["parameter/lr"]]},
    }
    if gpu == 0:
        writer = SummaryWriter(confs["save_dirn"])
        writer.add_custom_scalars(layout)

    # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    rank = gpu
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = confs["master_port"]
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=confs["n_gpu"])
    logging.info(text_filling(f"Finished Setting DDP: CUDA:{rank}"))

    # get train fns
    train_data_dirn = confs["train_data_dirn"]
    train_fns = list(glob.glob(train_data_dirn + "/*.pkl"))

    # get val fns
    val_data_dirn = confs["valid_data_dirn"]
    valid_fns = list(glob.glob(val_data_dirn + "/*.pkl"))

    # get test fns
    test_data_dirn = confs["test_data_dirn"]
    test_fns = list(glob.glob(test_data_dirn + "/*.pkl"))

    # if data debug state
    if confs["debug_data"] == "single":
        test_fns = [os.path.join(test_data_dirn, f"{confs['debug_data_id']}.pkl")]
        test_fns = test_fns * 100
        train_fns, valid_fns, test_fns = test_fns, test_fns, test_fns
        confs["bs"] = 4
        if gpu == 0:
            logging.info(f"overfitting with single data, {test_fns[0]}")
    elif confs["debug_data"] == "double":
        test_fns = test_fns[:2] * 100
        train_fns, valid_fns, test_fns = test_fns, test_fns, test_fns
        confs["bs"] = 4
        if gpu == 0:
            logging.info(f"overfitting with double data, {test_fns}")
    elif confs["debug_data"] == "10K":
        train_fns = train_fns[:10000]
        valid_fns = valid_fns * 10
        test_fns = test_fns * 10
        if gpu == 0:
            logging.info(f"training with 10K data, {len(train_fns)}")
    else:
        valid_fns = valid_fns * 30 * confs["n_gpu"]
        test_fns = test_fns * 30 * confs["n_gpu"]

    # get dataset
    train_set = RecLigDataset(
        train_fns, center_to="rec", rec_noise=confs["model"]["noise_rec_node"], pre_load=confs["pre_load_dataset"]
    )
    valid_set = RecLigDataset(valid_fns, center_to="rec", pre_load=confs["pre_load_dataset"])
    test_set = RecLigDataset(test_fns, center_to="rec", pre_load=confs["pre_load_dataset"])
    logging.info(
        f"size of actual dataset: {len(train_set)} / {len(valid_set)} / {len(test_set)}"
    )

    # sampler
    train_sampler = DistributedSampler(
        train_set, num_replicas=confs["n_gpu"], rank=rank, shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_set, num_replicas=confs["n_gpu"], rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_set, num_replicas=confs["n_gpu"], rank=rank, shuffle=True
    )

    # loader
    train_loader = DataLoader(
        train_set,
        collate_fn=rec_lig_collate_fn,
        batch_size=int(confs["bs"] / confs["n_gpu"]),
        num_workers=confs["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_set,
        collate_fn=rec_lig_collate_fn,
        batch_size=int(confs["bs"] / confs["n_gpu"]),
        num_workers=confs["num_workers"],
        pin_memory=True,
        sampler=valid_sampler,
    )
    test_loader = DataLoader(
        test_set,
        collate_fn=rec_lig_collate_fn,
        batch_size=int(confs["bs"] / confs["n_gpu"]),
        num_workers=confs["num_workers"],
        pin_memory=True,
        sampler=test_sampler,
    )

    # model
    model = GenDiff(confs)
    if confs["start_model_fn"] != None:
        if gpu == 0:
            logging.info("model loaded")
        sd = torch.load(confs["start_model_fn"], map_location="cpu")
        nsd = dict()
        for k in sd.keys():
            if "transition" in k:
                continue
            nk = k.replace("module.", "").replace("model.", "")
            nsd[nk] = sd[k]
        model.load_state_dict(nsd)
    if confs["start_dirn"] != None:
        if gpu == 0:
            logging.info("model loaded")
        sd = torch.load(
            os.path.join(confs["start_dirn"], "model/model_last.pt"), map_location="cpu"
        )
        nsd = dict()
        for k in sd.keys():
            if "transition" in k:
                continue
            nk = k.replace("module.", "").replace("model.", "")
            nsd[nk] = sd[k]
        model.load_state_dict(nsd)

    model.cuda()
    model.train()

    if gpu == 0:
        logging.info(
            f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    # beta schdeule
    h_betas = get_beta_schedule(confs["h_noise"], confs["n_timestep"])
    x_betas = get_beta_schedule(confs["x_noise"], confs["n_timestep"])
    e_betas = get_beta_schedule(confs["e_noise"], confs["n_timestep"])
    i_betas = get_beta_schedule(confs["i_noise"], confs["n_timestep"])

    # transition
    H_INIT_PROB = confs["h_prior"]  # uniform or absorb
    E_INIT_PROB = confs["e_prior"]  # uniform or absorb
    INTER_E_INIT_PROB = confs["i_prior"]  # uniform or absorb

    lig_h_transition = CategoricalTransition(
        h_betas, n_class=confs["model"]["lig_h_dim"], init_prob=H_INIT_PROB
    ).cuda()
    lig_x_transition = ContinuousTransition(x_betas).cuda()
    lig_e_transition = CategoricalTransition(
        e_betas, n_class=confs["model"]["lig_e_dim"], init_prob=E_INIT_PROB
    ).cuda()

    if not confs["abl_igen"]:
        i_betas = get_beta_schedule(confs["i_noise"], confs["n_timestep"])
        inter_e_transition = CategoricalTransition(
            i_betas, n_class=confs["model"]["inter_e_dim"], init_prob=INTER_E_INIT_PROB
        ).cuda()
    else:
        inter_e_transition = None

    transitions = [
        lig_h_transition,
        lig_x_transition,
        lig_e_transition,
        inter_e_transition,
    ]

    parallel_loss = BaselineLoss(confs, model, transitions)

    # Wrap the model
    model = DDP(
        parallel_loss, device_ids=[gpu], output_device=gpu, find_unused_parameters=True
    )
    cudnn.benchmark = True  # why needed?

    # optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=confs["lr"],
        weight_decay=confs["weight_decay"],
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=confs["factor"],
        patience=confs["patience"],
        min_lr=confs["min_lr"],
    )
    if confs["start_dirn"] != None:
        sd = torch.load(os.path.join(confs["start_dirn"], "model/optimizer_last.pt"))
        optimizer.load_state_dict(sd)
        sd = torch.load(os.path.join(confs["start_dirn"], "model/scheduler_last.pt"))
        scheduler.load_state_dict(sd)

    # Scaler (AMP)
    if confs["autocast"]:
        scaler = None
    else:
        scaler = None

    # train loop
    best_loss = 9e9
    for epoch_idx in range(1, confs["n_epoch"] + 1):
        # train
        train_loader.sampler.set_epoch(epoch_idx)
        valid_loader.sampler.set_epoch(epoch_idx)
        test_loader.sampler.set_epoch(epoch_idx)
        model.train()
        tr = process(model, train_loader, confs, optimizer, scaler)
        tr["lr"] = optimizer.param_groups[0]["lr"]
        scheduler.step(tr["loss"])

        # log tain
        if gpu == 0:
            logging.info(f"{epoch_idx} train\n{tr}")
            writer.add_scalar("train/all_loss", tr["loss"], epoch_idx)
            writer.add_scalar("train/atom_type_loss", tr["lig_h_loss"], epoch_idx)
            writer.add_scalar("train/atom_type_CE_loss", tr["lig_h_ce_loss"], epoch_idx)
            writer.add_scalar("train/atom_position_loss", tr["lig_x_loss"], epoch_idx)
            writer.add_scalar("train/bond_type_loss", tr["lig_e_loss"], epoch_idx)
            writer.add_scalar("train/bond_type_CE_loss", tr["lig_e_ce_loss"], epoch_idx)
            writer.add_scalar(
                "train/interaction_type_loss", tr["inter_e_loss"], epoch_idx
            )
            writer.add_scalar(
                "train/interaction_type_CE_loss", tr["inter_e_ce_loss"], epoch_idx
            )
            writer.add_scalar(
                "parameter/lr", optimizer.param_groups[0]["lr"], epoch_idx
            )
            writer.flush()

        # validation and test
        if epoch_idx % confs["n_valid"] == 0:
            model.eval()
            with torch.no_grad():
                vr = process(model, valid_loader, confs, scaler=scaler)
                testr = process(model, test_loader, confs, scaler=scaler)

            # log valid
            if gpu == 0:
                logging.info(f"{epoch_idx} valid\n{vr}")
                writer.add_scalar("valid/all_loss", vr["loss"], epoch_idx)
                writer.add_scalar("valid/atom_type_loss", vr["lig_h_loss"], epoch_idx)
                writer.add_scalar(
                    "valid/atom_type_CE_loss", vr["lig_h_ce_loss"], epoch_idx
                )
                writer.add_scalar(
                    "valid/atom_position_loss", vr["lig_x_loss"], epoch_idx
                )
                writer.add_scalar("valid/bond_type_loss", vr["lig_e_loss"], epoch_idx)
                writer.add_scalar(
                    "valid/bond_type_CE_loss", vr["lig_e_ce_loss"], epoch_idx
                )
                writer.add_scalar(
                    "valid/interaction_type_loss", vr["inter_e_loss"], epoch_idx
                )
                writer.add_scalar(
                    "valid/interaction_type_CE_loss", vr["inter_e_ce_loss"], epoch_idx
                )
                writer.flush()

            # log test
            if gpu == 0:
                logging.info(f"{epoch_idx} test\n{testr}")
                writer.add_scalar("test/all_loss", testr["loss"], epoch_idx)
                writer.add_scalar("test/atom_type_loss", testr["lig_h_loss"], epoch_idx)
                writer.add_scalar(
                    "test/atom_type_CE_loss", testr["lig_h_ce_loss"], epoch_idx
                )
                writer.add_scalar(
                    "test/atom_position_loss", testr["lig_x_loss"], epoch_idx
                )
                writer.add_scalar("test/bond_type_loss", testr["lig_e_loss"], epoch_idx)
                writer.add_scalar(
                    "test/bond_type_CE_loss", testr["lig_e_ce_loss"], epoch_idx
                )
                writer.add_scalar(
                    "test/interaction_type_loss", testr["inter_e_loss"], epoch_idx
                )
                writer.add_scalar(
                    "test/interaction_type_CE_loss", testr["inter_e_ce_loss"], epoch_idx
                )
                writer.flush()

            # save model for each validation loop
            if gpu == 0:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/model_{epoch_idx}.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/optimizer_{epoch_idx}.pt"),
                )
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/scheduler_{epoch_idx}.pt"),
                )

            # save the last model
            if gpu == 0:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/model_last.pt"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/optimizer_last.pt"),
                )
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(confs["save_dirn"], f"model/scheduler_last.pt"),
                )

            # save the best model, optimizer, scheduler so far
            if gpu == 0:
                if vr["loss"] < best_loss:
                    best_loss = vr["loss"]
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(confs["save_dirn"], f"model_best.pt"),
                    )
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(confs["save_dirn"], f"optimizer_best.pt"),
                    )
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(confs["save_dirn"], f"scheduler_best.pt"),
                    )
                    logging.info(f"best valid model until now")

    # clean up
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    x = torch.cuda.is_available()
    print(x)
    # load yaml
    file_path = sys.argv[1]
    with open(file_path, "r") as file:
        confs = yaml.safe_load(file)

    # recreate dirs to save (overwrite)
    if os.path.exists(confs["save_dirn"]):
        if confs["recreate_directory"]:
            print("Removing the existing directory and creating new one...")
            recreate_directory(confs["save_dirn"])
            recreate_directory(os.path.join(confs["save_dirn"], "model"))
        else:
            print("Overwriting on the existing directory...")
    else:
        print("Creating new save directory...")
        recreate_directory(confs["save_dirn"])
        recreate_directory(os.path.join(confs["save_dirn"], "model"))

    # set and copy the config file
    conf_fn = os.path.join(confs["save_dirn"], "config.yaml")
    shutil.copy(file_path, conf_fn)
    shutil.copytree("./main/", os.path.join(confs["save_dirn"], "main"))
    time.sleep(1.0)

    # environment setting for DDP
    fix_seed(0)
    confs["master_port"] = find_free_port()
    confs["distributed"] = confs["n_gpu"] > 1
    os.environ["CUDA_VISIBLE_DEVICES"] = get_cuda_visible_devices(confs["n_gpu"])

    if confs["distributed"]:
        mp.spawn(
            main_worker,
            nprocs=confs["n_gpu"],
            args=(
                confs["n_gpu"],
                confs,
            ),
        )
    else:
        main_worker(0, confs["n_gpu"], confs)
