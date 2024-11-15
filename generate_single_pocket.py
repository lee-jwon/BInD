import glob
import logging
import os
import pickle
import random
import sys
import time
from pprint import pformat

import numpy as np
import torch
import yaml
from easydict import EasyDict
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from main.diffusion.beta_schedule import get_beta_schedule
from main.diffusion.transition import CategoricalTransition, ContinuousTransition
from main.diffusion.utils import *
from main.model.dataset import *
from main.model.guidance import (
    BondDistanceGuidance,
    SeparatedInterBondDistanceGuidance,
    StericClashGuidance,
    TwoHopDistanceGuidance,
)
from main.model.model import NCIVAE, GenDiff
from main.model.prior_atom import POVMESampler
from main.utils.file import recreate_directory, write_mols_to_sdf, extract_pocket, get_n_lines

from process_crossdocked import get_process 


ELSE = None
LIG_ATOM_SYMBOLS = [ELSE, "C", "N", "O", "F", "S", "P", "Cl", "Br", "I"]
LIG_BOND_TYPES = [
    ELSE,
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
INTERACTION_NAMES = [
    "SBA",
    "SBC",
    "HBD",
    "HBA",
    "HI",
    "PP",
]
TEMPFILE_DIR = "./temp"
if not os.path.exists(TEMPFILE_DIR):
    os.mkdir(TEMPFILE_DIR)


def fix_seed(seed=123, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def worker_init_fn(worker_id):
    np.random.seed(seed=123 + worker_id)
    random.seed(seed=123 + worker_id)


def generate_single_batch(
    model, transitions, batch, train_confs, confs, predictor_model=None
):
    """
    model: Trained model (nn.Module)
    transitions: List of transitions
    batch
    train_confs: config dict from trained model
    confs: config dict for generation
    predictor_model: Trained model for NCI prediction
    """
    # load transitions
    for transition in transitions:
        transition = transition.cuda()
    lig_h_transition = transitions[0]
    lig_x_transition = transitions[1]
    lig_e_transition = transitions[2]
    inter_e_transition = transitions[3]

    # load info from batch
    rec_graph, lig_graph, inter_e_index, answer_inter_e_type = batch
    n_sample = lig_graph.batch.max() + 1
    _, n_lig_node = torch.unique(lig_graph.batch, return_counts=True)
    n_lig_edge = (n_lig_node * (n_lig_node - 1) / 2).long()
    _, n_rec_node = torch.unique(rec_graph.batch, return_counts=True)
    n_inter_edge = n_lig_node * n_rec_node
    n_timestep = train_confs["n_timestep"]
    rec_graph.nl = n_lig_node  # append number of ligand nodes to rec_graph

    # construct batch matrixices
    lig_h_batch = lig_graph.batch
    lig_e_batch = lig_graph.batch[lig_graph.e_index[0]]
    inter_e_batch = lig_graph.batch[inter_e_index[1]]

    # get prior distribution
    lig_h_init = lig_h_transition.sample_init(n_lig_node.sum())
    gen_lig_h = lig_h_init.cuda()
    lig_x_init = lig_x_transition.sample_init([n_lig_node.sum(), 3]).cuda()
    gen_lig_x = lig_x_init.cuda()
    lig_e_init = lig_e_transition.sample_init(n_lig_edge.sum())
    gen_lig_e = lig_e_init.cuda()
    inter_e_init = inter_e_transition.sample_init(n_inter_edge.sum())
    gen_inter_e = inter_e_init
    gen_inter_e = gen_inter_e.cuda()

    # set answer distribution for inpainting
    if confs.given_reference_interaction == "include":
        answer_pred_inter_e = inter_e_transition.idx_to_logprob(
            answer_inter_e_type.cuda()
        )
        inter_e_inpaint_mask = (
            (answer_inter_e_type != 0).cuda().unsqueeze(1).float()
        )  # 1 to force
    elif (
        confs.given_reference_interaction == "exact"
        or confs.given_reference_interaction == "extracted"
    ):
        answer_pred_inter_e = inter_e_transition.idx_to_logprob(
            answer_inter_e_type.cuda()
        )
    elif confs.given_reference_interaction == "from_predictor":
        assert predictor_model is not None
        sampled_inter_e, _ = predictor_model.sample(rec_graph)
        answer_pred_inter_e = inter_e_transition.idx_to_logprob(sampled_inter_e.cuda())
        inter_e_inpaint_mask = (
            (answer_inter_e_type != 0).cuda().unsqueeze(1).float()
        )  # 1 to force
    elif confs.given_reference_interaction == "none":
        pass
    else:
        raise NotImplementedError


    # denoise
    for i, step in tqdm(enumerate(range(0, n_timestep)[::-1]), total=n_timestep):
        timestep = torch.full(size=(n_sample,), fill_value=step, dtype=torch.long).to(
            "cpu"
        )
        repaint_start_step, repaint_end_step = 0, n_timestep - 1
        if getattr(confs, "repaint_start_step", None) is not None:
            repaint_start_step = confs["repaint_start_step"]
        if getattr(confs, "repaint_end_step", None) is not None:
            repaint_end_step = confs["repaint_end_step"]
        if i < repaint_start_step:
            repaint_iter = 1
        elif i > repaint_end_step:
            repaint_iter = 1
        else:
            repaint_iter = confs["repaint_iter"]

        if confs["bond_distance_guidance"] is None:
            confs["bond_distance_guidance"] = 0.0
        if confs["bond_angle_guidance"] is None:
            confs["bond_angle_guidance"] = 0.0
        if confs["inter_distance_guidance"] is None:
            confs["inter_distance_guidance"] = 0.0
        if confs["steric_guidance"] is None:
            confs["steric_guidance"] = 0.0

        # adjust confs when repainting is applied
        a_bd = confs["bond_distance_guidance"] / repaint_iter
        a_ba = confs["bond_angle_guidance"] / repaint_iter
        a_id = confs["inter_distance_guidance"] / repaint_iter
        a_sc = confs["steric_guidance"] / repaint_iter

        for i_repaint in range(repaint_iter):
            (
                _,
                embded_lig_h,
                embded_lig_x,
                embded_lig_e,
                embded_inter_e,
            ) = model.propagate(
                rec_graph.h.cuda(),
                rec_graph.x.cuda(),
                rec_graph.e_index.cuda(),
                rec_graph.e_type.cuda(),
                rec_graph.batch.cuda(),
                gen_lig_h,
                gen_lig_x,
                lig_graph.e_index.cuda(),
                gen_lig_e,
                lig_graph.batch.cuda(),
                timestep.cuda(),
                inter_e_index.cuda(),
                gen_inter_e,
            )
            pred_lig_h = model.pred_lig_h_from_embded(embded_lig_h)
            pred_lig_h = torch.log_softmax(pred_lig_h, dim=1)
            pred_lig_x = embded_lig_x
            pred_lig_e = model.pred_lig_e_from_embded(embded_lig_e)
            pred_lig_e = torch.log_softmax(pred_lig_e, dim=1)
            pred_inter_e = model.pred_inter_e_from_embded(embded_inter_e)
            pred_inter_e = torch.log_softmax(pred_inter_e, dim=1)

            # reference NCI, or from predictor
            if (
                confs.given_reference_interaction == "include"
                or confs.given_reference_interaction == "from_predictor"
            ):
                pred_inter_e = (
                    inter_e_inpaint_mask * answer_pred_inter_e
                    + (1 - inter_e_inpaint_mask) * pred_inter_e
                )
            # reference NCI exact, or from extracted (dealt in dataset)
            elif (
                confs.given_reference_interaction == "exact"
                or confs.given_reference_interaction == "extracted"
            ):
                pred_inter_e = answer_pred_inter_e
            # without any NCI inpainting
            elif confs.given_reference_interaction == "none":
                pass
            else:
                raise NotImplementedError

            # bond distance guidance
            if a_bd > 0:
                with torch.enable_grad():
                    guidance = BondDistanceGuidance(
                        epsilon1=a_bd,
                        epsilon2=a_bd,
                    )
                    gen_lig_x_ = gen_lig_x.detach().requires_grad_(True)
                    gen_lig_e_ = pred_lig_e.argmax(dim=1).detach()  # \hat{e}_0
                    if gen_lig_e_.sum() > 0:
                        gui = guidance(gen_lig_x_, gen_lig_e_, lig_graph.e_index.cuda())
                        delta_x_bd = -torch.autograd.grad(gui, gen_lig_x_)[0]
                    else:
                        delta_x_bd = 0.0
            else:
                delta_x_bd = 0.0

            # bond angle guidance (using two-hop distance instead)
            if a_ba > 0:
                with torch.enable_grad():
                    guidance = TwoHopDistanceGuidance(
                        epsilon1=a_ba,
                        epsilon2=a_ba,
                    )
                    gen_lig_x_ = gen_lig_x.detach().requires_grad_(True)
                    gen_lig_e_ = pred_lig_e.argmax(dim=1).detach()  # \hat{e}_0
                    if gen_lig_e_.sum() > 0:
                        gui = guidance(gen_lig_x_, gen_lig_e_, lig_graph.e_index.cuda())
                        delta_x_ba = -torch.autograd.grad(gui, gen_lig_x_)[0]
                    else:
                        delta_x_ba = 0.0
            else:
                delta_x_ba = 0.0

            # steric collision guidance
            if a_sc > 0:
                with torch.enable_grad():
                    guidance = StericClashGuidance(epsilon=a_sc, mode="every")
                    gen_lig_x_ = gen_lig_x.detach().requires_grad_(True)
                    gui = guidance(rec_graph.x.cuda(), gen_lig_x_, inter_e_index.cuda())
                    delta_x_sc = -torch.autograd.grad(gui, gen_lig_x_)[0]
            else:
                delta_x_sc = 0.0

            # interaction distance guidance
            if not train_confs["abl_igen"] and a_id > 0:
                with torch.enable_grad():
                    guidance = SeparatedInterBondDistanceGuidance(
                        epsilon1=a_id,
                        epsilon2=a_id,
                    )
                    gen_lig_x_ = gen_lig_x.detach().requires_grad_(True)
                    gen_inter_e_ = pred_inter_e.argmax(dim=1).detach()  # \hat{i}_0
                    if gen_inter_e_.sum() > 0:  # if interaction predicted is none, pass
                        gui = guidance(
                            rec_graph.x.cuda(),
                            gen_lig_x_,
                            gen_inter_e_,
                            inter_e_index.cuda(),
                        )
                        delta_x_id = -torch.autograd.grad(gui, gen_lig_x_)[0]
                    else:
                        delta_x_id = 0.0
            else:
                delta_x_id = 0.0

            # get prevs
            prev_lig_h_prob, _ = lig_h_transition.calc_posterior_and_sample(
                pred_lig_h, gen_lig_h.cuda(), timestep.cuda(), lig_h_batch.cuda()
            )
            if train_confs["mse_train_objective"] == "data":
                _, _, _, prev_lig_x = lig_x_transition.xtprev_from_xt_x0(
                    gen_lig_x, pred_lig_x, timestep.cuda(), lig_h_batch.cuda()
                )
            elif train_confs["mse_train_objective"] == "noise":
                _, _, _, prev_lig_x = lig_x_transition.xtprev_from_xt_epsilon(
                    gen_lig_x, pred_lig_x, timestep.cuda(), lig_h_batch.cuda()
                )
            else:
                raise NotImplementedError
            prev_lig_e_prob, _ = lig_e_transition.calc_posterior_and_sample(
                pred_lig_e, gen_lig_e, timestep.cuda(), lig_e_batch.cuda()
            )
            if not train_confs["abl_igen"]:
                prev_inter_e_prob, _ = inter_e_transition.calc_posterior_and_sample(
                    pred_inter_e,
                    gen_inter_e.cuda(),
                    timestep.cuda(),
                    inter_e_batch.cuda(),
                )

            # sample
            if timestep[0] == 0:
                prev_lig_h = torch.argmax(prev_lig_h_prob, dim=1)
                prev_lig_e = torch.argmax(prev_lig_e_prob, dim=1)
                prev_inter_e = torch.argmax(prev_inter_e_prob, dim=1)
            else:
                prev_lig_h = lig_h_transition.sample_from_logprob(prev_lig_h_prob)
                prev_lig_e = lig_e_transition.sample_from_logprob(prev_lig_e_prob)
                prev_inter_e = inter_e_transition.sample_from_logprob(prev_inter_e_prob)

            # apply position guidance
            prev_lig_x = prev_lig_x + delta_x_bd + delta_x_ba + delta_x_id + delta_x_sc

            # reassign to generated
            gen_lig_h = prev_lig_h
            gen_lig_x = prev_lig_x
            gen_lig_e = prev_lig_e
            if not train_confs["abl_igen"]:
                gen_inter_e = prev_inter_e

            # if repaint, noise previous (gen prefix)
            if repaint_iter > 1 and i_repaint < repaint_iter - 1 and timestep[0] > 0:
                _, gen_lig_h = lig_h_transition.sample_xtaft_from_xt(
                    gen_lig_h, timestep.cuda() - 1, lig_h_batch.cuda()
                )
                _, _, _, gen_lig_x = lig_x_transition.xtaft_from_xt(
                    gen_lig_x, timestep.cuda() - 1, lig_h_batch.cuda()
                )
                _, gen_lig_e = lig_e_transition.sample_xtaft_from_xt(
                    gen_lig_e, timestep.cuda() - 1, lig_e_batch.cuda()
                )
                if not train_confs["abl_igen"]:
                    _, gen_inter_e = inter_e_transition.sample_xtaft_from_xt(
                        gen_inter_e, timestep.cuda() - 1, inter_e_batch.cuda()
                    )


        torch.cuda.empty_cache()

    final_gen_h_type = gen_lig_h
    final_gen_x = gen_lig_x.cpu()
    final_gen_e_type = gen_lig_e
    if not train_confs["abl_igen"]:
        final_gen_inter_e_type = gen_inter_e

    # split the graph to each batch
    gen_dict_list = []
    cuml_n_node = 0
    cuml_n_rec_node = 0
    for i in range(n_sample):
        graph_dict = {}

        gen_h = final_gen_h_type[lig_h_batch == i]
        graph_dict["h"] = gen_h

        gen_x = final_gen_x[lig_h_batch == i]
        graph_dict["x"] = gen_x

        gen_e = final_gen_e_type[lig_e_batch == i]
        graph_dict["e"] = gen_e

        gen_e_index = lig_graph.e_index[:, lig_e_batch == i] - cuml_n_node
        graph_dict["e_index"] = gen_e_index

        gen_i = final_gen_inter_e_type[inter_e_batch == i]
        graph_dict["i"] = gen_i

        gen_i_index = inter_e_index[:, inter_e_batch == i]
        gen_i_index[0] = gen_i_index[0] - cuml_n_rec_node
        gen_i_index[1] = gen_i_index[1] - cuml_n_node
        graph_dict["i_index"] = gen_i_index

        # make interaction to sparse matrix
        rec_centroid = rec_graph.centroid[i].cpu()
        if not train_confs["abl_igen"]:
            interaction_names = [
                "SBA",
                "SBC",
                "HBD",
                "HBA",
                "HI",
                "PP",
            ]  # in dimension order
            interaction_dict = {}
            for j, interaction_name in enumerate(interaction_names):
                is_inter = gen_i == j + 1
                inter_idxs = gen_i_index[:, is_inter]
                interaction_dict[interaction_name] = inter_idxs.numpy()
        else:
            interaction_dict = None

        graph_dict["i_dict"] = interaction_dict
        graph_dict["centroid"] = rec_centroid

        cuml_n_node += n_lig_node[i]
        cuml_n_rec_node += n_rec_node[i]

        gen_dict_list.append(graph_dict)

    return gen_dict_list


def graph_3d_to_rdmol(h, x, edge_index, edge_attr, one_hot=True):
    """
    Convert 3D graph into rdmol object

    INPUT:
        h
        x
        edge_index
        edge_attr
        one_hot

    OUTPUT:
        mol
    """
    if one_hot:  # one-hot to index
        h = h.argmax(dim=1).detach()
        edge_attr = edge_attr.argmax(dim=1).detach()  # 얘는 왜 detach 안해줘?

    h = h.long()
    edge_attr = edge_attr.long()

    atom_smbols = [LIG_ATOM_SYMBOLS[idx] for idx in h]
    mask = h == 0
    mol = Chem.EditableMol(Chem.Mol())
    old_to_new = [0] * len(h)
    new_idx = 0
    for old_idx, atom_smbol in enumerate(atom_smbols):
        if atom_smbol is not None:
            atom = Chem.Atom(atom_smbol)
            mol.AddAtom(atom)
            old_to_new[old_idx] = new_idx  # old_to_new[old_idx] = new_idx
            new_idx += 1
    for bond_idx in range(edge_attr.size(0)):
        bond_type = edge_attr[bond_idx]
        bond_type = LIG_BOND_TYPES[bond_type]
        i, j = edge_index[0, bond_idx].item(), edge_index[1, bond_idx].item()
        if not mask[i] and not mask[j]:
            if bond_type is not None:
                mol.AddBond(old_to_new[i], old_to_new[j], bond_type)

    rw_mol = Chem.RWMol(mol.GetMol())
    conf = Chem.Conformer(rw_mol.GetNumAtoms())
    for idx, pos in enumerate(x):
        if not mask[idx]:
            pos = pos.tolist()
            conf.SetAtomPosition(old_to_new[idx], Chem.rdGeometry.Point3D(*pos))
    rw_mol.AddConformer(conf)
    return rw_mol.GetMol()


def prepare_input_data(receptor_fn, protein_fn=None, ligand_fn=None, extract=True):
    if extract:
        assert protein_fn is not None and ligand_fn is not None
        assert os.path.exists(protein_fn) and os.path.exists(ligand_fn)
        # 1. Extract pocket
        extract_pocket(ligand_fn, protein_fn, receptor_fn)
    else:
        assert os.path.exists(receptor_fn)

    # 2. Run POVME
    os.chdir(TEMPFILE_DIR) # ./temp
    cmnd = (
        f"python ../POVME/POVME_pocket_id.py --filename ../{receptor_fn} --processors 8"
    )
    os.system(cmnd)
    povme_fn = "./pocket1.pdb"
    os.system("pwd")
    n_povme = get_n_lines(povme_fn)
    # n_lig = Chem.SDMolSupplier(ligand_fn)[0].GetNumAtoms()
    # os.remove(r"./pocket[0-9].pdb")
    os.chdir("..")

    # 3. Data processing
    input_data = get_process(receptor_fn, ligand_fn, filter=False)
    input_data["v"] = n_povme
    input_data["n"] = input_data["lig"]["x"].shape[0]
    return input_data


if __name__ == "__main__":
    # get confs
    file_path = sys.argv[1]
    with open(file_path, "r") as file:
        confs = yaml.safe_load(file)
        confs = EasyDict(confs)

    recreate_directory(confs["save_dirn"])
    confs["save_mol_dirn"] = os.path.join(confs["save_dirn"], "gen")
    recreate_directory(confs["save_mol_dirn"])
    conf_fn = os.path.join(confs["save_dirn"], "config.yaml")

    os.system(f"cp -r {file_path} {conf_fn}")
    time.sleep(2)

    # set logger
    log_file_path = os.path.join(confs["save_dirn"], "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s - %(levelname)s]\n%(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    logging.info(pformat(confs))
    time.sleep(2)

    # fix seed
    if confs["seed"] is not None:
        fix_seed(confs["seed"])
        logging.info(f"seed fixed to {confs.seed}")

    train_confs_fn = os.path.join(confs["model_dirn"], "config.yaml")
    with open(train_confs_fn, "r") as file:
        train_confs = yaml.safe_load(file)
    train_confs = EasyDict(train_confs)

    # prior atom sampler
    if confs.prior_atom.method == "ref":
        prior_atom_sampler = "ref"
    elif confs.prior_atom.method == "povme":
        prior_atom_sampler = POVMESampler(
            confs.prior_atom.povme_train_result_fn,
            confs.prior_atom.povme_v_sigma,
            confs.prior_atom.povme_n_sigma,
        )
    else:
        raise NotImplementedError
    logging.info(f"prior atom sampler loaded: {prior_atom_sampler}")
    time.sleep(2)

    # prepare input
    if confs["receptor_fn"] is not None:
        # input_data = prepare_input_data(confs["receptor_fn"], extract=False)
        input_data = prepare_input_data(
            receptor_fn=confs["receptor_fn"],
            protein_fn=None,
            ligand_fn=confs["ligand_fn"],
            extract=False
        )
    else:
        receptor_fn = os.path.join(
            TEMPFILE_DIR, 
            os.path.basename(confs["protein_fn"])[:-4] + "_poc.pdb",
        )
        input_data = prepare_input_data(
            receptor_fn,
            protein_fn=confs["protein_fn"],
            ligand_fn=confs["ligand_fn"],
            extract=True
        )
    input_data["my_key"] = 0

    logging.info(f"Generating with receptor: {input_data['rec_fn']} and ligand: {input_data['lig_fn']}")
    test_set = RecLigDataset(
        [input_data] * confs["bs"],
        center_to="rec",
        rec_noise=0.0,
        mode=confs.prior_atom.method,
        is_dict=True,
    )
    test_set.append_sampler(prior_atom_sampler)

    # loader
    test_loader = DataLoader(
        test_set,
        collate_fn=rec_lig_collate_fn,
        batch_size=confs["bs"],
        num_workers=0,  # set 0 to prevent overhead
        shuffle=False,
        worker_init_fn=worker_init_fn,
    )

    # load model
    model = GenDiff(train_confs)
    model.cuda()
    model.eval()
    if confs["model_cut"] == "best":
        sd = torch.load(os.path.join(confs["model_dirn"], "model_best.pt"))
    elif confs["model_cut"] == "last":
        sd = torch.load(os.path.join(confs["model_dirn"], "model/model_last.pt"))
    else:  # model_cut must be a specific epoch
        sd = torch.load(
            os.path.join(confs["model_dirn"], f"model/model_{confs['model_cut']}.pt")
        )

    # clear keys of state dict due to DDP
    nsd = dict()
    for k in sd.keys():
        if "transition" in k:
            continue
        nk = k.replace("module.", "").replace("model.", "")
        nsd[nk] = sd[k]
    a = model.load_state_dict(nsd, strict=False)
    model.eval()
    logging.info(f"{a}")
    logging.info("model loaded")
    time.sleep(2)

    # beta schedule and transitions for diffusion
    h_betas = get_beta_schedule(train_confs["h_noise"], train_confs["n_timestep"])
    lig_h_transition = CategoricalTransition(
        h_betas,
        n_class=train_confs["model"]["lig_h_dim"],
        init_prob=train_confs["h_prior"] ,
    )
    x_betas = get_beta_schedule(train_confs["x_noise"], train_confs["n_timestep"])
    lig_x_transition = ContinuousTransition(x_betas)
    e_betas = get_beta_schedule(train_confs["e_noise"], train_confs["n_timestep"])
    lig_e_transition = CategoricalTransition(
        e_betas,
        n_class=train_confs["model"]["lig_e_dim"],
        init_prob=train_confs["e_prior"] ,
    )
    i_betas = get_beta_schedule(train_confs["i_noise"], train_confs["n_timestep"])
    inter_e_transition = CategoricalTransition(
        i_betas,
        n_class=train_confs["model"]["inter_e_dim"],
        init_prob=train_confs["i_prior"],
    )
    transitions = [
        lig_h_transition,
        lig_x_transition,
        lig_e_transition,
        inter_e_transition,
    ]

    # generate loop
    for n_gen_idx in range(confs.n_generate // confs.bs):
        test_idx = 0
        for _, batch in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                gen_dict_list = generate_single_batch(
                    model, transitions, batch, train_confs, confs, None
                )

                for gen_dict in gen_dict_list:
                    h_type, x, e_index, e_type, rec_centroid, interaction_dict = (
                        gen_dict["h"],
                        gen_dict["x"],
                        gen_dict["e_index"],
                        gen_dict["e"],
                        gen_dict["centroid"],
                        gen_dict["i_dict"],
                    )
                    x = x + rec_centroid

                    # test_fn = test_fns[test_idx]
                    ori_data_graph = test_set[test_idx]

                    gen_dirn = confs["save_mol_dirn"]
                    if not os.path.exists(gen_dirn):
                        os.makedirs(gen_dirn)

                    mol = graph_3d_to_rdmol(
                        h_type,
                        x,
                        e_index,
                        e_type,
                        one_hot=False,
                    )

                    gen_fn = os.path.join(
                        gen_dirn, f"gen_{n_gen_idx * confs.bs + test_idx + 1}.sdf"
                    )
                    write_mols_to_sdf([mol], gen_fn)
                    #gen_inter_fn = os.path.join(
                    #    gen_dirn, f"gen_{n_gen_idx * confs.bs + test_idx + 1}.pkl"
                    #)
                    #with open(gen_inter_fn, "wb") as f:
                    #    pickle.dump(interaction_dict, f)

                    test_idx += 1

                    logging.info(Chem.MolToSmiles(mol))
