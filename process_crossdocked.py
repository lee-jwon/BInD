import argparse
import os
import pickle
import random
import secrets
import string
import sys
import tempfile
import time
import warnings
from copy import deepcopy
from itertools import product
from multiprocessing import process
from pprint import pprint

import numpy as np
import parmap
import torch
import yaml
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SanitizeMol
from tqdm import tqdm

from plip.structure.preparation import Mol, PDBComplex

warnings.simplefilter("ignore", BiopythonWarning)
RDLogger.DisableLog("rdApp.*")

from torch_geometric.data import Data

from main.utils.file import read_mol_file, recreate_directory, write_mols_to_sdf

# Directories
TEMPFILE_DIRN = "./temp/"
PARMAP_OR_MP = "parmap"  # "parmap", "mp"
N_PROCESS = 16

# Flags
RECREATE_DATA = True  # if False, run() do not recreate existing data
LIG_USE_AROMATIC = True  # if False, do not use aromatic bond (will use double and single bonds alternating)

# Ligand atom features
ELSE = None
LIG_ATOM_SYMBOLS = [
    ELSE,
    "C",
    "N",
    "O",
    "F",
    "S",
    "P",
    "Cl",
    "Br",
    "I",
]  # if not included, abort
LIG_ATOM_CHARGES = [-2, -1, 0, 1, 2]
LIG_ATOM_NUMHS = [0, 1, 2, 3, 4]
LIG_BOND_TYPES = [ELSE, "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

# Receptor C_a feature
REC_AMINO_ACIDS = [
    ELSE,
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

# Receptor atom features
REC_ATOM_SYMBOLS = [ELSE, "C", "N", "O", "S"]
REC_ATOM_CHARGES = [-2, -1, 0, 1, 2]
REC_ATOM_NUMHS = [0, 1, 2, 3, 4, 5, 6]
REC_BOND_TYPES = [ELSE, "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def get_aromatic_atom_indices(mol) -> np.ndarray:
    aromatic_indice = []
    natoms = mol.GetNumAtoms()
    for atom_idx in range(natoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetIsAromatic():
            aromatic_indice += [atom_idx]
    aromatic_indice = np.array(aromatic_indice)
    return aromatic_indice


def get_anion_atom_indices(mol) -> np.ndarray:
    anion_indice = []
    for i, atom in enumerate(mol.GetAtoms()):
        x = get_amino_acid_name(atom)
        if x in ["GLU"]:
            if atom.GetSymbol() in [
                "O"
            ] and atom.GetPDBResidueInfo().GetName().strip() in ["OE1", "OE2"]:
                anion_indice.append(i)
        elif x in ["ASP"]:
            if atom.GetSymbol() in [
                "O"
            ] and atom.GetPDBResidueInfo().GetName().strip() in ["OD1", "OD2"]:
                anion_indice.append(i)
        else:
            continue
    anion_indice = np.array(anion_indice)
    return anion_indice


def filter_ligand(lig_mol, num_atom_min_max=[6, 60], avail_atom_types=LIG_ATOM_SYMBOLS):
    # return True
    if lig_mol is None:
        return False
    if len(lig_mol.GetConformers()) == 0:
        return False
    num_atom = lig_mol.GetNumAtoms()
    if num_atom < num_atom_min_max[0] or num_atom > num_atom_min_max[1]:
        return False
    for atom in lig_mol.GetAtoms():
        if atom.GetSymbol() not in avail_atom_types:
            return False
    return True


def filter_receptor(
    rec_mol,
    num_atom_min_max=[140, 660],
):
    # return True
    if rec_mol is None:
        return False
    if len(rec_mol.GetConformers()) == 0:
        return False
    num_atom = rec_mol.GetNumAtoms()
    if num_atom < num_atom_min_max[0] or num_atom > num_atom_min_max[1]:
        return False
    return True


def radius_edges(points, radius, self_loop=False):
    """
    INPUT:
        points (list)  : list of point vectors
        radius (float) : radius cutoff for getting edges
    OUTPUT:
        2 X E numpy array

    Same as torch_geometric.nn.radius_graph(points, radius)

    WONHO: for loop 안쓰게 수정함
    """
    N = points.shape[0]
    x1 = points.reshape(-1, N, 3)
    x2 = points.reshape(N, -1, 3)
    distance = np.linalg.norm(x1 - x2, axis=-1)
    adj = np.where(distance <= radius, 1, 0)
    src, tar = np.triu(adj, k=int(not self_loop)).nonzero()  # k=0 if self_loop=True
    return np.stack([src, tar], axis=0)


def read_pdb_file(file_path):
    with open(file_path, "r") as f:
        pdb_block = f.read()
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
    mol = Chem.rdmolops.RemoveAllHs(mol, sanitize=False)
    # mol = Chem.RemoveHs(mol, implicitOnly=False)
    Chem.rdmolops.SanitizeMol(
        mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
    )
    return mol


def is_c_alpha(atom):
    return (
        atom.GetAtomicNum() == 6 and atom.GetPDBResidueInfo().GetName().strip() == "CA"
    )


def get_amino_acid_name(atom):
    return atom.GetPDBResidueInfo().GetResidueName()


def join_complex(ligand_fn, pocket_fn, complex_fn=None):
    """
    Join ligand file and pocket file to obtain complex file, while ligand ligname is
    forced to LIG for later processing
    """
    if complex_fn is None:
        fd, complex_fn = tempfile.mkstemp(
            suffix=".pdb",
            prefix="temp_process_crossdocked_plip_input_",
            dir=TEMPFILE_DIRN,
        )

    lig_mol = Chem.SDMolSupplier(ligand_fn, sanitize=False)[0]
    try:
        lig_mol = Chem.RemoveHs(lig_mol)
    except:
        pass

    if not LIG_USE_AROMATIC:  # not use aromatic
        Chem.Kekulize(lig_mol, clearAromaticFlags=True)
    else:  # use aromatic
        Chem.rdmolops.SanitizeMol(
            lig_mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
        )
    num_ligand_atom = lig_mol.GetNumAtoms()

    command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d 2> /dev/null"
    os.system(command)
    with open(complex_fn, "r") as f:
        lines = f.readlines()
    new_lines = []
    for i, line in enumerate(lines):
        if i > 1 and i < num_ligand_atom + 2:
            new_line = (
                line[:17] + "LIG" + line[20:25] + "1 " + line[27:]
            )  # enforce lig_resname as LIG
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(complex_fn, "w") as f:
        f.writelines(new_lines)
    complex_mol = Chem.MolFromPDBFile(complex_fn, sanitize=False)
    return complex_mol, complex_fn


def get_complex_interaction_info(complex_fn):  # Receptor - Ligand
    """
    From protein-ligand complex file, protein-ligand interaction is profiled via PLIP.
    OUTPUT: (dictionary)
        "salt_bridge_anion"
        "salt_bridge_cation"
        "hydrogen_bond_donor"
        "hydrogen_bond_acceptor"
        "hydrogen_bond_donor"
        "hydrophobic_interaction"
        "pi_stacking"
    """
    out = {}
    my_mol = PDBComplex()
    my_mol.load_pdb(complex_fn)
    ligs = [
        ":".join([x.hetid, x.chain, str(x.position)])
        for x in my_mol.ligands
        if x.hetid == "LIG"
    ]
    if len(ligs) == 0:
        return
    my_mol.analyze()
    my_interactions = my_mol.interaction_sets[ligs[0]]

    anions = my_interactions.saltbridge_pneg
    cations = my_interactions.saltbridge_lneg
    hbds = my_interactions.hbonds_pdon
    hbas = my_interactions.hbonds_ldon
    hydros = my_interactions.hydrophobic_contacts
    pipis = my_interactions.pistacking

    # 0 nontype

    # 1 salt brdige (protein = anion)
    pairs = []
    for x in anions:  # for each interactions
        rec_idx = [x - 1 for x in x.negative.atoms_orig_idx]
        lig_idx = [x - 1 for x in x.positive.atoms_orig_idx]
        pairs += list(product(rec_idx, lig_idx))
    out["SBA"] = pairs

    # 2 salt bridge (protein = cation)
    pairs = []
    for x in cations:  # for each interactions
        rec_idx = [x - 1 for x in x.positive.atoms_orig_idx]
        lig_idx = [x - 1 for x in x.negative.atoms_orig_idx]
        pairs += list(product(rec_idx, lig_idx))
    out["SBC"] = pairs

    # 3 hydrogen bond (protein = donor)
    pairs = []
    for x in hbds:
        pairs += [(x.d_orig_idx - 1, x.a_orig_idx - 1)]
    out["HBD"] = pairs

    # 4 hydrogen bond (protein = acceptor)
    pairs = []
    for x in hbas:
        pairs += [(x.a_orig_idx - 1, x.d_orig_idx - 1)]
    out["HBA"] = pairs

    # 5 hydrophobic interactions
    pairs = []
    for hyd in hydros:
        pairs += [(hyd.bsatom_orig_idx - 1, hyd.ligatom_orig_idx - 1)]
    out["HI"] = pairs

    # 6 pipi (many to many) #TODO
    pairs = []
    for x in pipis:  # for each interactions
        rec_idx = [x - 1 for x in x.proteinring.atoms_orig_idx]
        lig_idx = [x - 1 for x in x.ligandring.atoms_orig_idx]
        pairs += list(product(rec_idx, lig_idx))
    out["PP"] = pairs

    return out


def sanitize_pocket_interaction(inter_dict, ligand_n):
    """
    Since the indices obtained by PLIP counts from ligand to receptor,
    thus receptor atom indices should be subtracted by the number of ligand atoms.
    """
    for k, v in inter_dict.items():
        inter_dict.update({k: [[p - ligand_n, l] for p, l in v]})
    return inter_dict


def process_ligand(lig_mol):
    assert filter_ligand(lig_mol), "ligand filtered out!"
    out = {}

    # process ligand mol node
    lig_h_type = [LIG_ATOM_SYMBOLS.index(x.GetSymbol()) for x in lig_mol.GetAtoms()]
    lig_h_charge = [
        LIG_ATOM_CHARGES.index(x.GetFormalCharge()) for x in lig_mol.GetAtoms()
    ]
    lig_h_numh = [LIG_ATOM_NUMHS.index(x.GetTotalNumHs()) for x in lig_mol.GetAtoms()]
    lig_x = lig_mol.GetConformer().GetPositions()
    assert not np.isnan(lig_x).any()
    out["h_type"] = np.array(lig_h_type)
    out["h_charge"] = np.array(lig_h_charge)
    out["h_numH"] = np.array(lig_h_numh)
    out["x"] = lig_x

    out["centroid"] = np.mean(lig_x, axis=0)

    # process ligand mol edge
    ad = Chem.GetDistanceMatrix(lig_mol)  # Topological distance
    e1, e2 = [], []  # Edge index
    edge_attr = []  # Edge feature
    e_hop = []  # Hop between start, end atoms
    for i in range(lig_mol.GetNumAtoms()):
        for j in range(i + 1, lig_mol.GetNumAtoms()):
            bond = lig_mol.GetBondBetweenAtoms(i, j)
            if bond != None:
                bond = str(bond.GetBondType())
                e_hop.append(ad[i, j])
            else:
                e_hop.append(ad[i, j])
            b = LIG_BOND_TYPES.index(bond)
            e1.append(i), e2.append(j)
            edge_attr.append(b)
    out["e_index"] = np.array([e1, e2])
    out["e_type"] = np.array(edge_attr)
    out["e_hop"] = np.array(e_hop).astype(int)

    assert not np.isnan(out["centroid"]).any()
    assert out["h_type"].shape[0] == out["x"].shape[0]
    assert np.all(out["x"] < 1000), "Not all elements are less than 1000"
    assert np.all(out["x"] > -1000), "Not all elements are less than 1000"
    assert not np.isnan(out["centroid"]).any()
    assert out["e_index"].shape[1] == out["e_type"].shape[0]

    return out


def process_receptor(rec_mol, radius_cutoff=8.0, filter=True):
    if filter:
        assert filter_receptor(rec_mol), "receptor filtered out!"
    out = {}
    # print("num atom", rec_mol.GetNumAtoms())

    # process receptor mol node
    # rec_surf_rho = get_surface_aware_node_feature(rec_mol)

    rec_h_aa, rec_h_type, rec_h_charge, rec_h_numh, rec_h_is_ca, rec_h_surf = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for atom_idx in range(rec_mol.GetNumAtoms()):
        atom = rec_mol.GetAtomWithIdx(atom_idx)
        h_aa = (
            REC_AMINO_ACIDS.index(atom.GetPDBResidueInfo().GetResidueName())
            if atom.GetPDBResidueInfo().GetResidueName() in REC_AMINO_ACIDS
            else 0
        )
        h_type = (
            REC_ATOM_SYMBOLS.index(atom.GetSymbol())
            if atom.GetSymbol() in REC_ATOM_SYMBOLS
            else 0
        )
        h_charge = REC_ATOM_CHARGES.index(atom.GetFormalCharge())
        h_numh = REC_ATOM_NUMHS.index(atom.GetTotalNumHs())
        h_is_ca = is_c_alpha(atom)
        # h_surf = rec_surf_rho[atom_idx]

        rec_h_aa.append(h_aa)
        rec_h_type.append(h_type)
        rec_h_charge.append(h_charge)
        rec_h_numh.append(h_numh)
        rec_h_is_ca.append(h_is_ca)
        # rec_h_surf.append(h_surf)

    rec_x = rec_mol.GetConformer().GetPositions()
    assert not np.isnan(rec_x).any()

    out["h_aa"] = np.array(rec_h_aa)
    out["h_type"] = np.array(rec_h_type)
    out["h_charge"] = np.array(rec_h_charge)
    out["h_numH"] = np.array(rec_h_numh)
    out["h_isCA"] = np.array(rec_h_is_ca)
    # out["h_surf"] = np.array(rec_h_surf)
    out["x"] = rec_x

    out["centroid"] = np.mean(rec_x, axis=0)

    if True:  # radius edge
        e_index = radius_edges(rec_x, radius=radius_cutoff)
        e_attr = []
        e_adj = []
        for p in e_index.T.tolist():
            i, j = p
            bond = rec_mol.GetBondBetweenAtoms(i, j)
            if bond != None:
                bond = str(bond.GetBondType())
                e_adj.append(1)
            else:
                e_adj.append(0)
            e_attr.append(REC_BOND_TYPES.index(bond))
        out["e_index"] = e_index.astype(int)
        out["e_type"] = np.array(e_attr).astype(int)
        out["e_adj"] = np.array(e_adj).astype(bool)
    else:  # real bond edge
        e_index_1 = []
        e_index_2 = []
        e_attr = []
        e_adj = []
        for bond in rec_mol.GetBonds():
            bond_type = str(bond.GetBondType())
            e_attr.append(REC_BOND_TYPES.index(bond_type))
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if start_idx > end_idx:
                t = start_idx
                start_idx = end_idx
                end_idx = t
            e_index_1.append(start_idx)
            e_index_2.append(end_idx)
        e_index = np.array([e_index_1, e_index_2])
        out["e_index"] = e_index.astype(int)
        out["e_type"] = np.array(e_attr).astype(int)
        # out["e_adj"] = np.array(e_adj).astype(bool)

    assert not np.isnan(out["centroid"]).any()
    assert out["h_type"].shape[0] == out["x"].shape[0]
    assert np.all(out["x"] < 1000)
    assert np.all(out["x"] > -1000)
    assert not np.isnan(out["centroid"]).any()
    assert out["e_index"].shape[1] == out["e_type"].shape[0]
    # assert out["e_index"].shape[1] == out["e_adj"].shape[0]

    return out

def get_process(receptor_fn, ligand_fn, filter=False):

    out = {
        "rec_fn": receptor_fn,
        "lig_fn": ligand_fn
    }

    rec_mol = read_pdb_file(out["rec_fn"])
    lig_mol = Chem.SDMolSupplier(out["lig_fn"], sanitize=False)[0]
    try:
        lig_mol = Chem.RemoveHs(lig_mol)
    except:
        pass
    assert rec_mol != None, "rec mol is none"
    assert lig_mol != None, "lig mol is none"

    if not LIG_USE_AROMATIC:  # not use aromatic
        Chem.Kekulize(lig_mol, clearAromaticFlags=True)
    else:  # use aromatic
        Chem.rdmolops.SanitizeMol(
            lig_mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
        )

    out["lig"] = process_ligand(lig_mol)
    out["rec"] = process_receptor(rec_mol, filter=filter)
    n_lig, n_rec = lig_mol.GetNumAtoms(), rec_mol.GetNumAtoms()

    time.sleep(0.5)

    # process interaction
    _, cmplx_fn = join_complex(out["lig_fn"], out["rec_fn"])
    inter_info = get_complex_interaction_info(cmplx_fn)
    inter_info = sanitize_pocket_interaction(inter_info, n_lig)

    time.sleep(0.5)
    # remove temp file
    os.remove(cmplx_fn)

    # make interaction array with index
    out["inter"] = {}
    rec_to_lig_index = [[t1, t2] for t1 in range(n_rec) for t2 in range(n_lig)]
    rec_to_lig_index = np.array(rec_to_lig_index).T
    rec_to_lig_type = np.zeros([rec_to_lig_index.shape[1]]).astype(int)
    for i, key in enumerate(inter_info):
        for rec_idx, lig_idx in inter_info[key]:
            rec_to_lig_type[n_lig * rec_idx + lig_idx] = i + 1  # 0: non-intr
    out["inter"]["rec_to_lig_index"] = rec_to_lig_index.astype(int)
    out["inter"]["rec_to_lig_type"] = rec_to_lig_type.astype(int)
    assert (
        out["inter"]["rec_to_lig_index"].shape[1]
        == out["inter"]["rec_to_lig_type"].shape[0]
    )
    assert out != None
    return out

def run_process(input):
    sample, save_dir = input
    data_fn = f"{sample['my_key']}.pkl"

    if not RECREATE_DATA and os.path.exists(os.path.join(save_dir, data_fn)):
        return

    out = deepcopy(sample)

    rec_mol = read_pdb_file(sample["rec_fn"])
    lig_mol = Chem.SDMolSupplier(sample["lig_fn"], sanitize=False)[0]
    try:
        lig_mol = Chem.RemoveHs(lig_mol)
    except:
        pass
    assert rec_mol != None, "rec mol is none"
    assert lig_mol != None, "lig mol is none"

    if not LIG_USE_AROMATIC:  # not use aromatic
        Chem.Kekulize(lig_mol, clearAromaticFlags=True)
    else:  # use aromatic
        Chem.rdmolops.SanitizeMol(
            lig_mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
        )

    out["lig"] = process_ligand(lig_mol)
    out["rec"] = process_receptor(rec_mol)
    n_lig, n_rec = lig_mol.GetNumAtoms(), rec_mol.GetNumAtoms()

    time.sleep(0.5)

    # process interaction
    _, cmplx_fn = join_complex(sample["lig_fn"], sample["rec_fn"])
    inter_info = get_complex_interaction_info(cmplx_fn)
    inter_info = sanitize_pocket_interaction(inter_info, n_lig)

    time.sleep(0.5)
    # remove temp file
    os.remove(cmplx_fn)

    # make interaction array with index
    out["inter"] = {}
    rec_to_lig_index = [[t1, t2] for t1 in range(n_rec) for t2 in range(n_lig)]
    rec_to_lig_index = np.array(rec_to_lig_index).T
    rec_to_lig_type = np.zeros([rec_to_lig_index.shape[1]]).astype(int)
    for i, key in enumerate(inter_info):
        for rec_idx, lig_idx in inter_info[key]:
            rec_to_lig_type[n_lig * rec_idx + lig_idx] = i + 1  # 0: non-intr
    out["inter"]["rec_to_lig_index"] = rec_to_lig_index.astype(int)
    out["inter"]["rec_to_lig_type"] = rec_to_lig_type.astype(int)
    assert (
        out["inter"]["rec_to_lig_index"].shape[1]
        == out["inter"]["rec_to_lig_type"].shape[0]
    )

    # save
    assert out != None
    with open(os.path.join(save_dir, data_fn), "wb") as f:
        pickle.dump(out, f)
    return


def run_try(*inputs):
    try:
        x = run_process(*inputs)
    except Exception as e:
        emsg = f"{e} \n {inputs} \n"
        print(emsg, flush=True)  # for debugging
        return inputs
    return True


def multiprocessing_wrap(function, inputs, ncpu):
    if ncpu == 0:
        # for debugging
        function(inputs[0])
        return

    from multiprocessing import Pool

    pool = Pool(ncpu)
    r = pool.map_async(function, inputs)
    r.wait()
    pool.close()
    pool.join()
    return


if __name__ == "__main__":
    # all dirs should be given as relative path
    # split_fn = "./data/split/pocket.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--split_fn", type=str, default="./data/split/sequence.pt")
    parser.add_argument("--save_dirn", type=str)
    parser.add_argument("--raw_dirn", type=str)
    parser.add_argument("--tempfile_dirn", type=str, default="./temp/")
    parser.add_argument("--only_test", action="store_true", default=False)

    args = parser.parse_args()

    print(args)

    split_fn = args.split_fn
    TEMPFILE_DIRN = args.tempfile_dirn
    RECREATE_DATA = args.recreate
    data_dirn = args.raw_dirn
    train_save_dirn = os.path.join(args.save_dirn, "train")
    valid_save_dirn = os.path.join(args.save_dirn, "valid")
    test_save_dirn = os.path.join(args.save_dirn, "test")

    # recreate
    if RECREATE_DATA:
        print("Recreating directories...")
        recreate_directory(train_save_dirn)
        recreate_directory(valid_save_dirn)
        recreate_directory(test_save_dirn)

    # read split file
    print("Reading key file...")
    split_names = torch.load(split_fn)
    train_pairs = split_names["train"]
    valid_pairs = split_names["val"]
    test_pairs = split_names["test"]

    # make train and test data
    print("Make train, valid, and test data keys...")
    train_data = []
    for i, (rec_fn, lig_fn) in enumerate(train_pairs):
        sample = {}
        sample["rec_fn_tail"] = rec_fn
        sample["rec_fn"] = os.path.join(data_dirn, rec_fn)
        sample["lig_fn_tail"] = lig_fn
        sample["lig_fn"] = os.path.join(data_dirn, lig_fn)
        sample["my_key"] = i + 1
        train_data.append(sample)
    valid_data = []
    for i, (rec_fn, lig_fn) in enumerate(valid_pairs):
        sample = {}
        sample["rec_fn_tail"] = rec_fn
        sample["rec_fn"] = os.path.join(data_dirn, rec_fn)
        sample["lig_fn_tail"] = lig_fn
        sample["lig_fn"] = os.path.join(data_dirn, lig_fn)
        sample["my_key"] = i + 1
        valid_data.append(sample)
    test_data = []
    for i, (rec_fn, lig_fn) in enumerate(test_pairs):
        sample = {}
        sample["rec_fn_tail"] = rec_fn
        sample["rec_fn"] = os.path.join(data_dirn, rec_fn)
        sample["lig_fn_tail"] = lig_fn
        sample["lig_fn"] = os.path.join(data_dirn, lig_fn)
        sample["my_key"] = i + 1
        test_data.append(sample)

    # process test data
    print("test data raw", len(test_data))
    test_process_input = [(sample, test_save_dirn) for sample in test_data]
    if PARMAP_OR_MP == "mp":
        multiprocessing_wrap(run_process, test_process_input, 10)
    elif PARMAP_OR_MP == "parmap":
        test_result = parmap.map(
            run_try, test_process_input, pm_pbar=True, pm_processes=10
        )
        print("test data processed", list(test_result).count(True))

        # process valid data
        print("valid data raw", len(valid_data))
        valid_process_input = [(sample, valid_save_dirn) for sample in valid_data]
        if PARMAP_OR_MP == "mp":
            multiprocessing_wrap(run_process, valid_process_input, 10)
        elif PARMAP_OR_MP == "parmap":
            valid_result = parmap.map(
                run_try, valid_process_input, pm_pbar=True, pm_processes=10
            )
            print("valid data processed", list(valid_result).count(True))

    if not args.only_test:
        # process train data
        print("train data raw", len(train_data))
        train_process_input = [(sample, train_save_dirn) for sample in train_data]
        if PARMAP_OR_MP == "mp":
            multiprocessing_wrap(run_process, train_process_input, N_PROCESS)
        elif PARMAP_OR_MP == "parmap":
            train_result = parmap.map(
                run_try, train_process_input, pm_pbar=True, pm_processes=N_PROCESS
            )
            print("train data processed", list(train_result).count(True))
