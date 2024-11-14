import os
import pickle
from pprint import pprint

import numpy as np
from rdkit import Chem

"""
docking score based optmization
high docking affinity = good interaction profile

"""


def read_pdb_file(file_path):
    with open(file_path, "r") as f:
        pdb_block = f.read()
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
    mol = Chem.rdmolops.RemoveAllHs(mol, sanitize=False)
    # mol = Chem.RemoveHs(mol, implicitOnly=False)
    # mol = Chem.RemoveHeterogems(mol, keepSymbols=['H'])
    Chem.rdmolops.SanitizeMol(
        mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
    )
    return mol


def read_affinity_txt_try(fn):
    with open(fn, "r") as file:
        # Read the first line of the file
        content = file.readline()
        # Convert the string to a float
        number = float(content)
    return number


def get_percentile_values_and_indices(data, p):
    filtered_data = [
        (value, idx) for idx, value in enumerate(data) if value is not None
    ]
    sorted_data = sorted(filtered_data, key=lambda x: x[0])
    num_elements = int(len(sorted_data) * p)
    percentile_values_and_indices = sorted_data[:num_elements]
    values = [val for val, idx in percentile_values_and_indices]
    indices = [idx for val, idx in percentile_values_and_indices]
    return values, indices


def _extract_interaction_from_score(dirn, p, n_gen=100):
    """
    dirn: one upper dirn
    n_gen: number to scan
    gen: uses generated interactions
    min: uses minimized interactions
    dock: uses docked interactions (obtained from eval)
    """
    samples = []
    for i in range(1, n_gen + 1):
        out = {}
        out["mol_fn"] = os.path.join(dirn, f"gen_{i}.sdf")
        out["score_fn"] = os.path.join(dirn, f"gen_{i}_score_affinity.txt")
        out["inter_fn"] = os.path.join(dirn, f"gen_{i}_plip_interaction.pkl")
        out["rec_fn"] = os.path.join(dirn, "rec.pdb")

        try:
            smiles = Chem.SDMolSupplier(out["mol_fn"])[0]
        except:
            continue
        try:
            out["score"] = read_affinity_txt_try(out["score_fn"])
        except:
            continue
        try:
            with open(out["inter_fn"], "rb") as f:
                out["inter"] = pickle.load(f)
        except:
            continue
        samples.append(out)

        # read rec, lig
        rec_mol = read_pdb_file(out["rec_fn"])
        lig_mol = Chem.SDMolSupplier(out["mol_fn"], sanitize=False)[0]
        out["n_rec"] = rec_mol.GetNumAtoms()
        out["n_lig"] = lig_mol.GetNumAtoms()

    scores = [x["score"] for x in samples]
    _, idxs = get_percentile_values_and_indices(scores, p)
    sel_samples = [samples[i] for i in idxs]
    return sel_samples


def convert_inter_info_to_data(inter_info, n_rec, n_lig):
    rec_to_lig_index = [[t1, t2] for t1 in range(n_rec) for t2 in range(n_lig)]
    rec_to_lig_index = np.array(rec_to_lig_index).T
    rec_to_lig_type = np.zeros([rec_to_lig_index.shape[1]]).astype(int)
    for i, key in enumerate(inter_info):
        for rec_idx, lig_idx in inter_info[key]:
            rec_to_lig_type[n_lig * rec_idx + lig_idx] = i + 1
    e_index = rec_to_lig_index.astype(int)
    e_type = rec_to_lig_type.astype(int)
    return e_index, e_type


# /home/ljw/storage/NCIDiff/generate/240714_exp_vb_loss/1/300_3/gen/2/gen_1_plip_interaction_distance.pkl
# /home/ljw/storage/NCIDiff/generate/240714_exp_vb_loss/1/300_3/gen/2/gen_1_plip_interaction.pkl
x = _extract_interaction_from_score("./generate/", p=0.2, n_gen=100)
y = convert_inter_info_to_data(x[0]["inter"], x[0]["n_rec"], x[0]["n_lig"])
print(x[0]["inter"])
print(y)
