import os
import pickle
import random
import statistics
import sys
import tempfile
import time
import warnings
import subprocess
from copy import deepcopy
from multiprocessing import Pool, process
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, SanitizeMol, rdMolDescriptors
from rdkit.Chem.QED import qed

# Get the absolute path of the directory two levels up
current_file_path = Path(__file__).resolve()  # Resolve to get the absolute path
two_levels_up = current_file_path.parent.parent.parent
# print(two_levels_up)

# Add this directory to sys.path
sys.path.append(str(two_levels_up))

from plip.structure.preparation import PDBComplex


def join_complex(ligand_fn, pocket_fn, complex_fn=None, tempfile_dirn=None):
    if complex_fn is None:
        fd, complex_fn = tempfile.mkstemp(
            suffix=".pdb", prefix="temp_plip_input_complex_", dir=tempfile_dirn
        )
        os.close(fd)  # Close the file descriptor to avoid issues
    command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d"
    os.system(command)
    # print(command)
    subprocess.call(command, shell=True)
    """if complex_fn is None:
        fd, complex_fn = tempfile.mkstemp(
            suffix=".pdb", prefix="temp_plip_input_complex_", dir="./temp/"
        )
    command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d"
    os.system(command)"""
    with open(complex_fn, "r") as f:
        lines = f.readlines()
    num_ligand_atom = Chem.SDMolSupplier(ligand_fn)[0].GetNumAtoms()
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
    complex_mol = Chem.MolFromPDBFile(complex_fn)
    return complex_mol, complex_fn


def get_complex_interaction_info(complex_fn, tempfile_dirn):  # lig - rec
    my_mol = PDBComplex()
    my_mol.output_path = tempfile_dirn
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
    out = {}
    out2 = {}

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
        pairs += [[t1, t2] for t1 in rec_idx for t2 in lig_idx]
    out["SBA"] = pairs
    out2["n_SBA"] = len(anions)

    # 2 salt bridge (protein = cation)
    pairs = []
    for x in cations:  # for each interactions
        rec_idx = [x - 1 for x in x.positive.atoms_orig_idx]
        lig_idx = [x - 1 for x in x.negative.atoms_orig_idx]
        pairs += [[t1, t2] for t1 in rec_idx for t2 in lig_idx]
    out["SBC"] = pairs
    out2["n_SBC"] = len(cations)

    # 3 hydrogen bond (protein = doner)
    pairs = []
    for x in hbds:
        pairs += [[x.d_orig_idx - 1, x.a_orig_idx - 1]]
    out["HBD"] = pairs
    out2["n_HBD"] = len(hbds)

    # 4 hydrogen bond (protein = acceptor)
    pairs = []
    for x in hbas:
        pairs += [[x.a_orig_idx - 1, x.d_orig_idx - 1]]
    out["HBA"] = pairs
    out2["n_HBA"] = len(hbas)

    # 5 hydrophobic interactions
    pairs = []
    for hyd in hydros:
        pairs += [[hyd.bsatom_orig_idx - 1, hyd.ligatom_orig_idx - 1]]
    out["HI"] = pairs
    out2["n_HI"] = len(hydros)

    # 6 pipi (many to many)
    pairs = []
    for x in pipis:  # for each interactions
        rec_idx = [x - 1 for x in x.proteinring.atoms_orig_idx]
        lig_idx = [x - 1 for x in x.ligandring.atoms_orig_idx]
        pairs += [[t1, t2] for t1 in rec_idx for t2 in lig_idx]
    out["PP"] = pairs
    out2["n_PP"] = len(pipis)
    os.unlink(my_mol.sourcefiles['pdbcomplex'])
    assert "temp" in complex_fn
    os.system(f"rm {complex_fn[:-4]}*")
    return out, out2


def sanitize_pocket_interaction(inter_dict, ligand_n):
    for key in inter_dict.keys():
        for p in inter_dict[key]:
            p[0] = p[0] - ligand_n
    return inter_dict


def get_sanitized_interaction_info(rec_fn, lig_fn, tempfile_dirn):
    _, cmplx_fn = join_complex(lig_fn, rec_fn, tempfile_dirn=tempfile_dirn)
    suppl = Chem.SDMolSupplier(lig_fn)
    lig_mol = suppl[0]
    lig_n = lig_mol.GetNumAtoms()
    out, out2 = get_complex_interaction_info(cmplx_fn, tempfile_dirn)
    out = sanitize_pocket_interaction(out, lig_n)
    out.update(out2)
    return out


def get_complex_interaction_geometry(rec_fn, lig_fn, tempfile_dirn):
    _, cmplx_fn = join_complex(lig_fn, rec_fn, tempfile_dirn=tempfile_dirn)
    complex_fn = cmplx_fn
    my_mol = PDBComplex()
    my_mol.output_path = tempfile_dirn
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
    out = {}

    anions = my_interactions.saltbridge_pneg
    cations = my_interactions.saltbridge_lneg
    hbds = my_interactions.hbonds_pdon
    hbas = my_interactions.hbonds_ldon
    hydros = my_interactions.hydrophobic_contacts
    pipis = my_interactions.pistacking

    # 0 nontype

    # 1 salt brdige (protein = anion)
    ds = []
    for inter in anions:
        c1 = np.array(inter.positive.center)
        c2 = np.array(inter.negative.center)
        d = np.linalg.norm(c1 - c2)
        ds.append(d)
    out["SBA"] = ds

    # 2 salt bridge (protein = cation)
    ds = []
    for inter in cations:  # for each interactions
        c1 = np.array(inter.positive.center)
        c2 = np.array(inter.negative.center)
        d = np.linalg.norm(c1 - c2)
        ds.append(d)
    out["SBC"] = ds

    # 3 hydrogen bond (protein = doner)
    ds = []
    for x in hbds:
        d = x.distance_ad
        ds.append(d)
    out["HBD"] = ds

    # 4 hydrogen bond (protein = acceptor)
    ds = []
    for x in hbas:
        d = x.distance_ad
        ds.append(d)
    out["HBA"] = ds

    # 5 hydrophobic interactions
    ds = []
    for hyd in hydros:
        d = hyd.distance
        ds.append(d)
    out["HI"] = ds

    # 6 pipi (many to many) #TODO
    ds, angs = [], []
    for x in pipis:  # for each interactions
        d = x.distance
        a = x.angle
        ds.append(d)
        angs.append(a)
    out["PP"] = ds
    out["PP_angle"] = angs
    os.unlink(my_mol.sourcefiles['pdbcomplex'])
    time.sleep(0.3)
    assert "temp" in complex_fn
    os.system(f"rm {complex_fn[:-4]}*")
    return out


if __name__ == "__main__":
    rec_fn = sys.argv[1]
    lig_fn = sys.argv[2]
    output_interaction_fn = sys.argv[3]
    output_interaction_distance_fn = sys.argv[4]
    tempfile_dirn = sys.argv[5]

    x = get_sanitized_interaction_info(rec_fn, lig_fn, tempfile_dirn)
    y = get_complex_interaction_geometry(rec_fn, lig_fn, tempfile_dirn)

    print(x, y)

    with open(output_interaction_fn, "wb") as f:
        pickle.dump(x, f)
    with open(output_interaction_distance_fn, "wb") as f:
        pickle.dump(y, f)
