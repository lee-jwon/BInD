import os
import shutil
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Geometry import Point3D


def xyz_to_sdf(xyz_fn, sdf_fn):
    assert xyz_fn[:-3] == "xyz" and sdf_fn[:-3] == "sdf"
    os.system(f"obabel {xyz_fn} -O {sdf_fn} &> /dev/null")


def xyz_to_pdb(xyz_fn, pdb_fn):
    assert xyz_fn[:-3] == "xyz" and pdb_fn[:-3] == "pdb"
    os.system(f"obabel {xyz_fn} -O {pdb_fn} &> /dev/null")


def sdf_to_pdb(sdf_fn, pdb_fn):
    assert sdf_fn[:-3] == "sdf" and pdb_fn[:-3] == "pdb"
    os.system(f"obabel {sdf_fn} -O {pdb_fn} &> /dev/null")


def pdb_to_sdf(pdb_fn, sdf_fn):
    assert sdf_fn[:-3] == "sdf" and pdb_fn[:-3] == "pdb"
    os.system(f"obabel {pdb_fn} -O {sdf_fn} &> /dev/null")


def split_pdb(pdb_fn, split_pdb_fn_prefix):
    os.system(f"obabel {pdb_fn} -O {split_pdb_fn_prefix}_.pdb -m &> /dev/null")


def split_sdf(sdf_fn, split_sdf_fn_prefix):
    os.system(f"obabel {sdf_fn} -O {split_sdf_fn_prefix}_.sdf -m &> /dev/null")


def read_mol_file(filename, sanitize=True):
    extension = filename.split(".")[-1]
    # extension = filename[-3:]
    if extension == "sdf":
        mol = Chem.SDMolSupplier(filename, sanitize=sanitize)[0]
    elif extension == "mol2":
        mol = Chem.MolFromMol2File(filename)
    elif extension == "pdb":
        mol = Chem.MolFromPDBFile(filename, sanitize=sanitize)
    elif extension == "xyz":
        mol = Chem.MolFromXYZFile(filename)
    else:
        print(f"Wrong file format... {filename}")
        return
    if mol is None:
        # print(f"No mol from file... {filename}")
        return
    return mol


def read_mols_file(filename):
    extension = filename.split(".")[-1]
    if extension == "sdf":
        mol = Chem.SDMolSupplier(filename)
    return mol


def write_mols_to_sdf(mols, fn, kekulize=False):
    writer = Chem.SDWriter(fn)
    if not kekulize:
        rdmolfiles.SDWriter.SetKekulize(writer, False)
    for mol in mols:
        # Chem.SanitizeMol(mol)
        writer.write(mol)
    writer.close()
    return


def write_conformers_to_sdf(mol1, output_file):
    """
    Write all conformers from two RDKit Mol objects to a single SDF file.

    Args:
        mol1 (rdkit.Chem.rdchem.Mol): First RDKit Mol object.
        mol2 (rdkit.Chem.rdchem.Mol): Second RDKit Mol object.
        output_file (str): Output file path for the SDF file.
    """
    writer = Chem.SDWriter(output_file)

    for conformer in mol1.GetConformers():
        writer.write(mol1, confId=conformer.GetId())

    writer.close()


def write_coordinates_to_conformer(coordinates, mol):
    coordinates = list(coordinates)
    mol = deepcopy(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol


def recreate_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return
