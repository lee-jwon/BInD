import contextlib
import os
import random
import string
import subprocess
import sys
import tempfile

import AutoDockTools
import rdkit.Chem as Chem
from meeko import MoleculePreparation, obutils
from openbabel import pybel
from rdkit.Chem import AllChem
from vina import Vina
import numpy as np
np.int = np.int64

# from utils.reconstruct import reconstruct_from_generated
# from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


def get_random_id(length=30):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


class BaseDockingTask(object):
    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == "smi":
            self.ob_mol = pybel.readstring("smi", input_mol)
        elif mol_format == "sdf":
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f"mol_format {mol_format} not supported")

    def addH(self, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, "tmp_h.sdf")

    def gen_conf(self):
        sdf_block = self.ob_mol.write("sdf")
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring("sdf", Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, "conf_h.sdf")

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()


class PrepProt(object):
    def __init__(self, pdb_file):
        self.prot = pdb_file

    def del_water(self, dry_pdb_file):  # optional
        with open(self.prot) as f:
            lines = [
                l
                for l in f.readlines()
                if l.startswith("ATOM") or l.startswith("HETATM")
            ]
            dry_lines = [l for l in lines if not "HOH" in l]

        with open(dry_pdb_file, "w") as f:
            f.write("".join(dry_lines))
        self.prot = dry_pdb_file

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(
            ["pdb2pqr30", "--ff=AMBER", self.prot, self.prot_pqr],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(
            AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py"
        )
        subprocess.Popen(
            ["python", prepare_receptor, "-r", self.prot_pqr, "-o", prot_pdbqt],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()


class VinaDock(object):
    def __init__(self, lig_pdbqt, prot_pdbqt):
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt

    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, "r") as f:
            lines = [
                l
                for l in f.readlines()
                if l.startswith("ATOM") or l.startswith("HEATATM")
            ]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [
                (max(xs) + min(xs)) / 2,
                (max(ys) + min(ys)) / 2,
                (max(zs) + min(zs)) / 2,
            ]
            box_size = [
                (max(xs) - min(xs)) + buffer,
                (max(ys) - min(ys)) + buffer,
                (max(zs) - min(zs)) + buffer,
            ]
            return pocket_center, box_size

    def get_box(self, ref=None, buffer=0):
        """
        ref: reference pdb to define pocket.
        buffer: buffer size to add

        if ref is not None:
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension
        else:
            use the entire protein to define pocket
        """
        if ref is None:
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(
        self,
        score_func="vina",
        seed=2024,
        mode="dock",
        exhaustiveness=8,
        save_pose=False,
        **kwargs,
    ):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, cpu=1, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == "score_only":
            score = v.score()[0]
        elif mode == "minimize":
            score = v.optimize()[0]
        elif mode == "dock":
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError

        if not save_pose:
            return score
        else:
            if mode == "score_only":
                pose = None
            elif mode == "minimize":
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, "w") as f:
                    v.write_pose(tmp.name, overwrite=True)
                with open(tmp.name, "r") as f:
                    pose = f.read()

            elif mode == "dock":
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


class VinaDockingTask(BaseDockingTask):

    """@classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())
        return cls(protein_path, ligand_rdmol, **kwargs)"""

    @classmethod
    def from_original_data(
        cls,
        data,
        ligand_root="./data/crossdocked_pocket10",
        protein_root="./data/crossdocked",
        **kwargs,
    ):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + ".pdb",
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(
        cls, ligand_rdmol, ligand_filename, protein_root="./data/crossdocked", **kwargs
    ):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + ".pdb",  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(
        self,
        protein_path,
        ligand_rdmol,
        tmp_dir="./temp",
        center=None,
        size_factor=1.0,
        buffer=5.0,
    ):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + "_receptor"
        self.ligand_id = self.task_id + "_ligand"

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + ".sdf")

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (
                pos.max(0) - pos.min(0)
            ) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode="dock", exhaustiveness=8, **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + ".pdbqt"
        protein_pqr = self.receptor_path[:-4] + ".pqr"
        protein_pdbqt = self.receptor_path[:-4] + ".pdbqt"

        lig = PrepLig(self.ligand_path, "sdf")
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)
        if not os.path.exists(protein_pqr):
            # if True:
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            # if True:
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        dock.pocket_center, dock.box_size = self.center, [
            self.size_x,
            self.size_y,
            self.size_z,
        ]
        score, pose = dock.dock(
            score_func="vina",
            mode=mode,
            exhaustiveness=exhaustiveness,
            save_pose=True,
            **kwargs,
        )
        return {"affinity": score, "pose": pose}, [ligand_pdbqt, self.ligand_path]


if __name__ == "__main__":
    method = sys.argv[1]  # score, min, dock
    rec_fn = sys.argv[2]
    lig_fn = sys.argv[3]
    output_fn = sys.argv[4]
    score_fn = sys.argv[5]
    tempfile_dirn = sys.argv[6]

    # make temp files (for preventing collision in mp)
    # ori_output_fn = output_fn
    _, temp_rec_fn = tempfile.mkstemp(
        suffix=".pdb", prefix="temp_dock_rec_fn_", dir=tempfile_dirn
    )
    _, temp_lig_fn = tempfile.mkstemp(
        suffix=".sdf", prefix="temp_dock_lig_fn_", dir=tempfile_dirn
    )
    _, temp_output_fn = tempfile.mkstemp(
        suffix=".sdf", prefix="temp_dock_output_fn_", dir=tempfile_dirn
    )

    os.system(f"cp {rec_fn} {temp_rec_fn}")
    os.system(f"cp {lig_fn} {temp_lig_fn}")

    temp_rec_pqr_fn = temp_rec_fn[:-4] + ".pqr"
    temp_rec_pqr_log_fn = temp_rec_fn[:-4] + ".log"
    temp_rec_pdbqt_fn = temp_rec_fn[:-4] + ".pdbqt"
    temp_output_pdbqt_fn = temp_lig_fn[:-4] + ".pdbqt"

    # change name
    if method == "score":
        method = "score_only"
    elif method == "min":
        method = "minimize"
    elif method == "dock":
        method = "dock"

    lig_mol = Chem.SDMolSupplier(temp_lig_fn)[0]
    dock = VinaDockingTask(temp_rec_fn, lig_mol, tempfile_dirn)
    x, ts = dock.run(mode=method, exhaustiveness=16)
    t1, t2 = ts
    s = x["affinity"]
    p = x["pose"]

    if method != "score_only":
        with open(temp_output_pdbqt_fn, "w") as file:
            file.write(p)
    else:
        ligand_rdmol = Chem.AddHs(lig_mol, addCoords=True)
        _, temp_mid_sdf_fn = tempfile.mkstemp(
            suffix=".sdf", prefix="temp_dock_output_fn_", dir=tempfile_dirn
        )
        sdf_writer = Chem.SDWriter(temp_mid_sdf_fn)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        lig = PrepLig(temp_mid_sdf_fn, mol_format="sdf")
        lig.get_pdbqt(temp_output_pdbqt_fn)
        # cmnd = f"obabel {lig_fn} -d -O {temp_output_pdbqt_fn}"
        # os.system(cmnd)

    # write affinity as txt
    out_aff_fn = score_fn
    with open(out_aff_fn, "w") as file:
        file.write(str(s))

    cmnd = f"obabel {temp_output_pdbqt_fn} -d -O {temp_output_fn}"
    os.system(cmnd)
    os.system(f"cp {temp_output_fn} {output_fn}")

    os.unlink(temp_rec_fn)
    os.unlink(temp_lig_fn)
    os.unlink(temp_output_fn)
    os.unlink(temp_rec_pqr_fn)
    os.unlink(temp_rec_pdbqt_fn)
    os.unlink(temp_output_pdbqt_fn)
    os.unlink(temp_rec_pqr_log_fn)
    if method != "score_only":
        pass 
    else:
        os.unlink(temp_mid_sdf_fn)
    os.unlink(t1)
    os.unlink(t2)
