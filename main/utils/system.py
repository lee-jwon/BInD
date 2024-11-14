import random
import socket
import subprocess
from contextlib import closing

import numpy as np
import torch


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def fix_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cuda_visible_devices(num_gpus: int) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 16
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]

            import sys

            major, minor = sys.version_info[0], sys.version_info[1]
            if major == 3 and minor > 6:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True
                )  # after python 3.7
            if major == 3 and minor <= 6:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, universal_newlines=True
                )  # for python 3.6

            if "No devices were found" in proc.stdout:
                break

            if "No running" in proc.stdout:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


def stat_cuda(msg):
    print("--", msg)
    print(
        "allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM"
        % (
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_reserved() / 1024 / 1024,
            torch.cuda.max_memory_reserved() / 1024 / 1024,
        )
    )


def text_filling(text, char="#", num_char=80):
    if len(text) > num_char:
        return text
    elif len(text) % 2 == 0:
        right = left = (num_char - len(text)) // 2 - 1
        return char * left + f" {text} " + char * right
    else:
        right = (num_char - len(text)) // 2 - 1
        left = (num_char - len(text)) // 2
        return char * left + f" {text} " + char * right


def print_NCI_pattern(h, i):
    REC_AMINO_ACIDS = [
        None,
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
    REC_ATOM_SYMBOLS = [None, "C", "N", "O", "S"]
    INTERACTION_TYPE = [
        None,
        "SBA",
        "SBC",
        "HBD",
        "HBA",
        "HI",
        "PP",
    ]

    aas = [REC_AMINO_ACIDS[j] for j in h[:, 5:26].argmax(-1)]
    atoms = [REC_ATOM_SYMBOLS[j] for j in h[:, 0:5].argmax(-1)]
    ncis = [INTERACTION_TYPE[j] for j in i.argmax(-1)]

    for j, (aa, atom, nci) in enumerate(zip(aas, atoms, ncis)):
        if nci is not None:
            print(f"{j}\t{atom}\t{aa}\t{nci}")
