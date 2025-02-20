# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import subprocess

from .molsys_mpi import mpiobject
from . import molsys_mpi
from .mol import mol
#from .topo import topo
from . import addon

def git_revision_hash():
    wrk_dir = os.getcwd()
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(src_dir)
        rev_no = len(subprocess.check_output(['git', 'rev-list', 'HEAD'], universal_newlines=True).strip().split("\n"))
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True).strip()
        status = subprocess.check_output(['git', 'status'], universal_newlines=True).strip()
        if "nothing to commit, working tree clean" in status:
            status = "Fine"
        else:
            status = "Found changes"

    except:
        # catchall in case the code is downloaded via a zip file
        rev_no = 0.0
        commit = "abcdefghijklmnop"
    finally:
        os.chdir(wrk_dir)
    return (rev_no, commit, status)

rev_no, commit, status = git_revision_hash()
print(f"You are molsys commit (short): {commit}")
if status != "Fine":
    print("Found uncommited changes: commit hash is not accurate!")
__version_info__ = (0, 0, rev_no, "%s"%commit)
__version__ = "%i.%i.%i.%s"%__version_info__


__all__=["mol", "mpiobject", "molsys_mpi"]
