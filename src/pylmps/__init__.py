# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import subprocess

from .pylmps import pylmps
from .expot_base import expot_base, expot_test, expot_ase, expot_ase_turbomole, expot_xtb

def git_revision_hash():
    wrk_dir = os.getcwd()
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(src_dir)
        rev_no = len(subprocess.check_output(['git', 'rev-list', 'HEAD'], universal_newlines=True).strip().split("\n"))
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True).strip()
    except:
        # catchall in case the code is downloaded via a zip file
        rev_no = 0.0
        commit = "abcdefghijklmnop"
    finally:
        os.chdir(wrk_dir)
    return (rev_no, commit)

rev_no, commit = git_revision_hash()
__version_info__ = (0, 0, rev_no, "%s"%commit)
__version__ = "%i.%i.%i.%s"%__version_info__

__all__=["pylmps", "mol_coms", "expot_base", "expot_test", "expot_ase", "expot_xtb"]
