import pytest

import molsys
import molsys.util.toper as toper
from molsys.util.color import make_mol
from molsys.util.misc import argsorted

import glob
import os
from molsys.util.sysmisc import _makedirs

moffolder = "mofs"

frameworks = glob.glob("weave/run/333/*.mfpx")
@pytest.mark.parametrize("mofpath", frameworks)
def test_compute_colors(mofpath):
    mofname = mofpath.rstrip(".mfpx").split(os.sep)[-1]
    m = molsys.mol.from_file(mofpath)

    tt = toper.topotyper(m)
    folder = tt.write_bbs("%s%s%s" % ("run", os.sep, mofname), index_run=True)
    tt.write_colors(folder)

