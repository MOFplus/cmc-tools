import pytest

import molsys
import molsys.util.toper as toper
from molsys.util.color import make_mol
from molsys.util.misc import argsorted

import os

fnames = [  "1",  "2",  "3",  "4",  "5"]
nets   = ["rtl","eea","eea","apo","eea"]

@pytest.mark.parametrize("fname,net", zip(fnames,nets))
def test_compute_colors(fname, net):
    name = fname.rstrip('.cif')
    m = molsys.mol()
    mofpath = "%s%s%s.cif" % ("mofs", os.sep, fname)
    m.read(mofpath)

    tt = toper.topotyper(m)
    assert tt.get_net() == [net]

    folder = tt.write_bbs("%s%s%s" % ("run", os.sep, name), index_run=True)

    tt.write_colors(folder, scell=[3,3,3])
    m.make_supercell([3,3,3])
    m.write("%s%s%s" % (folder, os.sep, "net.txyz"), pbc=False)

