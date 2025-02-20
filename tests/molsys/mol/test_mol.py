import pytest

import molsys
import molsys.addon
import molsys.fileIO.formats

from molsys.util.sysmisc import _makedirs

import sys
import os
from os.path import splitext

rundir = _makedirs("run")

fname = "HKUST-1.mfpx"
name = os.path.splitext(fname)[0]

m = molsys.mol.from_file(fname)

def test_manipulate_atoms():
    pass

def test_manipulate_coordinates():
    pass

scells = [
    [1,1,1],
    [2,1,1],
    [1,2,1],
    [1,1,2],
    [2,2,1],
    [1,2,2],
    [2,1,2],
]
@pytest.mark.slow
@pytest.mark.xpass(reason="too slow")
@pytest.mark.parametrize("scell", scells)
def test_manipulate_cell(scell):
    # another instance is needed each time to avoid HUGE supercells
    m_ = molsys.mol.from_file(fname)
    m_.make_supercell(scell)

writefmts = molsys.fileIO.formats.write.keys()
@pytest.mark.parametrize("fmt", writefmts)
def test_write(fmt):
    m.write("%s/%s.%s" % (rundir, name, fmt))

@pytest.mark.parametrize("fmt", writefmts)
def test_write_supercell(fmt):
    scell = [2,2,2]
    m_ = molsys.mol.from_file(fname)
    m_.make_supercell(scell)
    strcell = 3*"%d" % tuple(scell)
    m_.write("%s/%s_%s.%s" % (rundir, strcell, name, fmt))

readfmts = set(molsys.fileIO.formats.read.keys())
readfmts -= set(["mol2", "freq", "array", "cif", "plain", "cell", "aginp"])
@pytest.mark.parametrize("fmt", readfmts)
def test_read(fmt):
    m.read("%s/%s.%s" % (rundir, name, fmt))

@pytest.mark.parametrize("addon", molsys.addon.__all__)
def test_load_addon(addon):
    if addon == "topo":
        pytest.xfail("known fail: topo is deprecated")
    from molsys.addon.ff import ff # TODO: improve testing
    m.addon(addon)

@pytest.mark.parametrize("addon", molsys.addon.__all__)
def test_load_wrong_addon(addon):
    m.addon("")
    if addon == "topo":
        pytest.xfail("known fail: topo is deprecated")
    m.addon("_"+addon+"_")

