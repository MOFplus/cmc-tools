import pytest
import os

import weaver
from molsys.util.sysmisc import _makedirs
from random import sample

bbfold = "%s%s" % ("bbs", os.sep)
runfold = "%s%s" % ("run", os.sep)
_makedirs("%s%s" % (runfold, os.sep))

def test_weave_linker_vertex():
    f = weaver.framework("uio")
    f.net.read("fcu")
    f.net.make_supercell([2,2,2])

    pdefects = float(0.20)
    ndefects = int((f.net.natoms * pdefects)//1)
    lidefects = sample(xrange(f.net.natoms), ndefects) #uniform random sampling
    f.net.atypes = [ '1' if i in lidefects else f.net.atypes[i] for i in xrange(f.net.natoms) ] #randomly change indices from 0-cluster to 1-vacuum

    #f.add_linker_vertex('1') # cluster-cluster linker

    f.add_linker_vertex('2', between=("0","0") ) #cluster-cluster linker
    f.add_linker_vertex('3', between=("0","1") ) #vacuum-cluster linker
    f.add_linker_vertex('4', between=("1","1") ) #vacuum-vacuum linker

    f.net.write(runfold+"net.mfpx")
    f.net.write(runfold+"net.txyz", pbc=False)

    f.scale_net(14.5)
    f.assign_bb('0', bbfold+'Zr6O4.mfpx') #cluster
    f.assign_bb('1', bbfold+'12-d.mfpx') #vacuum
    f.assign_bb('2', bbfold+'bdc.mfpx', linker=True, zflip=True, nrot=1) #bdc linker
    f.assign_bb('3', bbfold+'h_d.mfpx', linker=True, zflip=True, nrot=1, specific_conn=["0","1"] ) #formate modulator, order is defined in bbconn inside h_d.mfpx file
    f.assign_bb('4', bbfold+'d_d.mfpx', linker=True, zflip=True, nrot=1) #vacuum linker

    f.scan_orientations(10)
    f.write_orientations(runfold+'UiO-66.orients')

    f.generate_framework([0]*len(f.norientations))
    f.write_framework(runfold+'UiO-66.mfpx')
    f.write_framework(runfold+'UiO-66.txyz', pbc=False)
