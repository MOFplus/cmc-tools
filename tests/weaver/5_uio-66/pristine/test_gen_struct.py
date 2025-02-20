import pytest
import os

import weaver
from molsys.util.sysmisc import _makedirs

bbfold = "%s%s" % ("bbs", os.sep)
runfold = "%s%s" % ("run", os.sep)
_makedirs("%s%s" % (runfold, os.sep))

def test_weave_linker_vertex():
    f = weaver.framework("uio")
    f.net.read("fcu")

    f.add_linker_vertex('1') # cluster-cluster linker

    f.net.write(runfold+"net.mfpx")
    f.net.write(runfold+"net.txyz", pbc=False)

    f.scale_net(14.5)
    f.assign_bb('0', bbfold+'Zr6O4') # cluster
    f.assign_bb('1', bbfold+'bdc', linker=True, zflip=True, nrot=1) # linker

    f.scan_orientations(10)
    f.write_orientations(bbfold + 'UiO-66.orients')

    f.generate_framework([0]*len(f.norientations))
    f.write_framework(runfold+'UiO-66.mfpx')
    f.write_framework(runfold+'UiO-66.txyz', pbc=False)
