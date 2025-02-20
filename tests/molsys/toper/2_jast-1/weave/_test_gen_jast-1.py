import pytest
import os

import weaver
from molsys.util.sysmisc import _makedirs

nets = [
    '333/0_pcu-333-21_18.mfpx',
    '333/1_pcu-333-21_3.mfpx',
    '666/0_pcu-666-21_36.mfpx',
    '666/1_pcu-666-21_36.mfpx',
    '666/2_pcu-666-21_36.mfpx',
    '666/3_pcu-666-21_36.mfpx',
    '666/4_pcu-666-21_6.mfpx',
    '666/5_pcu-666-21_18.mfpx',
    '666/6_pcu-666-21_18.mfpx',
    '666/7_pcu-666-21_3.mfpx'
]
netfolder = "colors"

@pytest.mark.parametrize("net",nets)
def test_weave_from_colors(net):
    f = weaver.framework('jast-1', use_new=True)
    netpath = "%s%s%s" % (netfolder, os.sep, net)
    f.read_topo(netpath)

    f.assign_bb('n','bbs/CuPW.mfpx', specific_conn=["f","b"])
    f.assign_bb('b','bbs/dabco.mfpx', linker=True, zflip=True, nrot=1)
    f.assign_bb('f','bbs/bdc.mfpx'  , linker=True, zflip=True, nrot=1)

    f.scale_net(11.0)

    f.generate_bblist(use_elems=True)
    f.scan_orientations(6)
    f.generate_framework()


    moffold = "run"
    netfold,netname = net.split(os.sep)
    _makedirs(moffold+os.sep+netfold)
    mofpath = moffold+os.sep+netfold+os.sep+"mof_"+netname
    f.framework.write(mofpath,ftype='mfpx')
    f.framework.write(mofpath.rstrip("mfpx")+"txyz",ftype='txyz',pbc=False)
