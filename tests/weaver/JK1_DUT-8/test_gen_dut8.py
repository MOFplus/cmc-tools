import pytest
import os

import weaver
from molsys.util.sysmisc import _makedirs

netfolder = 'nets'
net = ['pcu_2c_tertiary.mfpx']

#@pytest.mark.parametrize("dut-8_all")
def test_weave_norot():
    f = weaver.framework('dut-8')
    netpath = "%s%s%s" % (netfolder, os.sep, net[0])
    f.read_topo(netpath)
    
    f.assign_bb('0','bbs/CuPWL6.mfpx',specific_conn = [['1x','1y'],'2'])
    f.assign_bb('1x','bbs/ndc_x.mfpx',no_rot=True,zflip=True)
    f.assign_bb('1y','bbs/ndc_y.mfpx',no_rot=True,zflip=True)
    f.assign_bb('2','bbs/dabco.mfpx',linker=True)

    f.net.make_supercell([2,2,1])
    f.net.scale_cell([13.0,13.0,9.0])

    f.generate_bblist()
    f.scan_orientations(20)

    
    try:
        os.mkdir('run')
    except:
        pass
    pwd = os.getcwd()
    os.chdir('run')
    
    f.generate_all_frameworks('dut8')
    
    os.chdir(pwd)

