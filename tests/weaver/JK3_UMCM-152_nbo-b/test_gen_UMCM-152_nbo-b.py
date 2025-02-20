import pytest
import os

import weaver
from molsys.util.sysmisc import _makedirs

net = 'nbo-b.mfpx'

#@pytest.mark.parametrize("dut-8_all")
def test_weave_target_orients():
    if os.path.isdir('run') == False: os.mkdir('run')
    f = weaver.framework('UMCM-152_nbo-b')
    f.read_topo(net)
    
    f.assign_bb('0','CuPW.mfpx',target_norients = 1,rmsdthresh=0.1)
    f.assign_bb('1','asym_btb3.mfpx',target_norients = 4,rmsdthresh=0.1)
    f.get_target_orientations(interactive=False) # setting interactive to true will open vmd 
    f.autoscale_net(fiddle_factor=2.25)
    #test reading wnd writing of the orientations 
    f.write_orientations('run/UMCM-152_nbo.b.orients',version='2.0')
    del f

    f = weaver.framework('UMCM-152_nbo-b')
    f.read_topo(net)
    f.assign_bb('0','CuPW.mfpx',target_norients = 1,rmsdthresh=0.1)
    f.assign_bb('1','asym_btb3.mfpx',target_norients = 4,rmsdthresh=0.1)
    f.generate_bblist()
    
    f.read_orientations('run/UMCM-152_nbo.b.orients')
    f.autoscale_net(fiddle_factor=2.25)
    pwd = os.getcwd()
    os.chdir('run')
    f.generate_all_frameworks('UMCM-152_nbo-b')
    os.chdir(pwd)

