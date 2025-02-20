import pytest

import molsys
import molsys.util.toper as toper

import os

infolder = "struct"
infile = "%s%s%s" % (infolder, os.sep, "UMCM-1_dioxole.cif")
m = molsys.mol.from_file(infile)


scell_net = [5,5,2]
strscell_net = "".join([str(i) for i in scell_net])

outfolder = "run"

@pytest.mark.slow
def test_standard_bbs():
    tt = toper.topotyper(m)
    netname = tt.get_net()
    outfolder_ = "%s%s%s" % (outfolder, os.sep,"standard")
    outfolder_bbs = tt.write_bbs(outfolder_, index_run=True)

    m_net = tt.tg.mol
    m_net.make_supercell(scell_net)
    outfile = "%s%s%s%s%s%s" % (outfolder_bbs, os.sep, netname[0], "_", strscell_net, "_standard")
    m_net.write(outfile + ".mfpx")
    m_net.write(outfile + ".txyz", pbc=False)

@pytest.mark.slow
def test_write_no_organicity_bbs():
    tt = toper.topotyper(m, split_by_org=False)
    netname = tt.get_net()
    outfolder_ = "%s%s%s" % (outfolder, os.sep, "not_split_by_org")
    outfolder_bbs = tt.write_bbs(outfolder_, index_run=True)

    m_net = tt.tg.mol
    m_net.make_supercell(scell_net)
    outfile = "%s%s%s%s%s%s" % (outfolder_bbs, os.sep, netname[0], "_", strscell_net, "_not_split_by_org")
    m_net.write(outfile + ".mfpx")
    m_net.write(outfile + ".txyz", pbc=False)

def test_write():
    scell = [2,2,2]
    m.make_supercell(scell)
    strscell = "".join([str(i) for i in scell])
    outstruct = "%s%s%s%s" % (outfolder, os.sep, "UMCM-1_", strscell)
    m.write(outstruct + ".mfpx")
    m.write(outstruct + ".txyz", pbc=False)
