import pylmps
import os
import pytest

# make shure that the hkust1.pdlp file is removed

@pytest.mark.first
def test_restart_pdlp():
    if os.path.exists("hkust1.pdlp"):
    	os.remove("hkust1.pdlp")
    pl = pylmps.pylmps("hkust1")
    # CHECK: do we really need this bcond=2 thing???
    pl.setup(ff="file", bcond=2, use_pdlp=True)
    e1 = pl.get_energy_contribs()
    pl.end()
    # now restart in a new instance from the pdlp file
    plr = pylmps.pylmps("hkust1")
    plr.setup(restart="default")
    e2 = plr.get_energy_contribs()
    plr.end()
    for e in e1.keys():
        assert abs(e1[e]-e2[e]) < 1e-6

#@pytest.mark.second
#def test_pdlp_traj():
#    pl = pylmps.pylmps("hkust1")
#    pl.setup(restart="default", bcond=2)
#
#    pl.MD_init("equil", T=300.0, startup=True, ensemble="nvt", thermo = "ber", relax=[1000.0], traj=["xyz", "vel"], tnstep=10, rnstep=10)
#    pl.MD_run(100)
#
#    pl.end()

test_restart_pdlp()
# test_pdlp_traj()
