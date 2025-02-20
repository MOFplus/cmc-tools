import pylmps

def test_pdlp_traj():
    pl = pylmps.pylmps("hkust1")
    pl.setup(restart="default", bcond=2)

    pl.MD_init("equil", T=300.0, startup=True, ensemble="nvt", thermo = "ber", relax=[1000.0], traj=["xyz", "vel"], tnstep=10, rnstep=10)
    pl.MD_run(100)

    pl.end()

test_pdlp_traj()
