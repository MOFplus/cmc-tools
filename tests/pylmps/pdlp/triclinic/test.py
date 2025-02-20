import pylmps

pl = pylmps.pylmps("nott101")
pl.setup(ff="file", use_pdlp=True)

pl.MD_init("equil", T=300.0, startup=True, ensemble="nvt", thermo = "ber", traj=["xyz", "vel", "cell"])
pl.MD_run(200)

pl.MD_init("sample", T=300.0, p=1.0, ensemble="npt", thermo = "hoover", traj=["xyz","cell"])
pl.MD_run(1000)

pl.end()


