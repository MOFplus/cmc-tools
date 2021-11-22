import molsys
import pylmps

m = molsys.mol.from_file("init.xyz")

pl = pylmps.pylmps("reax")
pl.setup(local=True, mol=m, ff="ReaxFF", use_mfp5=True, origin="center")

#
# Prepare heat up stage before we actually sample (here very short for illustraion purposes) 
#
pl.MD_init("heat", startup=True, startup_seed=110, T=2000.0, thermo="ber", relax=[0.1], ensemble="nvt")
pl.MD_run(500)

#
# Prepare sample stage for the actuall production level calculation (here very very short for illustraion purposes) 
#
pl.MD_init("sample", T=2000.0, thermo="hoover", relax=[0.1], ensemble="nvt", traj=["xyz"])
pl.MD_run(5000)
