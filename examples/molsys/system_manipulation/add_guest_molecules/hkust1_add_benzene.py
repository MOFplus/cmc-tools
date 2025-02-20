import molsys

# read molsys object directly from mfpx/ric/par incuding MOF-FF forcefield
#  note: only the hkust mof is periodic
mof = molsys.mol.from_ff("hkust1")
b = molsys.mol.from_ff("benz")

# load the molecule addon of the mof
mof.addon("molecules")

# add 20 benzene molecules using packmol (must be installed and in the path)
mof.molecules.add_molecule(b,20,pack=True)

# write out mfpx/ric/par of the combined system
mof.ff.write("hkust1_benz")
mof.write("hkust1_benz.mfpx")



