import molsys

# Load molecule 
m = molsys.mol.from_file("anisole.xyz")
# load O2 to be filled in the box
g1 = molsys.mol.from_file("o2.xyz")
# We need the molecules addon
m.addon("molecules")
# and can then add 50 O2 molecules to anisole via packmol (needs to be available in your PATH)
m.molecules.add_molecule(g1, 50, pack=True)
# and finally write them to disk
m.write("init.xyz")

