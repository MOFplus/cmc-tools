import molsys

mol = molsys.mol.from_file("educt_0.xyz")
mol.detect_conn()
mol.addon("obabel")
smiles = mol.obabel.cansmiles
is_chiral, centers = mol.obabel.check_chirality()

assert is_chiral == True, "check chirality"
assert smiles == "[O]Oc1cc2[C@@H]([O])C=C[CH]c2c2c1cccc2", "SMILES string"
