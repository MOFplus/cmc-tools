import molsys
import copy

def make_molecular_graph(mol):
    mol.addon("graph")
    mol.graph.make_comp_graph()
    mg = mol.graph
    return mg

mol1 = molsys.mol.from_file('spec_0.mfpx','mfpx')
mol2 = molsys.mol.from_file('spec_1.mfpx','mfpx')
mol3 = molsys.mol.from_file('spec_3.mfpx','mfpx')

mg_mol1 = make_molecular_graph(mol1)
mg_mol2 = make_molecular_graph(mol2)
mg_mol3 = make_molecular_graph(mol3)

checks = []

checks.append(molsys.addon.graph.is_equal(mg_mol1.molg, mg_mol2.molg, use_fast_check=False)[0])
checks.append(molsys.addon.graph.is_equal(mg_mol2.molg, mg_mol3.molg, use_fast_check=False)[0])

refs = [ False, True]

for chk, ref in zip(checks,refs):
   assert chk == ref
