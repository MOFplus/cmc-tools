import molsys
import copy

def make_molecular_graph(mol):
    mol.addon("graph")
    mol.graph.make_graph()
    mg = mol.graph
    return mg

mol1 = molsys.mol.from_file('mol1.mfpx','mfpx')
mol2 = molsys.mol.from_file('mol2.mfpx','mfpx')

mg_mol1 = make_molecular_graph(mol1)
mg_mol2 = make_molecular_graph(mol2)

checks = []

checks.append(molsys.addon.graph.is_equal(mg_mol1.molg, mg_mol2.molg, use_fast_check=False)[0])

refs = [ True]

for chk, ref in zip(checks,refs):
   assert chk == ref
