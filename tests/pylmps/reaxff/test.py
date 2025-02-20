import molsys
import pylmps


# Note: This will test an loose optimization using ReaxFF in pylmps

ref_energies = { 'reax_bond' : -9446.84294381
               , 'Coulomb'   : -35.66730987
               }

m = molsys.mol.from_file("phenantrene.xyz")

pl = pylmps.pylmps("reax")
pl.setup(local=True, mol=m, ff="ReaxFF", use_pdlp=True, origin="center")

pl.MIN(0.001)

energies = pl.get_energy_contribs()


for e in ref_energies.keys():
    assert abs(ref_energies[e]-energies[e])<1.0e-4
