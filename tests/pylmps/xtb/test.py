
import numpy as np

import pylmps
import molsys


mol = molsys.mol.from_file("geom.xyz")

pl = pylmps.pylmps("opt")

# instantiate the external potential expot_xtb

# GFN1 calculatuion (muted verbose=0, set to 2 for full output)
gfn = 1 #  0: GFN0
        #  1: GFN1
        #  2: GFN2
        # -1: GFN-FF
ep = pylmps.expot_xtb(mol,gfn,verbose=0,maxiter=1000)

# we need to register the objects callback as a global function
callback = ep.callback

# now add the expot object together with the name of the global callback
pl.add_external_potential(ep, "callback")

# setup xTB
pl.setup(local=True, mol = mol, ff="xTB")

final_energy = pl.MIN(0.01)


#
# Do the test
#
ref_energies = {
            'xtb': -26264.94056182 
}


energies = pl.get_energy_contribs()

for e in ref_energies.keys():
   assert abs(ref_energies[e]-energies[e])<1.0e-6

