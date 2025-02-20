import pylmps
import molsys
from ase.calculators.turbomole import Turbomole
from ase.build import molecule
from ase.io import read
import sys
import numpy as np

xyzfile = "methanol.xyz"
# instantiate the mol object
m = molsys.mol.from_file(xyzfile, ftype='xyz')
# instantiate the ase atoms object
m_ase = read(xyzfile)
# prepare the parameters for turbomole define
params = {
    'title': 'test',
    'use redundant internals': True,
    'basis set name': 'cc-pVDZ',
    'total charge': 0,
    'multiplicity': 1,
    'use dft': True,
    'density functional': 'b3-lyp',
    'grid size': 'm5',
    'use resolution of identity': True,
    'ri memory': 1000,
    'dispersion correction': 'bj',
    'scf energy convergence': 1e-7,
    'scf iterations': 300
}
# set the ase turbomole calculator
calc = Turbomole(**params)
m_ase.calc = calc
# instantiate the pylmps object
pl = pylmps.pylmps("ASE")
# instantiate the external potential expot_ase_turbomole
ep = pylmps.expot_ase_turbomole(m_ase,[i for i in range(m.natoms)])
# we need to register the objects callback as a global function
callback = ep.callback
# now add the expot object together with the name of the global callback
pl.add_external_potential(ep, "callback")
pl.setup(local=True,mol=m,ff="ase", bcond=0, kspace=False, origin="center")
# A few basic tests
pl.MIN(0.5)
pl.calc_energy(init=True)
ref_energy = -72576.28293080
assert (abs(ref_energy-pl.energy))<1.0e-6
fxyz, num_fxyz = pl.calc_numforce()
assert np.sqrt(np.mean((fxyz-num_fxyz)**2.0))<1.0e-3
# optionally remove the trubomole-specific files
#m_ase.calc.reset()
pl.end()
