
import molsys
from molsys.util.uff import UFFAssign

ref_types = ['C_R', 'C_R', 'C_R', 'C_R', 'C_R', 'C_R', 'H_', 'H_', 'H_', 'H_', 'H_', 'H_']

mol = molsys.mol()
mol.read("benz.mfpx", ftype = "mfpx")

uff = UFFAssign(mol)
uff.assign()

uff_types = uff.get_uff_types()

assert uff_types == ref_types, "Determination of UFF atom types"
