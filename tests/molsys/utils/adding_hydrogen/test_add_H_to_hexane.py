
import molsys
from molsys.util.hydrogen import hydrogen



def test_add_hydrogens():
    m = molsys.mol.from_file('hexane_noH.mfpx')
    h = hydrogen(m)
    
    for i in range(m.natoms):
        h.add_sp3(i)
    m.write('hexane_incl_H.mfpx')

    assert m.natoms == 20
