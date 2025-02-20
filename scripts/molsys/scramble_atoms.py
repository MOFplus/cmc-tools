#! /usr/bin/env  python
"""

scramble atoms

this function takes a mol object and returns a new one with all atoms scrambled in their order

it should be useful for testing the senitivity wrt to atom ordering

"""


import sys
import molsys
import random
import numpy as np

def scramble_atoms(min):
    m = min.clone()
    imap = np.random.permutation(m.natoms)
    # xyz
    xyz = m.get_xyz()
    m.set_xyz(xyz[imap])
    # elems
    elems = m.get_elems()
    m.set_elems([elems[i] for i in imap])
    m.set_real_mass()
    # atypes
    atypes = m.get_atypes()
    m.set_atypes([atypes[i] for i in imap])
    # fragtypes
    ftypes = m.get_fragtypes()
    m.set_fragtypes([ftypes[i] for i in imap])
    # fragnumbers
    fnumber = m.get_fragnumbers()
    m.set_fragnumbers([fnumber[i] for i in imap])
    # conn
    conn = m.get_conn()
    new_conn = []
    imap = imap.tolist()
    for i in range(m.natoms):
        iconn = conn[imap[i]]
        # remap the j indices
        new_iconn = [imap.index(j) for j in iconn]
        new_conn.append(new_iconn)
    m.set_conn(new_conn)
    return m









if __name__ == "__main__":
    name = sys.argv[1]
    m = molsys.mol.from_file(name + ".mfpx")
    mout = scramble_atoms(m)
    mout.write(name + "_scrambled.mfpx")




