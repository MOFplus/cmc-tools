# -*- coding: utf-8 -*-
from __future__ import print_function

### filename pdb_ instead of pdb is meant
### to avoid conflict w/ pdb = python debugger [RA]

# RS: fix residx and resnames when not set

def write(mol, f, resnames = None, residx = None, end = True):
    if resnames is None:
        if mol.nfrags > 0:
            resnames = [i.upper()[:3] for i in mol.fragtypes]
            residx = [i+1 for i in mol.fragnumbers]
        else:
            resnames = mol.natoms * ["XYZ"]
            residx   = mol.natoms * [1]
    # write header
    f.write("MODEL    1\n")
    # write unit cell information
    f.write("{:6s}{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".
    format("CRYST1",mol.cellparams[0],mol.cellparams[1],mol.cellparams[2],mol.cellparams[3],mol.cellparams[4],mol.cellparams[5]))
    # write coordinates
    for i in range(mol.natoms):
        if i >= 99998: 
            ii = 99998
        else:
            ii = i
        f.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".
        format("ATOM",ii+1,mol.elems[i].upper(),"", resnames[i], "X", residx[i],"",mol.xyz[i,0],mol.xyz[i,1],mol.xyz[i,2],1,0,mol.elems[i],""))
    if end:
        f.write("END\n")
    else:
        f.write("ENDMDL\n") 
    return
