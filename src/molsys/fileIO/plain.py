import numpy
import string

def read(mol, f, delimiter=','):
    """
    Routine, which reads a plain file
    :Parameters:
        -f   (obj): plain file object
        -mol (obj): instance of a molclass
        -delimiter=',' (str): coordinate delimiter
    """
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    splits = f.read().splitlines()
    xyz = [s.split(delimiter) for s in splits]
    mol.natoms = len(xyz)
    mol.xyz = numpy.array(xyz, dtype='float')
    mol.elems = ["x"]*mol.natoms
    mol.atypes = ["0"]*mol.natoms
    mol.set_empty_conn()
    mol.set_nofrags()
    return

def write(mol, f):
    """
    Routine, which writes an xyz file
    :Parameters:
        -mol    (obj): instance of a molclass
        -f (obj) : file object or writable object
    """
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    for i in range(mol.natoms):
        f.write("%12.6f %12.6f %12.6f\n" % (mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2]))
    return
