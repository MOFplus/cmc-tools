import numpy
import string

def read(mol, arr, use_pconn=False, **kwargs):
    """
    Read array as coordinates
    :Arguments:
        -arr    (ndarray): name of the coordinate array
        -mol        (obj): instance of a molclass
        -use_pconn (bool): True to set empty pconn
    """
    mol.set_natoms(len(arr))
    mol.set_xyz(numpy.array(arr, dtype='float'))
    mol.set_elems(["x"]*mol.natoms)
    mol.set_atypes(["0"]*mol.natoms)
    mol.set_empty_conn()
    if use_pconn: mol.set_empty_pconn()
    mol.set_nofrags()
    return
