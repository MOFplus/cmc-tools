import numpy
import string

def read(mol, f, cycle = 0):
    """
    Routine to read castep .cell files
    :Parameters:
        -f   (obj): xyz file object
        -mol (obj): instance of a molclass
    """
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    done=False
    elems,xyz = [],[]
    cell = numpy.zeros((3,3),dtype='float')
    while not done:
        line = f.readline()
        if "%block LATTICE_CART" in line:
            unit = f.readline()
            cell[:,0] = numpy.array([float(x) for x in f.readline().split() if x != ''])
            cell[:,1] = numpy.array([float(x) for x in f.readline().split() if x != ''])
            cell[:,2] = numpy.array([float(x) for x in f.readline().split() if x != ''])
        elif '%block POSITIONS' in line:
            done_atoms =False
            while not done_atoms:
                line = f.readline()
                if "%endblock POSITIONS" in line:
                    done_atoms=True
                    done=True
                    continue
                lsplit = [x for x in line.split() if x != '']
                elems.append(string.lower(lsplit[0]))
                xyz.append([float(x) for x in lsplit[1:4]])
    natoms = len(elems)
    mol.natoms = natoms
    mol.xyz = numpy.array(xyz)
    mol.set_cell(cell, cell_only=True)
    mol.wrap_in_box()
    mol.elems = elems   
    mol.atypes = elems 
    mol.set_empty_conn()
    mol.detect_conn(thresh=0.3)
    mol.set_nofrags()
    return

def write(mol, f):
    """
    Routine to write a castep .cell file
    :Parameters:
        -mol    (obj): instance of a molclass
        -f (obj) : file object or writable object
    """
    natoms = mol.natoms 
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
#    if mol.periodic:
#        f.write("%5d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n\n" % tuple([mol.natoms]+mol.cellparams))
#    else:
    f.write("%d\n\n" % mol.natoms)
    for i in range(natoms):
        f.write("%2s %12.6f %12.6f %12.6f\n" % (mol.elems[i], mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2]))
    return
