import numpy
import string

def read(mol, f, cycle = 0):
    """
    Routine, which reads an xyz file
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
    ncycle=0
    fline = f.readline().split()
    natoms = int(fline[0])
    line = f.readline()
    mol.xyzheader = line
    periodic = False
    # check for keywords used in extended xyz format
    # in the moment only the lattice keyword is implemented
    if "Lattice=" in line:
        periodic = True
        lattice = [float(i) for i in line.rsplit('"',1)[0].rsplit('"',1)[-1].split() if i != '']
        cell = numpy.array(lattice).reshape((3,3))
    xyz = numpy.zeros((natoms, 3))
    elements = []
    atypes = []
    for i in range(natoms):
        line = f.readline().split()
        elements.append(line[0].lower())
        atypes.append(line[0].lower())
        xyz[i,:] = list(map(float,line[1:4]))
    mol.natoms = natoms
    if periodic: mol.set_cell(cell)
    mol.xyz = numpy.array(xyz)
    mol.elems = elements
    mol.atypes = atypes
    mol.set_empty_conn()
    mol.set_nofrags()
    return

def write(mol, f, energy = None, forces = None, plain = False, charges = None):
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
    natoms = mol.natoms 
    header = ""
    if mol.periodic:
        header +=  'Lattice="%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f" ' %  tuple(mol.cell.ravel())
    header += "Properties=species:S:1:pos:R:3"
    if forces is not None:
        assert forces.shape == mol.xyz.shape, "Force array does not match the coordinates array"
        header+= ":forces:R:3"
    if charges is not None:
        assert charges.shape == (natoms,)
        header += ":charges:R:1"
    if energy is not None:
        header += " energy=%16.10f" % energy
    #if mol.periodic:
    #    f.write("%d\n" % mol.natoms)
    #    f.write('Lattice="%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f"\n' % 
    #        tuple(mol.cell.ravel()))
    #else:
    f.write("%d\n" % mol.natoms)
    f.write(header)
    f.write("\n")
    if plain:
        elems = [''.join([i for i in e if not i.isdigit()]) for e in mol.elems]
    else:
        elems = mol.elems
    #TODO: this has to be tidied up and formulated more elegantly!
    if forces is None:
        for i in range(natoms):
            f.write("%2s %12.8f %12.8f %12.8f\n" % (elems[i].capitalize(), mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2]))
    else:
        for i in range(natoms):
            if charges is None:
                f.write("%2s %12.8f %12.8f %12.8f   %12.8f %12.8f %12.8f\n" % 
                (elems[i].capitalize(), mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2], forces[i,0], forces[i,1], forces[i,2]))
            else:
                f.write("%2s %12.8f %12.8f %12.8f   %12.8f %12.8f %12.8f    %12.8f\n" % 
                (elems[i].capitalize(), mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2], forces[i,0], forces[i,1], forces[i,2], charges[i]))
    return
