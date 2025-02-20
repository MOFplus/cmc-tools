import numpy
import molsys.util.unit_cell as unit_cell

def read(mol,f,triclinic=False,atom_offset=0,idx_map={}):
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    stop = False
    natoms = 0
    while not stop:
        line = f.readline()
        if "ITEM: NUMBER OF ATOMS" in line:
            natoms = int(f.readline().split()[0])
        elif "ITEM: BOX BOUNDS" in line:
            cell = numpy.zeros([3,3])
            if triclinic is not True:
                cell[0,0] = float(f.readline().split()[1])
                cell[1,1] = float(f.readline().split()[1])
                cell[2,2] = float(f.readline().split()[1])
            else:
                c1 = [float(x) for x in f.readline().split() if x != '']
                c2 = [float(x) for x in f.readline().split() if x != '']
                c3 = [float(x) for x in f.readline().split() if x != '']
                xy,xz,yz = c1[2]      ,       c2[2],       c3[2]
                xlo = c1[0] - numpy.min([0.0,xy,xz,xy+xz])
                xhi = c1[1] - numpy.max([0.0,xy,xz,xy+xz])
                ylo = c2[0] - numpy.min([0.0,yz])
                yhi = c2[1] - numpy.max([0.0,yz])
                zlo,zhi = c3[0],c3[1]
                lx,ly,lz = xhi-xlo, yhi-ylo, zhi-zlo
                #lx,ly,lz = c1[2]-c1[1], c2[2]-c2[1], c3[2]-c3[1]
                #xy,xz,yz = c1[0]      ,       c2[0],       c3[0]
                a = lx
                b = numpy.sqrt(ly*ly+xy*xy)
                c = numpy.sqrt(lz*lz+xz*xz+yz*yz)
                alpha = numpy.rad2deg(numpy.arccos((xy*xz+ly*yz)/(b*c)))
                beta  = numpy.rad2deg(numpy.arccos(xz/c))
                gamma = numpy.rad2deg(numpy.arccos(xy/b))
                cell = unit_cell.vectors_from_abc([a,b,c,alpha,beta,gamma])
                #c1 = [float(x) for x in f.readline().split() if x != '']
                #cell[0,0] = c1[1]-c1[0]
                #cell[0,1] = c1[2]
                #c1 = [float(x) for x in f.readline().split() if x != '']
                #cell[1,1] = c1[1]-c1[0]
                #cell[0,2] = c1[2]
                #c1 = [float(x) for x in f.readline().split() if x != '']
                #cell[2,2] = c1[1]-c1[0]
                #cell[1,2] = c1[2]
        elif "ITEM: ATOMS" in line:
            assert natoms > 0
            xyz = numpy.zeros([natoms,3])
            elems = []
            atypes = []
            for i in range(natoms): elems.append("c")
            for i in range(natoms): atypes.append("c")
            for i in range(natoms):
                sline = f.readline().split()
                idx = int(sline[0]) -1 - atom_offset
                if len(idx_map.keys()) != 0:
                    idx = idx_map[idx]
                xyz[idx,0] = float(sline[3])
                xyz[idx,1] = float(sline[4])
                xyz[idx,2] = float(sline[5])
                elems[idx]  = sline[2].lower()
                atypes[idx] = sline[1]
            stop = True
    mol.natoms = natoms
    mol.set_cell(cell)
    mol.elems = elems
    mol.atypes = atypes
    mol.xyz = xyz
    mol.set_nofrags()
    mol.set_empty_conn()
    #mol.detect_conn()
    return






def write(mol, f,vel=None):
    '''
    Write lammpstrj to visualize GCMD runs, write lambda into velocities
    :Parameters:
        -mol    (obj): instance of a molclass
        -f (obj) : file object or writable object
    '''
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    natoms = mol.natoms 
    #### timestep header, not sure if necessary
    f.write('ITEM: TIMESTEP\n0.1\n')

    #### number of atoms with header
    f.write('ITEM: NUMBER OF ATOMS\n')
    f.write (str(natoms)+'\n')
    
    #### cell parameters with header
    # there is a problem here: either we wrap in box or we have to shift cell xyzminmax by the respective half 
    # as a test: go for the first one!
    mol.wrap_in_box()
    f.write('ITEM: BOX BOUNDS pp pp pp\n')
    f.write ('0.00 %f \n' % mol.cell[0,0])
    f.write ('0.00 %f \n' % mol.cell[1,1])
    f.write ('0.00 %f \n' % mol.cell[2,2])
    #### hope everything is orthorombic! otherwise tis aint gonna work! maybe irellevant anyway for  visualization ;)

    #### atom positions (and velocities) with header
    # i think vmd needs all, vx,vy,vz in order to read it properly!
    f.write('ITEM: ATOMS id type x y z vx vy vz\n')
    if vel is None: vel = numpy.zeros((natoms,3))
    for i in range(natoms):
        f.write("%i %2s %f %f %f %f %f %f \n" % (i, mol.elems[i],
            mol.xyz[i][0], mol.xyz[i][1], mol.xyz[i][2],
            vel[i][0], vel[i][1], vel[i][2]))
    #f.close()
    return


def write_raw(f,stepcount,natoms,cell,elems,xyz,lamb):
    '''
    Write lammpstrj to visualize GCMD runs, write lambda into velocities
    :Parameters:
        -fname  (str): name of the xyz file
        -mol    (obj): instance of a molclass
    '''
    #### timestep header, not sure if necessary
    f.write('ITEM: TIMESTEP\n%12.1f\n' % float(stepcount))

    #### number of atoms with header
    f.write('ITEM: NUMBER OF ATOMS\n')
    f.write (str(natoms)+'\n')
    
    #### cell parameters with header
    # there is a problem here: either we wrap in box or we have to shift cell xyzminmax by the respective half 
    # as a test: go for the first one!
    f.write('ITEM: BOX BOUNDS pp pp pp\n')
    f.write ('0.00 %f \n' % cell[0,0])
    f.write ('0.00 %f \n' % cell[1,1])
    f.write ('0.00 %f \n' % cell[2,2])
    #### hope everything is orthorombic! otherwise tis aint gonna work! maybe irellevant anyway for  visualization ;)

    #### atom positions (and velocities) with header
    # i think vmd needs all, vx,vy,vz in order to read it properly!
    f.write('ITEM: ATOMS id type x y z vx vy vz\n')
    for i in range(natoms):
        f.write("%i %2s %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f \n" % (i,elems[i], xyz[i,0], xyz[i,1], xyz[i,2],lamb[i], 0.0,0.0))
        #f.write("%2s %12.6f %12.6f %12.6f\n" % (mol.elems[i], mol.xyz[i,0], mol.xyz[i,1], mol.xyz[i,2]))
    return
