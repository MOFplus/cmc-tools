#### cell manipulations
import numpy as np
import molsys

def unwrap(mol):
    """ beware, works only for non-periodic molecules """
    m = mol
    wrapped = []
    counter=0; done=False
    while not done:
        counter += 1
        if counter >= 500: 
            print('maxiter reached')
            break
        for i,cc in enumerate(m.conn):
            for ic,c in enumerate(cc):
                d,v,imgi = m.get_distvec(i,c)
                print((i,c,d,v,imgi))
                if imgi != 13:
                    if wrapped.count(c) != 0:
                        continue
                    wrapped.append(c)
                    m.xyz[c] = m.get_image(m.xyz[c],imgi)
                    break

    return

def rec_unwrap(m,idx=0,doneidx=[]):
    #import pdb; pdb.set_trace()
    c = m.conn[idx]
    for i,ci in enumerate(c):
        if doneidx.count(ci) != 0:
            continue
        d,v,imgi = m.get_distvec(idx, ci)
        if imgi[0] != [13]:
            print((d,v,imgi[0]))
            m.xyz[ci] = m.get_image(m.xyz[ci],imgi[0])
        doneidx.append(idx)
        rec_unwrap(m,ci,doneidx=doneidx)
    return


def extend_cell(mol,offset):
    ''' Atoms as close as offset to the box boundaries are selected to be copied.
        They are then added at the other side of the cell to "extend" the system periodically 
        Mainly for visualization purposes
        WARNING: Connectivity is destroyed afterwards 
        :Params: 
            - offset: The distance (in Angstroms) from the box boundary at which to duplicate the atoms
    '''
    frac_xyz = mol.get_frac_xyz()
    wherexp = np.where(np.less(frac_xyz[:,0], offset))
    wherexm = np.where(np.greater(frac_xyz[:,0], 1.0-offset))
    whereyp = np.where(np.less(frac_xyz[:,1], offset))
    whereym = np.where(np.greater(frac_xyz[:,1], 1.0-offset))
    wherezp = np.where(np.less(frac_xyz[:,2], offset))
    wherezm = np.where(np.greater(frac_xyz[:,2], 1.0-offset))
    new_xyz = frac_xyz
    #print(new_xyz.shape)
    new_xyz = np.append(new_xyz, frac_xyz[wherexp[0]]+[1.0,0.0,0.0],0)
    new_xyz = np.append(new_xyz, frac_xyz[whereyp[0]]+[0.0,1.0,0.0],0)
    new_xyz = np.append(new_xyz, frac_xyz[wherezp[0]]+[0.0,0.0,1.0],0)
    new_xyz = np.append(new_xyz, frac_xyz[wherexm[0]]-[1.0,0.0,0.0],0)
    new_xyz = np.append(new_xyz, frac_xyz[whereym[0]]-[0.0,1.0,0.0],0)
    new_xyz = np.append(new_xyz, frac_xyz[wherezm[0]]-[0.0,0.0,1.0],0)
    #print(new_xyz)
    #print(new_xyz.shape)
    for i in range(len(wherexp[0])):
        mol.elems.append(mol.elems[wherexp[0][i]])
    for i in range(len(whereyp[0])):
        mol.elems.append(mol.elems[whereyp[0][i]])
    for i in range(len(wherezp[0])):
        mol.elems.append(mol.elems[wherezp[0][i]])
    for i in range(len(wherexm[0])):
        mol.elems.append(mol.elems[wherexm[0][i]])
    for i in range(len(whereym[0])):
        mol.elems.append(mol.elems[whereym[0][i]])
    for i in range(len(wherezm[0])):
        mol.elems.append(mol.elems[wherezm[0][i]])
    #print(new_xyz)
    mol.natoms = len(mol.elems)
    mol.set_xyz_from_frac(new_xyz)
    #logging.info('Cell was extended by %8.4f AA in each direction' % (offset))
    return

def extend_cell_with_fragments(mol,fragments, boxtol = [0.1,0.1,0.1]):
    m = mol
    addmol = molsys.mol()
    offset = np.zeros([3],"d")
    boxtol = np.array(boxtol)
    for ix in range(-1,2):
        for iy in range(-1,2):
            for iz in range(-1,2):
                cell_disp = np.sum(m.cell*np.array([ix,iy,iz],"d")[:,np.newaxis],axis=0)
                xyz = m.xyz +cell_disp
                inflags, frag_com, frag_flag = check_fragcom_in_cell(mol,xyz, boxtol, fragments)
                for i in range(m.natoms):
                    if inflags[i]:
                        addmol.add_atom(m.elems[i], m.atypes[i],xyz[i])
    return addmol


def check_fragcom_in_cell(mol,xyz, tol, fragments):
    m = mol
    tol = np.array(tol)
    frag_com = np.zeros([len(fragments), 3],"d")
    m.set_real_mass()
    for i in range(len(fragments)):
        frag_com[i] = m.get_com(idx = fragments[i], xyz = xyz[fragments[i]])
    # convert these coms xyz coordinates into fractional coordinates
    frag_com_fract = np.dot(frag_com, m.inv_cell)
    # now check these coms
    below = np.less_equal(frag_com_fract, -tol)
    above = np.greater(frag_com_fract, 1.0+tol)
    allbelow = np.logical_or.reduce(below, axis=1)
    allabove = np.logical_or.reduce(above, axis=1)
    out = np.logical_or(allabove, allbelow)
    incell = np.logical_not(out)
    flags = m.natoms*[False]
    for i,f in enumerate(incell):
        if f:
            for a in fragments[i]: flags[a] = True
    return flags, frag_com, incell

def cart2frac(xyzcart, cellparams):
    a, b, c, alpha, beta, gamma = np.array(cellparams).astype(float)
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    v = np.sqrt(1 - cosa*cosa - cosb*cosb - cosg*cosg + 2*cosa*cosb*cosg)
    cart2frac = np.array([
        [1./a, -cosg/(a*sing), (cosa*cosg-cosb)/(a*v*sing)],
        [  0.,    1./(b*sing), (cosb*cosg-cosa)/(b*v*sing)],
        [  0.,             0.,                  cosg/(c*v)]
    ])
    xyzfrac = xyzcart * cart2frac
    return xyzfrac
    

def frac2cart(xyzfrac, cellparams):
    a, b, c, alpha, beta, gamma = np.array(cellparams).astype(float)
    alpha = np.deg2rad(alpha)
    beta  = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)
    cosa = np.cos(alpha)
    cosb = np.cos(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    v = np.sqrt(1 - cosa*cosa - cosb*cosb - cosg*cosg + 2*cosa*cosb*cosg)
    frac2cart = np.array([
        [ a, b*cosg,                  c*cosb],
        [0., b*sing, c*(cosa-cosb*cosg)/sing],
        [0.,     0.,                c*v/sing]
    ])
    xyzcart = np.dot(xyzfrac, frac2cart)
    return xyzcart
