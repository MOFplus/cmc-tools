import numpy
import string
import molsys.util.images as images
from collections import Counter
import copy
import pprint
import logging

logger = logging.getLogger("molsys.io")

def read(mol, f,):
    """
    Routine, which reads an txyz file
    :Parameters:
        -f    (obj) : txyz file object
        -mol  (obj) : instance of a molclass
    """
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    lbuffer = f.readline().split()
    mol.natoms = int(lbuffer[0])
    if len(lbuffer) >1 and lbuffer[1] == "molden": lbuffer = [lbuffer[0]]
    if len(lbuffer) > 1 and lbuffer[1] not in ['special','coc','com']:
        boundarycond = 3
        if lbuffer[1] == "#":
            celllist = [float(i) for i in lbuffer[2:11]]
            cell = numpy.array(celllist)
            cell.shape = (3,3)
            mol.set_cell(cell)
        else:
            cellparams = [float(i) for i in lbuffer[1:7]]
            mol.set_cellparams(cellparams)
        if ((cellparams[3]==90.0) and (cellparams[4]==90.0) and (cellparams[5]==90.0)):
            boundarycond=2
            if ((cellparams[0]==cellparams[1])and(cellparams[1]==cellparams[2])and\
                (cellparams[0]==cellparams[2])):
                    boundarycond=1
    mol.elems, mol.xyz, mol.atypes, mol.conn, mol.fragtypes, mol.fragnumbers =\
            read_body(f,mol.natoms,frags=False)
    mol.set_ctab_from_conn() 
    mol.set_etab_from_tabs()
    ### this has to go at some point
    mol.set_nofrags()
    return

def parse_connstring_new(mol, con_info, **kwargs):
    """
    Routines which parses the con_info string of a txyz or an mfpx file
    :Parameters:
        - mol      (obj) : instance of a molclass
        - con_info (str) : string holding the connectors info
    """
    mol.connector_atoms       = []
    mol.connector_dummies     = []
    mol.connectors            = []
    mol.connectors_complexity = []
    mol.connectors_group      = []
    mol.connectors_type       = []
    contype_count = 0
    for icon, con in enumerate(con_info):
        if con == "/":
            contype_count += 1
            continue
        ### PARSE ####
        logger.debug(con)
        line = con[:]
        markcount = line.count("?")
        starcount = line.count("*")
        if markcount == 0:
            complexity = ""
        elif markcount == 1:
            line, complexity = line.split('?')
        else:
            logger.error("More than one question mark in con_info group")
            raise ValueError
        if starcount == 0:
            connectors = ""
        elif starcount == 1:
            line, connectors = line.split('*')
        else:
            logger.error("More than one asterisk in con_info group")
            raise ValueError
        atoms = line.split(",")
        line = con #reset, debugging purpose
        logger.debug("a:%s c:%s t:%s *:%s ?:%s" % 
            (atoms, connectors, complexity, starcount, markcount))
        complexity = complexity.split()
        if complexity != []:
            complexity = complexity[0].split(",")
        connectors = connectors.split()
        if connectors != []:
            connectors = connectors[0].split(",")
        logger.debug("a:%s c:%s t:%s *:%s ?:%s" % 
            (atoms, connectors, complexity, starcount, markcount))
        logger.debug("la:%s lc:%s lt:%s *:%s ?:%s" % 
            (len(atoms), len(connectors), len(complexity), starcount, markcount))
        ### HANDLE ###
        if len(atoms) == 0:
            logger.error("No atoms in con_info group")
            raise ValueError
        if len(connectors) == 0:
            connectors = atoms[:]
        if len(complexity) == 0:
            complexity = [1]*len(atoms)
        elif len(complexity) == 1:
            complexity = complexity*len(atoms)
        if len(complexity) != len(atoms):
            logger.error("Complexity can only be: implicit OR one for all OR assigned per each")
            raise ValueError
        atoms = [int(a) - 1 for a in atoms] #python indexing
        connectors = [int(c) - 1 for c in connectors] #python indexing
        complexity = map(int,complexity)
        logger.debug("a:%s c:%s t:%s *:%s ?:%s" % 
            (atoms, connectors, complexity, starcount, markcount))
        logger.debug("la:%s lc:%s lt:%s *:%s ?:%s" % 
            (len(atoms), len(connectors), len(complexity), starcount, markcount))
        ### SET ######
        mol.connector_atoms.append(atoms) #((numpy.array(map(int,stt)) -1).tolist())
        for a in atoms:
            if mol.elems[a].lower() == "x":
                mol.connector_dummies.append(a) # simplest case only with two atoms being the connecting atoms
        mol.connectors.append(connectors) #(int(ss[1])-1)
        mol.connectors_complexity.append(complexity)
        mol.connectors_group.append(icon - contype_count)
        mol.connectors_type.append(contype_count)
    mol.connectors = numpy.array(mol.connectors)
    return


def read_body(f, natoms, frags = True, topo = False, cromo = False):
    """
    Routine, which reads the body of a txyz or a mfpx file
    :Parameters:
        -f      (obj)  : fileobject
        -natoms (int)  : number of atoms in body
        -frags  (bool) : flag to specify if fragment info is in body or not
        -topo   (bool) : flag to specify if pconn info is in body or not
        -cromo  (bool) : flag to specify if oconn info is in body or not
    """
    elems       = []
    xyz         = []
    atypes      = []
    conn        = []
    fragtypes   = []
    fragnumbers = []
    pconn       = []
    pimages     = []
    oconn      = []
    if topo: frags=False
    for i in range(natoms):
        lbuffer = f.readline().split()
        xyz.append([float(i) for i in lbuffer[2:5]])
        elems.append(lbuffer[1].lower())
        t = lbuffer[5]
        atypes.append(t)
        if frags == True:
            fragtypes.append(lbuffer[6])
            fragnumbers.append(int(lbuffer[7]))
            offset = 2
        else:
            fragtypes.append('0')
            fragnumbers.append(0)
            offset = 0
        if not topo:
            conn.append((numpy.array([int(i) for i in lbuffer[6+offset:]])-1).tolist())
        elif cromo:
            txt = lbuffer[6+offset:]
            a = [[int(j) for j in i.split('/')] for i in txt]
            c,pc,pim,oc = [i[0]-1 for i in a], [images[i[1]] for i in a], [i[1] for i in a], [i[2] for i in a]
            conn.append(c)
            pconn.append(pc)
            pimages.append(pim)
            oconn.append(oc)
        else:
            txt = lbuffer[6+offset:]
            a = [[int(j) for j in i.split('/')] for i in txt]
            c,pc,pim = [i[0]-1 for i in a], [images[i[1]] for i in a], [i[1] for i in a]
            conn.append(c)
            pconn.append(pc)
            pimages.append(pim)
    if topo:
        if cromo:
            return elems, numpy.array(xyz), atypes, conn, pconn, pimages, oconn
        else:
#            return elems, numpy.array(xyz), atypes, conn, fragtypes, fragnumbers, pconn
            return elems, numpy.array(xyz), atypes, conn, pconn, pimages
    else:
        return elems, numpy.array(xyz), atypes, conn, fragtypes, fragnumbers


def write_body(f, mol, frags=True, topo=False, pbc=True, plain=False):
    """
    Routine, which writes the body of a txyz file
    :Parameters:
        -f      (obj)  : fileobject
        -mol    (obj)  : instance of molsys object
        -frags  (bool) : flag to specify if fragment info should be in body or not
        -topo   (bool) : flag to specigy if pconn info should be in body or not
        -pbc    (bool) : if False, removes connectivity out of the box (meant for visualization)
        -plain  (bool) : plain Tinker file supported by molden
    """
    if topo:
        frags = False   #from now on this is convention!
    if topo: pconn = mol.pconn
    if frags:
        fragtypes   = mol.fragtypes
        fragnumbers = mol.fragnumbers
    else:
        fragtypes = None
        fragnumbers = None
    elems       = mol.elems
    xyz         = mol.xyz
    cnct        = mol.conn
    natoms      = mol.natoms
    atypes      = mol.atypes
    if mol.cellparams is not None and not pbc:
        mol.set_conn_nopbc()
        cnct = mol.conn_nopbc
    ### BUG but feature so removable ###
    #    atoms_withconn = mol.atoms_withconn_nopbc[:]
    #    offset = numpy.zeros(natoms, 'int')
    #    for i in range(natoms):
    #        if i not in atoms_withconn:
    #            offset[i:] += 1
    #    cnct = [
    #        [
    #            # subtract the j-th offset to atom j in i-th conn if j in atoms_withconn
    #            j-offset[j] for j in cnct[i] if j in atoms_withconn
    #        ]
    #        # if atom i in atoms_withconn
    #        for i in range(natoms) if i in atoms_withconn
    #    ]
    #    natoms = len(atoms_withconn)
    #    xyz    = xyz[atoms_withconn]
    #    elems  = numpy.take(elems, atoms_withconn).tolist()
    #    atypes = numpy.take(atypes, atoms_withconn).tolist()
    #    if frags:
    #        fragtypes   = numpy.take(fragtypes, atoms_withconn).tolist()
    #        fragnumbers = numpy.take(fragnumbers, atoms_withconn).tolist()
    if plain:
        if frags:
            fragtypes = [None]*len(atypes)
            fragnumbers = [None]*len(atypes)
            oldatypes = list(zip(atypes, fragtypes, fragnumbers))
        else:
            oldatypes = atypes[:]
        # unique atomtypes
        u_atypes = set(Counter(oldatypes).keys())
        u_atypes -= set([a for a in u_atypes if str(a).isdigit()])
        u_atypes = sorted(list(u_atypes))
        # old2new atomtypes
        o2n_atypes = {e:i for i,e in enumerate(u_atypes)}
        n2o_atypes = {i:e for i,e in enumerate(u_atypes)}
        atypes = [a if str(a).isdigit() else o2n_atypes[a] for a in oldatypes]
        frags = False ### encoded in one column only
    xyzl = xyz.tolist()
    for i in range(natoms):
        line = ("%3d %-3s" + 3*"%12.6f" + "   %-24s") % \
            (i+1, elems[i], xyzl[i][0], xyzl[i][1], xyzl[i][2], atypes[i])
        if frags == True:
            line += ("%-16s %5d ") % (fragtypes[i], fragnumbers[i])
        conn = (numpy.array(cnct[i])+1).tolist()
        if len(conn) != 0:
            if topo:
                pimg = []
                for pc in pconn[i]:
                    for ii,img in enumerate(images):
                        if pc is None:
                            raise TypeError("Something went VERY BAD in pconn")
                        if all(img==pc):
                            pimg.append(ii)
                            break
                for cc,pp in zip(conn,pimg):
                    if pp < 10:
                        line +="%8d/%1d " % (cc,pp)
                    else:
                        line += "%7d/%2d " % (cc,pp)
            else:
                line += (len(conn)*"%7d ") % tuple(conn)
        f.write("%s \n" % line)
    if plain:
        if frags:
            f.write("### atype: (old_atype, fragment_type, fragment_number)\n")
            n2o_fmt = pprint.pformat(n2o_atypes, indent=4)
        else:
            f.write("### atype: old_atype\n")
            n2o_fmt = pprint.pformat(n2o_atypes, indent=4, width=1) #force \n
        n2o_fmt = n2o_fmt.strip("{}")
        n2o_fmt = "{\n " + n2o_fmt + "\n}\n"
        f.write(n2o_fmt)
    return


def write(mol, f, topo = False, frags = False, pbc=True, plain=False):
    """
    Routine, which writes an txyz file
    :Parameters:
        -mol    (obj) : instance of a molclass
        -f (obj) : file object or writable object
        -topo   (bool): flag top specify if pconn should be in txyz file or not
    """
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    cellparams = mol.cellparams
    if cellparams is not None:
        ### BUG but feature so removable ###
        #if pbc:
        #    natoms = mol.natoms
        #else:
        #    mol.remove_conn_pbc()
        #    natoms = len(mol.atoms_withconn_nopbc)
        natoms = mol.natoms
        f.write("%5d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n" % tuple([natoms]+list(cellparams)))
    else:
        f.write("%5d  generated by molsys\n" % mol.natoms)
    write_body(f, mol, topo=topo, frags=frags, pbc=pbc, plain=plain)
    return
