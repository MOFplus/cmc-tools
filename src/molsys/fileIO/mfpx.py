import numpy
# from . import txyz  # read_body and write_body live now here in mfpx.py
import logging
from molsys.util.misc import argsorted
import molsys.util.images as images

"""
These functions implement the reading and writing of mfpx files

for the handling of metadata of topologies/embedding the system of keywords is largely extended.
Note: the code still reads old mfpx file of type top correctly but it writes only the new format.

TBI: check if the cromo file type is still needed and remove in case this is ok

"""

logger = logging.getLogger("molsys.io")

def read(mol, f):
    """
    Read mfpx file
    :Parameters:
        -f   (obj): mfpx file object or mfpx-like readable object
        -mol (obj): instance of a molclass
    """
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    ftype = 'xyz'
    lbuffer = f.readline().split()
    stop = False
    bbinfo = {}
    # new dir for topo metadata with defaults
    topoinfo = {}
    topoinfo["format"]    = "old"
    topoinfo["systrekey"] = "None"
    topoinfo["RCSRname"]  = "None"
    topoinfo["spgr"]      = "None" 
    topoinfo["coord_seq"] = "None"
    topoinfo["transitivity"] = "None"
    topoinfo["emapping"]  = "None"
    has_color = False
    while not stop:
        if lbuffer[0] != '#':
            mol.natoms = int(lbuffer[0])
            stop = True
        else:
            keyword = lbuffer[1]
            if keyword == 'type':
                ftype = lbuffer[2]
            elif keyword == 'cell':
                cellparams = [float(i) for i in lbuffer[2:8]]
                mol.set_cellparams(cellparams)
            elif keyword == 'cellvect':
                mol.periodic = True
                celllist = [float(i) for i in lbuffer[2:11]]
                cell = numpy.array(celllist)
                cell.shape = (3,3)
                mol.set_cell(cell)
            elif keyword == 'bbcenter':
                mol.is_bb = True
                bbinfo['center_type'] = lbuffer[2]
                if lbuffer[2] == 'special':
                    bbinfo['center_xyz'] = numpy.array([float(i) for i in lbuffer[3:6]])
            elif keyword == 'bbconn':
                mol.is_bb = True
                con_info = lbuffer[2:]
            elif keyword == 'bbatypes':
                mol.is_bb = True
                bbinfo['connector_atypes'] = lbuffer[2:]
            elif keyword == 'orient':
                orient = [int(i) for i in lbuffer[2:]]
                mol.orientation = orient
            elif keyword[:5] == "topo_":
                assert ftype == "topo"
                topokeyw = keyword[5:]
                if topokeyw == "format":
                    if lbuffer[2] == "new":
                        topoinfo["format"] = "new"
                    else:
                        logger.warning("unknown topo format")
                elif topokeyw == "systrekey":
                    topoinfo["systrekey"] = " ".join(lbuffer[2:])
                elif topokeyw == "RCSRname":
                    topoinfo["RCSRname"] = lbuffer[2]
                elif topokeyw == "spgr":
                    topoinfo["spgr"] = lbuffer[2]
                elif topokeyw == "coord_seq":
                    topoinfo["coord_seq"] = " ".join(lbuffer[2:])
                elif topokeyw == "transitivity":
                    topoinfo["transitivity"] = " ".join(lbuffer[2:])
                elif topokeyw == "emapping":
                    topoinfo["emapping"] = [int(i) for i in lbuffer[2:]]
                elif topokeyw == "color":
                    if lbuffer[2] == "True":
                        has_color = True
                else:
                    assert topokeyw not in topoinfo, "keyword %s appears twice" % ("topo_" + topokeyw)
                    topoinfo[topokeyw] = lbuffer[2]
            lbuffer = f.readline().split()
    if mol.is_bb == True: ftype = 'bb'
    ### read body
    if ftype == 'xyz':
        mol.elems, mol.xyz, mol.atypes, mol.conn, mol.fragtypes, mol.fragnumbers =\
            read_body(f,mol.natoms,frags=True)
    elif ftype == 'topo':
        # topo file so set the corresponding flags
        mol.is_topo =   True
        mol.use_pconn = True
        topo_new = False
        if topoinfo["format"] == "new":
            topo_new = True
        if topo_new:
            mol.set_nofrags()
            if has_color:
                mol.elems, mol.xyz, mol.atypes, mol.conn, mol.pconn, mol.pimages, mol.fragnumbers, vcolor, ecolor_conn=\
                read_body(f,mol.natoms, topo = True, topo_new=True, has_color=True)
            else:
                mol.elems, mol.xyz, mol.atypes, mol.conn, mol.pconn, mol.pimages, mol.fragnumbers =\
                read_body(f,mol.natoms, topo = True, topo_new=True)
            # if we read a new topo file we add the topo addon right away and fill it with the info from topoinfo
            mol.addon("topo")
            if topoinfo["coord_seq"] == "None":
                cs = None
            else:
                cs = topoinfo["coord_seq"].split("|")
            mol.topo.topoinfo = topoinfo   # RS why do we do this? there is _info and not topoinfo?? WHAT A MESS 
            mol.topo.set_topoinfo(skey=topoinfo["systrekey"],\
                                  vmapping=mol.fragnumbers,\
                                  emapping=topoinfo["emapping"],\
                                  spgr=topoinfo["spgr"],\
                                  transitivity=topoinfo["transitivity"],\
                                  RCSRname=topoinfo["RCSRname"],\
                                  coord_seq=cs)
            # if we have color (from metadata .. onyl for new format)
            if has_color:
                # ecolor is currently a redundant table accoridng to conn, but we need to map it on the edge tab called etab
                # we need only the entries where i<j
                mol.set_etab_from_conns()
                ecolor = [0 for i in range(len(mol.etab))] # is there a simpler way to do this? nedges?
                for i,c in enumerate(mol.conn):
                    for jj,j in enumerate(c): # Note: j is sthe second atom index but jj its index within conn
                        if i < j:
                            pi = mol.pimages[i][jj]
                            ei = mol.etab.index((i, j, pi))
                            ecolor[ei] = ecolor_conn[i][jj]
                mol.topo.set_color(vcolor, ecolor)
        else:
            mol.elems, mol.xyz, mol.atypes, mol.conn, mol.pconn, mol.pimages =\
                read_body(f,mol.natoms, topo = True)
            mol.set_nofrags()
    elif ftype == 'cromo':
        mol.is_topo =   True
        mol.is_cromo =   True
        mol.use_pconn = True
        mol.elems, mol.xyz, mol.atypes, mol.conn, mol.pconn, mol.pimages, mol.oconn =\
            read_body(f,mol.natoms,frags=True, topo = True, cromo = True)
        mol.set_nofrags()
    elif ftype == 'bb': # 
        connector, connector_atoms, connector_types = parse_connstring(mol,con_info)
        #bbinfo['connector'] = connector  ## that actually has to be an argument as the setup currently is coded.
        bbinfo['connector_atoms'] = connector_atoms
        bbinfo['connector_types'] = connector_types
        mol.elems, mol.xyz, mol.atypes, mol.conn, mol.fragtypes, mol.fragnumbers =\
            read_body(f,mol.natoms,frags=True, topo = False, cromo = False)
        mol.addon('bb')
        mol.bb.setup(connector,**bbinfo)
    else:
        ftype = 'xyz'
        logger.warning('Unknown mfpx file type specified. Using xyz as default')
        mol.elems, mol.xyz, mol.atypes, mol.conn, mol.fragtypes, mol.fragnumbers =\
                read_body(f,mol.natoms,frags=False)
    mol.set_ctab_from_conn(pconn_flag=mol.use_pconn)
    mol.set_etab_from_tabs()
    if ftype == 'cromo':
        mol.set_otab_from_oconn()
    ### pass bb info
    try:
        line = f.readline().split()
        if line != [] and line[0][:5] == 'angle':
            mol.angleterm = line
    except:
        pass
    return

def write(mol, f, fullcell = True, topoformat = "new"):
    """
    Routine, which writes an mfpx file
    :Parameters:
        -mol   (obj) : instance of a molsys class
        -f (obj) : file object or writable object
        -fullcell  (bool): flag to specify if complete cellvectors arre written
    """
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    if len(mol.fragtypes) == 0:
        mol.set_nofrags()
    if mol.is_topo:
        ftype = 'topo'
        if mol.use_pconn == False:
            # this is a topo object but without valid pconn. for writing we need to generate it
            mol.add_pconn()
        # this is for the new format
        topoinfo = {}
        if topoformat == "new":
            topoinfo["format"] = "new"
        else:
            topoinfo["format"] = "old"
        if "topo" in mol.loaded_addons:
            # add topo attributes to topoinfo
            for k in ["systrekey", "RCSRname", "spgr", "coord_seq", "transitivity", "sk_emapping", "has_color"]:
                v = getattr(mol.topo, k)
                if v != None:
                    if k == "has_color" and v == True:
                        topoinfo["color"] = "True"
                    elif type(v) == type([]):
                        if type(v[0]) == type(""):
                            # a list of strings .. use | as separator
                            topoinfo[k] = "|".join(v)
                        elif type(v[0]) == type(0):
                            # a list of integers .. 
                            topoinfo[k] = " ".join([str(i) for i in v])
                    else:
                        topoinfo[k] = v
            # now add all the stuff in the _info dir to topoinfo .. this needs to be simplified 
            for k in mol.topo._info:
                if k not in topoinfo:
                    topoinfo[k] = mol.topo._info[k]
        else:
            topoinfo["format"] = "old"
    elif mol.is_bb:
        ftype = "bb"
    else:
        ftype = 'xyz'
    f.write('# type %s\n' % ftype)
    if mol.bcond>0:
        if fullcell:
#            elif keyword == 'cellvect':
#                mol.periodic = True
#                celllist = [float(i) for i in lbuffer[2:11]]
#                cell = numpy.array(celllist)
#                cell.shape = (3,3)
#                mol.set_cell(cell)
            f.write('# cellvect %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %\
                    tuple(mol.cell.ravel()))
        else:
            f.write('# cell %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %\
                    tuple(mol.cellparams))
    if mol.is_bb:
        ### BLOCK DEFINITION ###################################################
        ### bbcenter write ###
        if mol.bb.center_type != 'special':
            f.write('# bbcenter %s\n' % mol.bb.center_type)
        else:
            f.write('# bbcenter %s %12.6f %12.6f %12.6f\n' %
                    tuple([mol.bb.center_type]+ mol.bb.center_xyz.tolist()))
        ### bbconn write ###
        # sort by connectors' types
        _argsorted_type = argsorted(mol.bb.connector_types)
        connectors_type = [mol.bb.connector_types[_] for _ in _argsorted_type]
        connector_atoms = [sorted(mol.bb.connector_atoms[_]) for _ in _argsorted_type]
        connectors      = [mol.bb.connector[_] for _ in _argsorted_type]
        # sort by connector atoms after connectors' types
        _sort_priority = [_*mol.natoms for _ in connectors_type]
        connector_atoms_priority = [
            [__+_sort_priority[_k] for __ in _] for _k,_ in enumerate(connector_atoms)
        ] # preparatory for nested sorting
        _argsorted_catoms = argsorted(connector_atoms_priority)
        connector_atoms = [connector_atoms[_] for _ in _argsorted_catoms]
        connectors      = [connectors[_]      for _ in _argsorted_catoms]
        # make bbcon line
        connstrings = ''
        ctype = 0
        for i,d in enumerate(mol.bb.connector_atoms):
            if mol.bb.connector_types[i] != ctype:
                ctype +=1
                connstrings += '/ ' # ctypes sep
            for j in d:
                connstrings = connstrings + str(j+1) +',' # connector_atoms sep
            connstrings = connstrings[0:-1] + '*' + str(connectors[i]+1)+' '
        # write bbconn line
        f.write('# bbconn %s\n' % connstrings)
        if mol.bb.connector_atypes is not None:
            # also write connector atypes
            temp_atypes = []
            for at in mol.bb.connector_atypes:
                if type(at) == type([]):
                    temp_atypes.append(",".join(at))
                else:
                    temp_atypes.append(at)
            f.write('# bbatypes %s\n' % " ".join(temp_atypes))
    if ftype == "topo":
        tks = list(topoinfo.keys())
        tks.sort()
        for k in tks:
            f.write("# topo_%s %s\n" % (k, topoinfo[k]))
    if hasattr(mol, "orientation"):
        ### ORIENTATION DEFINITION #############################################
        o = len(mol.orientation) * "%3d" % tuple(mol.orientation)
        f.write('# orient '+o+"\n")
    f.write('%i\n' % mol.natoms)
    if ftype == 'xyz' or ftype == "bb":
        write_body(f,mol)
    else:
        # this is all a hack .. restructure write/read body etc. .... currently we keep it makeing the mess even bigger .. sick!
        topo_new = False
        if topoinfo["format"] == "new":
            topo_new = True
            if mol.topo.sk_vmapping != None:
                # make sure that fragnumbers contains the skey_mapping of the topo addon (just for storing it)
                mol.fragnumbers = mol.topo.sk_vmapping
            else:
                # we use a new format but there is for some reason no systrekey => no sk_vmapping
                mol.fragnumbers = [0]*mol.natoms
        write_body(f,mol,topo=True, topo_new=topo_new)
    return

def parse_connstring(mol, con_info): 
    """ 
    Routines which parses the con_info string of a txyz or an mfpx file 
    :Parameters: 
        - mol      (obj) : instance of a molclass 
        - con_info (str) : string holding the connectors info 
    """ 
    connector             = [] 
    connector_atoms       = [] 
    connector_types       = [] 
    contype_count = 0 
    for c in con_info: 
        if c == "/": 
            contype_count += 1 
        else: 
            ss = c.split('*') # ss[0] is the dummy neighbors, ss[1] is the connector atom 
            if len(ss) > 2: 
                raise IOError('This is not a proper BB file, convert with script before!') 
            elif len(ss) < 2: # neighbor == connector 
                ss *= 2 
            stt = ss[0].split(',') 
            connector.append(int(ss[1])-1) 
            connector_types.append(contype_count) 
            connector_atoms.append((numpy.array([int(i) for i in stt]) -1).tolist()) 
    return connector, connector_atoms, connector_types

## this is read_body and write_body from txyz. due to the specific needs and special format of mfpx files these functions have been moved over here (RS)


def read_body(f, natoms, frags = True, topo = False, cromo = False, topo_new=False, has_color=False):
    """
    Routine, which reads the body of a txyz or a mfpx file
    :Parameters:
        -f      (obj)  : fileobject
        -natoms (int)  : number of atoms in body
        -frags  (bool) : flag to specify if fragment info is in body or not
        -topo   (bool) : flag to specify if pconn info is in body or not
        -cromo  (bool) : flag to specify if oconn info is in body or not

    RS this completely silly .. instead of passing mol object we return all kinds of things depending on settings
                                really a big mess which needs to be cleaned up.
                                do we still need the cromo format? ever used?

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
    vcolor     = []
    ecolor     = []
    if has_color:
        assert topo
        assert topo_new
    if topo: 
        frags=False
    for i in range(natoms):
        lbuffer = f.readline().split()
        xyz.append([float(j) for j in lbuffer[2:5]])
        e = lbuffer[1].lower()
        if has_color:
            e, vc = e.split("_")[:2]
            vcolor.append(int(vc))
        elems.append(e)
        t = lbuffer[5]
        atypes.append(t)
        if frags == True:
            fragtypes.append(lbuffer[6])
            fragnumbers.append(int(lbuffer[7]))
            offset = 2
        else:
            if topo_new:
                fragtypes.append('0')  # why "0" and not "-1" ???
                fragnumbers.append(int(lbuffer[6]))
                offset = 1
            else:
                fragtypes.append('0') # why "0" and not "-1" ???
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
            # this is to read the connectivity info for topos .. for topo_new=True and has_color=True the format is i/p/ec whre ec is the edge color
            txt = lbuffer[6+offset:]
            a = [[int(j) for j in i.split('/')] for i in txt]
            if has_color:
                c,pc,pim,ec = [i[0]-1 for i in a], [images[i[1]] for i in a], [i[1] for i in a], [i[2] for i in a]
                ecolor.append(ec)
            else:
                c,pc,pim = [i[0]-1 for i in a], [images[i[1]] for i in a], [i[1] for i in a]
            conn.append(c)
            pconn.append(pc)
            pimages.append(pim)
    if topo:
        if cromo:
            return elems, numpy.array(xyz), atypes, conn, pconn, pimages, oconn
        else:
            if topo_new:
                if has_color:
                    return elems, numpy.array(xyz), atypes, conn, pconn, pimages, fragnumbers, vcolor, ecolor
                else:
                    return elems, numpy.array(xyz), atypes, conn, pconn, pimages, fragnumbers
            else:
#                return elems, numpy.array(xyz), atypes, conn, fragtypes, fragnumbers, pconn
                return elems, numpy.array(xyz), atypes, conn, pconn, pimages
    else:
        return elems, numpy.array(xyz), atypes, conn, fragtypes, fragnumbers


def write_body(f, mol, frags=True, topo=False, pbc=True, plain=False, topo_new=False):
    """
    Routine, which writes the body of a txyz or a mfpx file
    :Parameters:
        -f      (obj)  : fileobject
        -mol    (obj)  : instance of molsys object
        -frags  (bool) : flag to specify if fragment info should be in body or not
        -topo   (bool) : flag to specigy if pconn info should be in body or not
        -pbc    (bool) : if False, removes connectivity out of the box (meant for visualization)
        -plain  (bool) : plain Tinker file supported by molden
        -topo_new(bool): if True the new format with skey mapping (hidden in fragnumbers) is written
    """
    if topo:
        frags = False   #from now on this is convention! yuck .. why do we specify it in the first place then?
    if topo: pconn = mol.pconn
    if frags:
        fragtypes   = mol.fragtypes
        fragnumbers = mol.fragnumbers
    else:
        fragtypes = None
        if topo_new:
            fragnumbers = mol.fragnumbers ## this hides the skey mapping
        else:
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
        # RS add color info in mfpx .. this is all a big mess and needs to be cleaned up (2022)
        e = elems[i]
        if topo_new:
            if mol.topo.has_color:
                e += "_%d" % mol.topo.vcolor[i] # only for topo_new we can expect that the mol object has a topo addon
        line = ("%3d %-3s" + 3*"%12.6f" + "   %-24s") % \
            (i+1, e, xyzl[i][0], xyzl[i][1], xyzl[i][2], atypes[i])
        if frags is True:
            line += ("%-16s %5d ") % (fragtypes[i], fragnumbers[i])
        elif topo_new is True:
            # write fragnumbers as skey mappings if topo_new is specified
            line += ("%5d ") % (fragnumbers[i])
        conn = (numpy.array(cnct[i])+1).tolist()
        if len(conn) != 0:
            if topo:
                pimg = []
                pimg_inv = []
                for pc in pconn[i]:
                    pc_inv = pc*-1
                    for ii,img in enumerate(images):
                        # if pc is None:
                        #     raise TypeError("Something went VERY BAD in pconn")
                        if all(img==pc):
                            pimg.append(ii)
                            # break
                        if all(img==pc_inv):
                            pimg_inv.append(ii)
                ecol = len(conn)*[""]
                if topo_new:
                    if mol.topo.has_color:
                        ecol = []
                        for cc,pp,ppi in zip(conn,pimg, pimg_inv):
                            if i < cc-1:
                                ecol.append("/%d" % mol.topo.ecolor[mol.etab.index((i, cc-1, pp))])
                            else:
                                ecol.append("/%d" % mol.topo.ecolor[mol.etab.index((cc-1, i, ppi))])
                for cc,pp,ec in zip(conn,pimg,ecol):
                    if pp < 10:
                        line +="%8d/%1d%s " % (cc,pp,ec)
                    else:
                        line += "%7d/%2d%s " % (cc,pp,ec)
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
