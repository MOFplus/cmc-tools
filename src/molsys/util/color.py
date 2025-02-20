from molsys.util.images import arr2idx, idx2arr
import numpy as np
import copy

### COLOR UTILITIES ###
# color dictionaries based on default molden element colors
ecolor2elem = [
    "b" ,"f" ,"n" ,"o" ,"c" ,"he","ne","ge","li","s" ,"cl","p" ,"al","si",
]
elem2ecolor = dict(ke[::-1] for ke in enumerate(ecolor2elem))
maxecolor = len(ecolor2elem)
vcolor2elem = [
    "n" ,"o" ,"b" ,"f" ,"c" ,"he","ne","ge","li","s" ,"cl","p" ,"si","al",
]
elem2vcolor = dict(kv[::-1] for kv in enumerate(vcolor2elem))
maxvcolor = len(vcolor2elem)

# string conversion tools: elem+atype+color <-> string #
def elematypecolor2string(elem, atype, color):
    """
    return formatted string from element, atomtype, and color
    """
    return "%s_%s/%s" % (elem, atype, color)

eac2str = elematypecolor2string #nickname

def string2elematypecolor(st):
    """
    return element, atomtype, and color from a given formatted string.

    Color is after the last slash
    Element is before the first underscore
    Atype is everything in the middle
    """
    self.assert_eacstr(st)
    colorsplit = st.split("/")
    rest, color = colorsplit[:-1], colorsplit[-1]
    rest = "".join(rest)
    elemsplit = rest.split("_")
    elem, atype = elemsplit[0], elemsplit[1:]
    return elem, atype, color

str2eac = string2elematypecolor #nickname

def assert_eacstr(st):
    """
    check format of element_atomtype/color string
    """
    assert st.index("_"), "string is not well-formatted: no \"_\" found"
    assert st.rindex("/"), "string is not well-formatted: no \"/\" found"
    assert st.index("_") < st.rindex("/"), "string is not well-formatted: first \"_\" after last \"/\""
    return

# make mol objects out of graph elements (edges and/or vertices) #

def make_emol(m, alpha=2, ecolors=None, etab=None, ec2e=None):
    """
    make mol object out of edge colors
    """
    if etab is None:
        etab = m.etab
    if ecolors is None:
        ecolors = [maxecolor-1]*len(etab)
    if ec2e is None:
        ec2e = ecolor2elem
    # coordinates #
    ralpha = 1./alpha #reverse alpha
    calpha = 1-ralpha #one's complement of reverse alpha
    xyz_a = []
    xyz_c = []
    if m.use_pconn:
        for ei,ej,p in etab: ### SELECTION TBI
            xyz_ai = m.xyz[ei]
            xyz_a.append(xyz_ai)
            xyz_ci = m.get_neighb_coords_(ei,ej,idx2arr[p])
            xyz_c.append(xyz_ci)
    else:
        for ei,ej in etab: ### SELECTION TBI
            xyz_ai = m.xyz[ei]
            xyz_a.append(xyz_ai)
            xyz_ci =  m.xyz[ej]
            xyz_c.append(xyz_ci)
    xyz_a = np.array(xyz_a)
    xyz_c = np.array(xyz_c)
    if alpha == 2:
        xyz_c = ralpha*(xyz_a + xyz_c)
        me = m.from_array(xyz_c, use_pconn=m.use_pconn)
    else:
        xyz_c1 = calpha*xyz_a + ralpha*xyz_c
        xyz_c2 = ralpha*xyz_a + calpha*xyz_c
        me = m.from_array(np.vstack([xyz_c1,xyz_c2]), use_pconn=m.use_pconn)
    me.xyz += 1e-24 # to avoid underflows when wrapping
    # attributes #
    if hasattr(m,'cell'):
        me.set_cell(m.cell)
    if hasattr(m,'supercell'):
        me.supercell = m.supercell[:]
    me.is_topo = True
    if m.use_pconn:
        me.use_pconn = True
    # elements/atomtypes #
    if alpha == 2:
        me.elems = [ec2e[v] for v in ecolors] # N.B.: no connectivity
        me.atypes = [0 for v in ecolors]
    else:
        me.elems = [ec2e[v] for v in list(ecolors)*2] # with connectivity
        me.atypes = [0 for v in list(ecolors)*2]
    # connectivity #
    new_etab = []
    if alpha == 2:
        if m.use_pconn:
            pimg = me.get_frac_xyz()//1
            me.xyz -= np.dot(pimg,me.cell)
            for k,(i,j,p) in enumerate(etab):
                newe1 = i,k,arr2idx[pimg[k]]
                newe2 = j,k,arr2idx[pimg[k]-idx2arr[p]]
                new_etab.append(newe1)
                new_etab.append(newe2)
        else:
            new_etab = etab[:]
    else:
        ctab = [(i,i+me.natoms//2) for i in range(me.natoms//2)]
        me.set_ctab(ctab, conn_flag=True)
        if m.use_pconn:
            pimg = me.get_frac_xyz()//1
            me.xyz -= np.dot(pimg,me.cell)
            ptab = [arr2idx[pimg[i+me.natoms//2]-pimg[i]] for i in range(me.natoms//2)]
            me.set_ptab(ptab, pconn_flag=True)
            for k,(i,j,p) in enumerate(etab):
                newe1 = i,k,arr2idx[pimg[k]]
                newe2 = j,k+len(etab),arr2idx[pimg[k+len(etab)]-idx2arr[p]]
                new_etab.append(newe1)
                new_etab.append(newe2)
        else:
            new_etab = ctab[:]
    me.new_etab = new_etab ### MOVE TO make_mol
    return me

def make_vmol(m, vcolors=None, vc2e=None):
    """
    make mol object out of graph vertices
    """
    if vcolors is None:
        vcolors = [maxvcolor-1]*m.natoms
    if vc2e is None:
        vc2e = vcolor2elem
    mv = copy.copy(m)
    for i in range(mv.natoms):
        mv.atypes[i] = elematypecolor2string(
            mv.elems[i],
            mv.atypes[i],
            vcolors[i]
        )
        mv.elems[i] = vc2e[vcolors[i]]
    if hasattr(m,'cell'): mv.set_cell(m.cell)
    if hasattr(m,'supercell'): mv.supercell = m.supercell[:]
    return mv

def make_mol(m, alpha=2, ecolors=None, vcolors=None, etab=None,
    use_edge=True, use_vertex=True, ec2e=None, vc2e=None):
    """
    make mol object out of graph elements (edges and/or vertices)
    if both edges and vertices, it takes care of the connectivity too
    """
    if use_edge and use_vertex:
        mm = make_emol(m, alpha=alpha, ecolors=ecolors, etab=etab, ec2e=ec2e)
        ne = mm.natoms # at the moment just edge mol
        mv = make_vmol(m, vcolors=vcolors, vc2e=vc2e)
        mm.add_mol(mv) # N.B.: in THIS EXACT ORDER, otherwise KO connectivity
        ### connectivity ###
        ctab = []
        if m.use_pconn:
            ptab = []
        if m.use_pconn:
            for i,j,p in mm.new_etab:
                ctab.append((i+ne,j))
                ptab.append(p)
        else:
            for i,j in mm.new_etab:
                ctab.append((i+ne,j))
        ctab += mm.ctab
        mm.set_ctab(ctab, conn_flag=True)
        if m.use_pconn:
            ptab += mm.ptab
            mm.set_ptab(ptab, pconn_flag=True)
        mm.sort_tabs(etab_flag=True, conn_flag=True)
    elif use_edge:
        mm = make_emol(m, alpha=alpha, ecolors=ecolors, etab=etab)
    elif use_vertex:
        mm = m.make_vmol(vcolors=vcolors)
    if hasattr(m,'cell'): mm.set_cell(m.cell)
    if hasattr(m,'supercell'): mm.supercell = m.supercell[:]
    return mm

