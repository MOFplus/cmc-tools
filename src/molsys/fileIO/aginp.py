import numpy
import string
from . import txyz
import logging

logger = logging.getLogger("molsys.io")

def read(mol, f, connectors=False, morebonds = False):
    """
    Routine, which reads an AuToGraFS input file
    :Parameters:
        -f    (obj) : AuToGraFS input file object
        -mol  (obj) : instance of a molclass
        -connectors(bool): additional connectors information
        -morebonds (bool): additional bonds information
    """
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    elems, xyz, atypes, qmmmtypes, bonds = [], [], [], [], []
    for line in f:
        split = line.split()
        if not split:
            pass
        elif split[0][:5] == "Data:":
            strip = line.strip("Data:")
            split = strip.split("=")
            split = [s.strip() for s in split]
            agattribute = split[0]
            agvalue = split[1]
            setattr(mol,agattribute,agvalue)
        elif split[0] not in ("GEOMETRY","END"):
            elem, coord, atype, qmmmtype, bond = split[0], split[1:4], split[4], split[5], split[6]
            atype = atype.split("=")[1]
            qmmmtype = qmmmtype.split("=")[1]
            bond = bond.split("=")[1]
            bond = bond.split(":")
            bond = [b.split('/')[0] for b in bond]
            bond = map(int,bond)
            bond = [b-1 for b in bond]
            elems.append(elem)
            xyz.append(coord)
            atypes.append(atype)
            qmmmtypes.append(qmmmtype)
            bonds.append(bond)
    xyz = numpy.array(xyz, dtype=numpy.float)
    mol.natoms = xyz.shape[0]
    mol.elems = elems
    mol.xyz = xyz
    mol.atypes = atypes
    mol.qmmmtypes = qmmmtypes
    mol.conn = bonds
    if connectors:
        econn = [e for (e,el) in enumerate(elems) if el == "X"]
        ag2weaver_connectors(mol,econn)
    if morebonds:
        raise NotImplementedError("additional bonds information is not implemented")
    mol.set_nofrags() ###dunno why
    return

def ag2weaver_connectors(mol,agconn):
    """
    :Parameters:
        -mol  (obj) : instance of a molclass
        -agconn(list of int): AuToGraFS connectors by indices (atoms starts from 0)
    :Attributes:
        -agneigh(nested list of int): list of connectors, grouped by multiplicity
        -newconn(list of int): list of connectors, ungrouped
    """
    ###CHECK NUMBER OF LINKED ATOMS PER CONNECTOR (should be 1 per atom)
    agneigh = map(mol.conn.__getitem__, agconn)
    lagneigh = map(len,agneigh)
    lagn = None #[i for i in lagneigh if i != 1]
    if lagn:
        raise NotImplementedError("more than one real atom per dummy")
    ### function starts here ###
    else:
        mol.is_bb = True
        agconn = numpy.array(agconn)
        mol.center_point = "coc" #default, center of connectors
        newconn = sum(agneigh, [])
        newconn = numpy.array(newconn)
        ### shift due to atom removal
        shift = numpy.where(newconn > agconn[:,numpy.newaxis], 1, 0).sum(axis=0)
        agneigh = [[i-shift[list(newconn).index(i)] for i in neigh] for neigh in agneigh]
        mol.delete_atoms(agconn)
        ###starts from 1 (txyz-like)
        mfpxneigh = [[i+1 for i in neigh] for neigh in agneigh]
        half_con_info = [','.join(map(str,neigh)) for neigh in mfpxneigh]
        ### half_con_info = [','.join(map(str,neigh)) for neigh in mfpxneigh][:-1] ###CHANGE, JUST TEST
        con_info = ["%s*%s" % i for i in zip(half_con_info,half_con_info)]
        txyz.parse_connstring(mol, con_info, new=True)
    return
