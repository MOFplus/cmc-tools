import numpy as np
from pprint import pprint
import sys

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

### CONSTANTS ###

digits = list('0123456789')

### molecule ###
avail_mol_type = ['SMALL', 'BIOPOLYMER', 'PROTEIN', 'NUCLEIC_ACID', 'SACCHARIDE']
avail_mstatus_bits = ['system', 'invalid_charges', 'analyzed', 'substituted', 'altered', 'ref_angle']
### atom ###
avail_charge_type = ['NO_CHARGES', 'DEL_RE', 'GASTEIGER', 'GAST_HUCK', 'HUCKEL', 'PULLMAN', 'GAUSS80_CHARGES', 'AMPAC_CHARGES', 'MULLIKEN_CHARGES', 'DICT_CHARGES', 'MMFF94_CHARGES', 'USER_CHARGES']
### bond ###
avail_bond_type = {
    '1' : 'single',
    '2' : 'double',
    '3' : 'triple',
    'am': 'amide',
    'ar': 'aromatic',
    'du': 'dummy',
    'un': 'unknown',
    'nc': 'notconnected',
    'po': 'polymeric',
    'dl': 'delocalized'
}
avail_bstatus_bits = [None, 'TYPECOL', 'GROUP', 'CAP', 'BACKBONE', 'DICT', 'INTERRES']
### substructure ###
avail_subst_type = [None, 'temp', 'perm', 'residue', 'group', 'domain']
avail_subst_status = [None, 'LEAF', 'ROOT', 'TYPECOL', 'DICT', 'BACKWARD', 'BLOCK']

### MARKERS ### by specification
markers = {
    "@<TRIPOS>MOLECULE":     "molecule",
    "@<TRIPOS>ATOM":         "atom",
    "@<TRIPOS>BOND":         "bond",
    "@<TRIPOS>SUBSTRUCTURE": "substructure",
    "@<TRIPOS>CRYSIN":       "crysin",
}

def MOL2_read_blocks(f):
    """
    Read MOL2 body as blocks
    Blocks supported: ATOM, BOND, SUBSTRUCTURE, CRYSIN
    Sharped ends are stored as comments
    Not assignable lines are stored in the anonymous block
    """
    ### INITIALIZE ###
    blocks = {
        "comments"     : [],
        "anonymous"    : [],
        "molecule"     : [],
        "atom"         : [],
        "bond"         : [],
        "substructure" : [],
        "crysin"       : [],
    }
    ### READ ###
    collector = blocks["anonymous"]
    for line in f:
        if not line:
            continue
        strip = line.strip()
        split = strip.split('#')
        text = split[0]
        comment = '#'.join(split[1:])
        if comment:
            blocks["comments"].append(comment)
        if not text:
            continue
        if text in markers:
            collector = blocks[markers[text]]
            continue
        collector.append(text)
    return blocks

### MOLECULE ###

def MOL2_setup_MOLECULE(block_molecule):
    """
    Setup MOL2 MOLECULE block as mol attributes
    """
    block = Bunch()
    ### get ###
    group_molecule = block_molecule[:4]
    if len(block_molecule) > 4: # by specifications, either both or none
        mstatus_bits = block_molecule[4].strip()
        mol_comment = '\n'.join(block_molecule[5:])
    mol_name, num_misc, mol_type, charge_type = group_molecule

    ### edit ###
    mol_name = mol_name.strip()
    mol_type = mol_type.strip()
    charge_type = charge_type.strip()
    num_misc = num_misc.split()

    ### set ###
    num_atoms, num_bonds, num_subst, num_feat, num_sets = -1, -1, -1, -1, -1
    num_atoms = int(num_misc[0])
    if len(num_misc) > 1:
        num_bonds = int(num_misc[1])
    if len(num_misc) > 2:
        num_subst = int(num_misc[2])
    if len(num_misc) > 3:
        num_feat =  int(num_misc[3])
    if len(num_misc) > 4:
        num_sets =  int(num_misc[4])

    ### check ###
    if list(set(mstatus_bits)) != ['*']:
        assert mstatus_bits in avail_mstatus_bits
    assert mol_type in avail_mol_type
    assert charge_type in avail_charge_type

    ### return ###
    block.mol_name    = mol_name
    block.num_atoms   = num_atoms
    block.num_bonds   = num_bonds
    block.num_subst   = num_subst
    block.num_feat    = num_feat
    block.num_sets    = num_sets
    block.mol_type    = mol_type
    block.charge_type = charge_type
    return block

### ATOM ###

def MOL2_setup_ATOM(block_atom):
    """
    Setup MOL2 ATOM block as mol attributes
    """
    block = Bunch()
    ### get ###
    natoms = len(block_atom)
    #assert natoms == num_atoms
    aidxs    = [None] * natoms
    anames   = [None] * natoms
    atypes   = [None] * natoms
    resids   = [None] * natoms
    resnames = [None] * natoms
    charges  = [None] * natoms
    coords   = np.zeros((natoms, 3))

    ### set ###
    for i,line in enumerate(block_atom):
        split = line.split()
        aidx, aname, x, y, z, atype, resid, resname, charge = split
        aidxs[i], anames[i], atypes[i], resids[i], resnames[i], charges[i] = int(aidx), aname, atype, int(resid), resname, float(charge)
        coords[i,:] = x, y, z

    ### check ###
    # idxs
    assert range(1,natoms+1) == aidxs

    ### edit ###
    # back to python indexing
    aidxs = [i-1 for i in aidxs]
    resids = [i-1 for i in resids]

    ### return ###
    block.natoms   = natoms
    block.aidxs    = aidxs
    block.anames   = anames
    block.atypes   = atypes
    block.resids   = resids
    block.resnames = resnames
    block.charges  = charges
    block.coords   = coords
    return block

### BOND ###

def MOL2_setup_BOND(block_bond):
    """
    Setup MOL2 BOND block as mol attributes
    """
    block = Bunch()
    ### get ###
    nbonds = len(block_bond)
    #assert nbonds == num_bonds
    bidxs         = [None] * nbonds
    asrcs         = [None] * nbonds
    atgts         = [None] * nbonds
    btypes        = [None] * nbonds
    bstatus_bitss = [None] * nbonds

    ### set ###
    for i,line in enumerate(block_bond):
        split = line.split()
        bstatus_bits = None
        bidx, asrc, atgt, btype = split[:4]
        if len(split) > 4:
            bstatus_bits = split[4]
        else:
            bidxs[i], asrcs[i], atgts[i], btypes[i], bstatus_bitss[i] = int(bidx), int(asrc), int(atgt), btype, bstatus_bits

    ### check ###
    # idxs
    assert range(1,nbonds+1) == bidxs
    # avail
    notavail_bstatus_bits = list(set(bstatus_bitss) - set(avail_bstatus_bits))
    assert len(notavail_bstatus_bits) == 0, "This is not available: %s" % notavail_bstatus_bits
    ### edit ###
    # back to python indexing
    bidxs = [i-1 for i in bidxs]
    asrcs = [i-1 for i in asrcs]
    atgts = [i-1 for i in atgts]

    ### return ###
    block.nbonds        = nbonds
    block.bidxs         = bidxs
    block.asrcs         = asrcs
    block.atgts         = atgts
    block.btypes        = btypes
    block.bstatus_bitss = bstatus_bitss
    return block

### SUBSTRUCTURE ###

def MOL2_setup_SUBSTRUCTURE(block_substructure):
    """
    Setup MOL2 SUBSTRUCTURE block as mol attributes
    """
    block = Bunch()
    ### get ###
    nsubst = len(block_substructure)
    #assert nsubst == num_subst
    sidxs          = [None] * nsubst
    snames         = [None] * nsubst
    sroots         = [None] * nsubst
    stypes         = [None] * nsubst
    sdicts         = [None] * nsubst
    schains        = [None] * nsubst
    sctypes        = [None] * nsubst
    sninter_bondss = [None] * nsubst
    sstatuss       = [None] * nsubst
    scomments      = [None] * nsubst

    ### set ###
    for i,line in enumerate(block_substructure):
        split = line.split()
        stype, sdict, schain, sctype, sstatus, scomment = [None] * 6
        sninter_bonds = -1
        sidx, sname, sroot = split[:3]
        if len(split) > 3:
            stype         = split[3]
        if len(split) > 4:
            sdict         = split[4]
        if len(split) > 5:
            schain        = split[5]
        if len(split) > 6:
            sctype        = split[6]
        if len(split) > 7:
            sninter_bonds = split[7]
        if len(split) > 8:
            sstatus       = split[8]
        if len(split) > 9:
            scomment      = split[9:]
            scomment = ' '.join(scomment) # to recover the space splitted after 9th position
        sidxs[i], snames[i], sroots[i], stypes[i], sdicts[i], schains[i], sctypes[i], sninter_bondss[i], sstatuss[i], scomments[i] = int(sidx), sname, int(sroot), stype, sdict, schain, sctype, int(sninter_bonds), sstatus, scomment

    ### check ###
    # idxs
    assert range(1,nsubst+1) == sidxs
    # avail
    #avail_subst_type #todo
    #avail_subst_status #todo
    ### edit ###
    # back to python indexing
    sidxs = [i-1 for i in sidxs]
    sroots = [i-1 for i in sroots]

    ### return ###
    block.nsubst         = nsubst
    block.sidxs          = sidxs
    block.snames         = snames
    block.sroots         = sroots
    block.stypes         = stypes
    block.sdicts         = sdicts
    block.schains        = schains
    block.sctypes        = sctypes
    block.sninter_bondss = sninter_bondss
    block.sstatuss       = sstatuss
    block.scomments      = scomments
    return block

### CRYSIN ###

def MOL2_setup_CRYSIN(block_crysin):
    """
    Setup MOL2 CRYSIN block as mol attributes
    """
    block = Bunch()
    ### get ###
    assert len(block_crysin) < 2, "max lenght is 1 for block crysin"
    a, b, c, alpha, beta, gamma, spgn, spgx = [None] * 8
    if len(block_crysin) == 1:
        line = block_crysin[0]
        a, b, c, alpha, beta, gamma, spgn, spgx = line.split()
        a, b, c, alpha, beta, gamma, spgn, spgx = float(a), float(b), float(c), float(alpha), float(beta), float(gamma), int(spgn), int(spgx)

    ### return ###
    block.a     = a
    block.b     = b
    block.c     = c
    block.alpha = alpha
    block.beta  = beta
    block.gamma = gamma
    block.spgn  = spgn
    block.spgx  = spgx
    return block

def make_elems(anames):
    return [''.join([' ' if i in digits else i.lower() for i in aname]).split()[0] for aname in anames]

def make_ctab(itab, jtab):
    return zip(itab, jtab)

def make_cellparams(a, b, c, al, be, ga):
    return [a, b, c, al, be, ga]

### SETUP ###
def read(mol, f):
    """
    Routine, which reads a mol2 formatted file
    :Parameters:
        -f   (obj): mol2 file object
        -mol (obj): instance of a molclass
    """
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    ### blocks reading ###
    blocks = MOL2_read_blocks(f)
    _molecule     = MOL2_setup_MOLECULE(blocks["molecule"])
    _atom         = MOL2_setup_ATOM(blocks["atom"])
    _bond         = MOL2_setup_BOND(blocks["bond"])
    _substructure = MOL2_setup_SUBSTRUCTURE(blocks["substructure"])
    _crysin       = MOL2_setup_CRYSIN(blocks["crysin"])
    ### assertions ###
    assert _molecule.num_atoms == _atom.natoms
    assert _molecule.num_bonds == _bond.nbonds
    assert _molecule.num_subst == _substructure.nsubst
    ### making ###
    cellparams = make_cellparams(_crysin.a, _crysin.b, _crysin.c, _crysin.alpha, _crysin.beta, _crysin.gamma)
    elems = make_elems(_atom.anames)
    ctab = make_ctab(_bond.asrcs, _bond.atgts)
    ### setting ###
    mol.set_natoms(_atom.natoms)
    mol.set_xyz(_atom.coords)
    mol.set_elems(elems)
    mol.set_atypes(_atom.atypes)
    mol.set_fragtypes(_atom.resnames)
    mol.set_fragnumbers(_atom.resids)
    mol.set_ctab(ctab, conn_flag=True)
    if any(cellparams):
        mol.set_cellparams(cellparams)
        ### TODO: investigate spgn > 231
        if 1 < _crysin.spgn < 231: # not P1 -> apply spg operators
            mol.addon("spg")
            mol.spg.make_P1(spgnum=_crysin.spgn, sg_setting=_crysin.spgx)
    ### keeping ###
    MOL2_blocks = Bunch()
    MOL2_blocks.molecule     = _molecule
    MOL2_blocks.atom         = _atom
    MOL2_blocks.bond         = _bond
    MOL2_blocks.substructure = _substructure
    MOL2_blocks.crysin       = _crysin
    mol.MOL2_blocks = MOL2_blocks
    return

def write(mol, f):
    """
    Write mol sytem in mol2 format
    ### NOT YET IMPLEMENTED
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

    print('NOT YET IMPLEMENTED!')
    return

