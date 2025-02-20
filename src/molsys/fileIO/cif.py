import numpy
import string
from . import txyz
from collections import Counter
import re
import logging

logger = logging.getLogger("molsys.io")

def write(mol,f, name='', write_bonds=True):
    """
    Routine, which writes a cif file in P1
    :Parameters:
        -mol (obj) : instance of a molclass
        -f (obj) : file object or writable object
    """
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    f.write("data_%s\n" % name)
    f.write("_symmetry_cell_setting           triclinic \n")
    f.write("_symmetry_space_group_name_H-M   'P 1' \n")
    f.write("_symmetry_Int_Tables_number      1 \n")
    f.write("loop_ \n")
    f.write("_symmetry_equiv_pos_site_id \n")
    f.write("_symmetry_equiv_pos_as_xyz \n")
    f.write("1 x,y,z \n")
    f.write("_cell_length_a          %12.6f        \n" % (mol.cellparams[0]))
    f.write("_cell_length_b          %12.6f        \n" % (mol.cellparams[1]))
    f.write("_cell_length_c          %12.6f        \n" % (mol.cellparams[2]))

    f.write("_cell_angle_alpha       %12.6f        \n" % (mol.cellparams[3]))
    f.write("_cell_angle_beta        %12.6f        \n" % (mol.cellparams[4]))
    f.write("_cell_angle_gamma       %12.6f        \n" % (mol.cellparams[5]))

    f.write("loop_  \n")
    f.write("_atom_site_label   \n")
    f.write("_atom_site_type_symbol  \n")
    f.write("_atom_site_fract_x  \n")
    f.write("_atom_site_fract_y  \n")
    f.write("_atom_site_fract_z \n")
    mol.wrap_in_box()
    frac_xyz = mol.get_frac_xyz()
    for i in range(mol.natoms):
        f.write(" %s%d  %s %12.6f  %12.6f  %12.6f \n" % (mol.elems[i].title(),i,mol.elems[i].title(),\
            frac_xyz[i,0],frac_xyz[i,1],frac_xyz[i,2],))
    if write_bonds:
        f.write("loop_  \n")
        f.write("_geom_bond_atom_site_label_1  \n")
        f.write("_geom_bond_atom_site_label_2  \n")
        mol.set_ctab_from_conn()
        for i,ctab in enumerate(mol.ctab):
            c1,c2=ctab[0],ctab[1]
            e1,e2 =mol.elems[c1].title(), mol.elems[c2].title()
            f.write('%s%d   %s%d \n' % (e1,c1,e2,c2) )
    f.write("  \n")
    f.write("#END  \n")
    return

def read(mol, f, make_P1=True, detect_conn=True, conn_thresh=0.1, disorder=None):
    """read CIF file
    :Arguments:
    - make_P1(bool): if True: make P1 unit cell from primitive cell
    - detect_conn(bool): if True: detect connectivity
    - disorder(dict or None): choose disorder group per each disorder assembly
        to consider. Use a dictionary of disorder assembly keys to disorder
        group items, e.g. {"A":"2", "B":"3", "C":2, ...}
        if None: first disordered group in lexical sort is taken per each
            disorder assembly (e.g. {"A":"1", "B":"1", ...}"""
    """BUG: cif instance cannot be deepcopied! (workaround w/ __mildcopy__?"""
    try: 
        import CifFile
    except ImportError:
        raise ImportError('pycifrw not installed, install via pip!')
    cf = CifFile.ReadCif(f)
    if len(cf.keys()) != 1:
        for key in cf.keys(): print(key)
        raise IOError('Cif File has multiple entries ?!')
    cf = cf[cf.keys()[0]]
    
    try:
        occ = [format_float(i) for i in cf["_atom_site_occupancy"]]
    except KeyError as e:
        if "No such item: _atom_site_occupancy" in str(e):
            disorder = None
        else:
            raise(e)
    else:
        if any( [i!=1 for i in occ] ):
            try:
                logger.info("fractional occupancies in cif file")
                mol.occupancy = occ
                disorder_assembly_full = [i for i in cf["_atom_site_disorder_assembly"]]
                disorder_group_full = [i for i in cf["_atom_site_disorder_group"]]
            except KeyError as e:
                if "No such item: _atom_site_disorder_assembly" in str(e) or\
                        "No such item: _atom_site_disorder_group" in str(e):
                    disorder = None
                    logger.warning("fractional occupancies: disorder not readable!")
                else:
                    raise(e)
            else:
                mol.occupancy_assembly = disorder_assembly_full
                mol.occupancy_group = disorder_group_full
                select_disorder = [i for i,e in enumerate(disorder_group_full) if e != '.']
                # remove fully occupied positions (data could be polluted)
                disorder_group = [disorder_group_full[i] for i in select_disorder]
                disorder_assembly = [disorder_assembly_full[i] for i in select_disorder]
                # create disorder list for each disorder assembly
                if disorder is None: # first sorted as disordered
                    disorder = {}
                    disorder_couple = set(zip(disorder_assembly, disorder_group))
                    disorder_dict = {}
                    for a,g in disorder_couple:
                        try:
                            disorder_dict[a].append(g)
                        except KeyError:
                            disorder_dict[a] = [g]
                    for a in disorder_dict:
                        disorder_dict[a].sort()
                        disorder[a] = disorder_dict[a][0] #take first (default)
                select = [i for i,e in enumerate(disorder_assembly_full) if i in select_disorder if disorder_group_full[i] == disorder[e]]
                select += [i for i,e in enumerate(disorder_group_full) if i not in select_disorder]
                if len(select) != sum(occ):
                    logger.warning(
                        "number of selected atoms not equal to total occupancy: %s != %s" % \
                        (len(select), sum(occ))
                    )
        else:
            disorder = None
    ### elems ##################################################################
    #elems = [str(i) for i in cf['_atom_site_type_symbol']
    try:
        elems = [str(i) for i in cf['_atom_site_type_symbol']]
    except KeyError as e:
        if "No such item: _atom_site_type_symbol" in str(e):
            logger.warning("atom labels as elements")
            labels = [str(i) for i in cf['_atom_site_label']]
            elems = [''.join(c for c in i if not c.isdigit()) for i in labels]
        else:
            raise(e)
    elems = [str(i).lower() for i in elems]
    ### atypes #################################################################
    atypes = [str(i) for i in cf['_atom_site_label']]
    ### xyz ####################################################################
    x = [format_float(i) for i in cf['_atom_site_fract_x']]
    y = [format_float(i) for i in cf['_atom_site_fract_y']]
    z = [format_float(i) for i in cf['_atom_site_fract_z']]

    if disorder is not None:
        if disorder:
            # select according to given disorder
            elems = [e for i,e in enumerate(elems) if i in select]
            atypes = [e for i,e in enumerate(atypes) if i in select]
            x = [e for i,e in enumerate(x) if i in select]
            y = [e for i,e in enumerate(y) if i in select]
            z = [e for i,e in enumerate(z) if i in select]
        else:
            logger.warning("auto disorder detection failed!")

    ### cellparams #############################################################
    a = format_float(cf['_cell_length_a'])
    b = format_float(cf['_cell_length_b'])
    c = format_float(cf['_cell_length_c'])
    alpha = format_float(cf['_cell_angle_alpha'])
    beta =  format_float(cf['_cell_angle_beta' ])
    gamma = format_float(cf['_cell_angle_gamma'])

    ### set ####################################################################
    mol.set_natoms(len(elems))
    mol.set_cellparams([a, b, c, alpha, beta, gamma])
    mol.set_xyz_from_frac(numpy.array([x,y,z]).T)
    #mol.wrap_in_box()
    mol.set_elems(elems)
    mol.set_atypes(atypes)
    mol.set_nofrags()
    mol.set_empty_conn()
    mol.cifdata = cf
    ### span and connectivity ##################################################
    ### TBI: connectivity read from the cif file directly
    ### currently: detected by scratch 
    if make_P1: 
        mol.addon('spg')
        mol.proper_cif = mol.spg.make_P1(conn_thresh=conn_thresh, onduplicates="keep")
    if detect_conn:
        mol.detect_conn()
    return

def format_float(data):
    if data.count('(') != 0:
        data = data.split('(')[0]
    return float(data)

