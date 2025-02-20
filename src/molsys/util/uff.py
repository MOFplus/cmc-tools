"""

            molsys uff implementation

            written by Gunnar Schmitz (RUB, CMC-group) within Oxyflame CRC-TRR 129

            This implementation is built on several codes implementing UFF (which partly differ)
            and using all available info on corrections to the UFF defined in the original paper.
            Especially the lammps_interface by Peter Boyd has been used.

            Additions:
            Vanessa 2023: added feature to keep connectivity in a mol file (if a mfpx file is read)

            RS 2023: 
            This is a slight revision of Gunnar's original code with keep bonds as the default
            => if a xyz file was read and bonds should be detected you need to require this
            (TBI: make this bond detection with overcoord a default in molsys outside of uff)
            => remove pylmps dependencies .. all data is written into the ff addon data structures
            this means wrting or packing can be done afterwards and writing lammps input will be 
            dealt with in pylmps (as if MOF-FF is used)

            REMARK:
            In the orignal paper by Rappe it is stated that UFF has been parameterized using geometric 
            combination rules for both distance and well depth. This is in contrast to the usual 
            Lorentz-Berthelot combirules, where the distance is combined arithmetic.
            From the code we believe that lammps_interface is using arithmetic, whereas openbabel uses
            geoemtric combionations for the vdW distance. We will follow the original suggestion to use geometric.

            The orignal paper states that 1-2 and 1-3 interaction are excluded for vdw and coulomb (this is done in openbabel).
            We will follwo this convention, even though using charges with UFF is disputed (to be tested: would 1-4 exclusion for charges improve things?) 

"""



# general import(s)
import sys
import numpy as np
import math

# special import(s)
import graph_tool as gt
import math

# molsys import(s)
from molsys import mpiobject
from .timer import Timer, timer
from .uff_param import atrad      # covalent atom radii
from .uff_param import maxcoord   # coordination number of atoms
from .uff_param import UFF_DATA,UFF4MOF_DATA # parameter set for UFF and UFF4MOF 
from .uff_param import valence_electrons     # number of valence electrons
from .uff_param import pse_group,pse_period  # info on groups and periods in PSE
from .uff_param import torsional_barriers,torsion # torsional barriers partially redundant info 
from .uff_param import metals,organic,transition_metals # for classifying atoms 


x2sigma = 2**(-1.0/6.0) # convert from LJ well depth (xi) to sigma as required by lammps input 

#####################################################################################################
#  Helper functions 
#####################################################################################################

def getgroup(elems):
    group = []
    # Could be done more elegantly....
    for elem in elems:
        e = elem.capitalize()
        if e in pse_group:
            group.append(pse_group[e])
        else:
            # Lantahnoide, Actinoide
            group.append(3)
            print("Group could not be assigned")
            print(elem)
    return group


def getbo(elems, nxtneigh, bnd, it, ufftypes):
    # Note: I don't like this function due to the endless ifs
    #       I hope there is a better way. One soltion might be
    #       to do the atom typing completely iteratively and then
    #       select the bond order based on the uff type. 
    iat = bnd[0]
    jat = bnd[1]
    bo = 1.0
    elemi = elems[iat].capitalize()
    elemj = elems[jat].capitalize()
    ufftypei = ufftypes[iat]
    ufftypej = ufftypes[jat]
    neighi = nxtneigh[iat]
    neighj = nxtneigh[jat]
    atoms = [elemi,elemj]
    is_assigned = False
    if "H" in atoms:
        bo = 1.0
        if "B" in atoms:
            bo = 2.0
        is_assigned = True
    if elemi == "C":
        # C-C connectivity
        if len(atoms) > len(set(atoms)):
            if neighi >= 4:
                bo = 1.0
                is_assigned = True
            elif neighi == 3:
                bo = 2.0
                if it > 1 and (ufftypei[2] == "R" or ufftypej[2] == "R"):
                   bo = 1.5
                is_assigned = True
            elif neighi == 2:
                if neighj == 2:
                    bo = 3.0
                else:
                    bo = 1.0
                is_assigned = True
            elif neighi == 1:
                bo = 1.0
                is_assigned = True 
            else:
                bo = 1.0
                is_assigned = True
        else:
            if "O" in atoms:
                if neighi == 3 and neighj == 1:
                    bo = 2.0
                else:
                    bo = 1.0
                is_assigned = True
    if elemi == "N": # TODO
        if neighi >= 3:
           bo = 1.0
           is_assigned = True
        elif neighi == 2:
            bo = 2.0
            is_assigned = True
        elif neighi == 1:
            bo = 3.0
            is_assigned = True
    if elemi in ("Zn", "Cu") :
        if len(atoms) > len(set(atoms)): # Zn-Zn bond    
            bo = 0.25
            is_assigned = True
        elif "O" in atoms:
            bo = 0.5
            is_assigned = True
        else:
            bo = 1.0
            is_assigned = True
    if elemi == "O":
        if neighi >= 2:
            bo = 1.0
            is_assigned = True
        elif neighi == 1:
            bo = 2.0
            is_assigned = True
    if not is_assigned:
        # Use default assignment:

        # C,N,O-group
        if pse_group[elemi] in [14,15,16] and len(ufftypei) == 3:
           sid = ufftypei[2]
           if sid == '3':
              bo = 1.0
              is_assigned = True
           elif sid == "R":
              if ufftypej[2] == "R":
                 bo = 1.5
              else:
                 bo = 1.0
              is_assigned = True
           elif sid == '2':
              if ufftypej[2] == '2':
                 bo = 2.0
              else:
                 bo = 1.0 
              is_assigned = True
           elif sid == '1':
              if ufftypej[2] == '1':
                 bo = 3.0
              else:
                 bo = 1.0
              is_assigned = True
        else:
           bo = 1.0
           #is_assigned = True

    if not is_assigned and it > 1:
        print(atoms)
        print(neighi)
        raise ValueError("bond order not assigned")

    return bo

def get_rij(uffdata, uff_typei, uff_typej, bo):
    #
    # r_ij = r_i + r_j + r_BO - r_EN
    #
    ri = uffdata[uff_typei][0]
    rj = uffdata[uff_typej][0]
    xi = uffdata[uff_typei][8]
    xj = uffdata[uff_typej][8]
    log_bo = math.log(bo)  
    lambd = 0.1332
    r_BO = -lambd * (ri + rj) * log_bo
    r_EN = ri * rj * (math.sqrt(xi)-math.sqrt(xj))**2 / (xi*ri + xj*rj)
    rij = ri + rj + r_BO - r_EN
    return rij

def get_U(elem):
    period = pse_period[elem]
    if period == 2:   # Li through Ne
        U = 2.0
    elif period == 3: # Na through Ar
        U = 1.25
    elif period == 4: # K through Kr
        U = 0.7
    elif period == 5: # Rb through Xe
        U = 0.2
    elif period == 6: # Cs through Rn
        U = 0.1
    else:
        print("Did not find U param")
        sys.exit()
    return U

#####################################################################################################
# class definition 
#
#  RS changes .. keep bondorder and sp (hybridization) as atributes in order to compare with openbabel


#####################################################################################################

class UFFAssign(mpiobject):

    def __init__(self, mol, use_obabel=False, detect_conn = False, mpi_comm = None, out = None, uff="UFF", verbose=False):
        super(UFFAssign,self).__init__(mpi_comm, out)
        self.mol = mol
        self.use_obabel = use_obabel
        self.detect_conn = detect_conn # flag to detect connectivity .. by default we assume it exists
        self.debug_ric = False
        self.timer = Timer("UFF-assign")
        self.uff_types = None
        self.unique_atom_types = {}
        self.uff = uff
        self.verbose = verbose
        if uff == "UFF":
           print("Setup for UFF")
           self.uff_data = UFF_DATA
        elif uff == "UFF4MOF":
           print("Setup for UFF4MOF")
           self.uff_data = UFF4MOF_DATA
        else:
           raise ValueError("Unknown UFF parameterization")
        self.assign() 
        return

    def assign(self, deps=0.4):
        """
        Main driver for auto assignment of atoms types
        containing the required potential terms
        => RS we fill only the ff addon data structures .. either we need to write this to ric/par or we convert to lammps in pylmps
 
        deps - default 0.4 (doi 10.1002/jcc.24309) needed to identify bonds
        """
        self.timer.start()
        # prepare mol objct
        if not hasattr(self.mol, 'graph'):
            self.mol.addon("graph")
        self.mol.graph.make_decomp_graph()
        if not hasattr(self.mol, 'ff'):     # TBI check if a force field is already assigned (check self.mol.ff.par.FF does it exist?)
            self.mol.addon("ff")
        #
        # 1. get connectivity 
        #
        natoms = self.mol.natoms
        if self.detect_conn:
            nxtneigh  = np.zeros(natoms,dtype=np.int8)
            conn = [[] for i in range(natoms)]
            bonds = [] # we hold info on bonds as (i,j,val) where i and j are the atom indices and val the bond length 
            # loop over all atoms in mol
            for iat,elemi in enumerate(self.mol.elems): 
                # Setup list of bonds for iat
                bonds_local = []    
                ei = elemi.capitalize() 
                # get neighbours for current atom
                nxtneigh[iat] = 0
                partner = []
                for jat,elemj in enumerate(self.mol.elems): 
                    if iat != jat:
                        dist2 = self.mol.get_distvec(iat,jat)[0]**2
                        sumradii  =  (atrad[elemi.capitalize()] + atrad[elemj.capitalize()]) + deps 
                        sumradii2 = sumradii*sumradii  
                        # it is a bond if it is smaller than the sum of the covalent radii plus a factor
                        if dist2 <= sumradii2: 
                            nxtneigh[iat] += 1
                            conn[iat].append(jat)
                            # create bond
                            if jat > iat:
                                dbnd = math.sqrt(dist2)
                                mybnd = (iat,jat,dbnd)
                                bonds_local.append(mybnd)
                                ej = elemj.capitalize()
                                partner.append(ej)
                # sort the atom's bonds according to determine the clossest to the equilibrium distance
                sorted_bonds = sorted(bonds_local, key=lambda tup: tup[2])      
                # 
                # remove overcoordination
                #
                if ei == "H" and len(sorted_bonds) == 2 and "B" in partner:
                    # special case for H_b hydrogen bonded to boron)
                    nbnd = 2
                else: 
                    nbnd = min(maxcoord[ei],len(sorted_bonds)) 
                bondsi = sorted_bonds[:nbnd]
                # Add to pool of bonds
                bonds = bonds + bondsi
                #
                # Recreate neighbour list based on bond info
                #
                nxtneigh.fill(0)
                conn = [[] for i in range(natoms)]
                for bnd in bonds:
                    iat = bnd[0]
                    jat = bnd[1]
                    conn[iat].append(jat)
                    conn[jat].append(iat)
                    nxtneigh[iat] += 1
                    nxtneigh[jat] += 1
            # set connectivity
            self.mol.set_conn(conn)
        else: # default detect_conn is False: we use the data in conn but still need to compute extra data
            nxtneigh  = np.zeros(natoms,dtype=np.int8)
            conn = self.mol.conn
            bonds = [] # we hold info on bonds as (i,j,val) where i and j are the atom indices and val the bond length 
            # loop over all atoms in mol
            for iat,elemi in enumerate(self.mol.elems): 
                # Setup list of bonds for iat
                bonds_local = []    
                ei = elemi.capitalize() 
                # get neighbours for current atom
                nxtneigh[iat] = 0
                partner = []
                for jat in self.mol.conn[iat]:
                    elemj = self.mol.elems[jat]
                    dist2 = self.mol.get_distvec(iat,jat)[0]**2
                    sumradii  =  (atrad[elemi.capitalize()] + atrad[elemj.capitalize()]) + deps 
                    sumradii2 = sumradii*sumradii  
                    # it is a bond if it is smaller than the sum of the covalent radii plus a factor
                    nxtneigh[iat] += 1
                    #conn[iat].append(jat)
                    # create bond
                    if jat > iat:
                        dbnd = math.sqrt(dist2)
                        mybnd = (iat,jat,dbnd)
                        bonds_local.append(mybnd)
                        ej = elemj.capitalize()
                        partner.append(ej) 
                # sort the atom's bonds according to determine the clossest to the equilibrium distance
                sorted_bonds = sorted(bonds_local, key=lambda tup: tup[2])      
                # 
                # check for overcoordination (if we use predefined bonds this should not happen)
                #
                if ei == "H" and len(sorted_bonds) == 2 and "B" in partner:
                    # special case for H_b hydrogen bonded to boron)
                    nbnd = 2
                else: 
                    nbnd = min(maxcoord[ei],len(sorted_bonds))
                assert nbnd == len(sorted_bonds) 
                # Add to pool of bonds
                bonds = bonds + sorted_bonds
                #
                # Recreate neighbour list based on bond info
                #
                nxtneigh.fill(0)
                conn = [[] for i in range(natoms)]
                for bnd in bonds:
                    iat = bnd[0]
                    jat = bnd[1]
                    conn[iat].append(jat)
                    conn[jat].append(iat)
                    nxtneigh[iat] += 1
                    nxtneigh[jat] += 1
            # reset connectivity in mol obejct 
            self.mol.set_conn(conn)    # RS: even though we have a connectivity (read in) it needs to be set becasue the order 
                                       # could have changed .. this is ugly! maybe we can fix that but 
                                       # since find_rics runs on conn and this is later used to process things, order is important!
        # find all redundand internal coordinates
        self.mol.ff.ric.find_rics(for_uff=True)
        
        # initalize FF data (empty)
        self.mol.ff._init_data()
        self.mol.ff._init_pardata(self.uff)
        
        #
        # 2. UFF atom type assignment 
        #

        rings = []
        if self.use_obabel == True:
            # a) Get bond orders
            bondorder,bndlst,aromatic,rings = self.get_bond_order_obabel()
            # b) Calculate Valence of atoms
            valence = self.get_valence(conn,bndlst,bondorder)
            # c) Calculate Hybridization
            sp = self.get_hybridization(valence,nxtneigh)
            # d) Atom type assignment
            atypes = self.assign_atom_types(nxtneigh,sp,conn,aromatic=aromatic,rings=rings)
        else:   
            # 0) get initial typing
            types = self.get_init_types(nxtneigh)

            # Currently only one iteration to refine typing
            last_iter = 2
            for it in [1,last_iter]:
                # a) Get bond orders
                bondorder,bndlst = self.get_bond_order(nxtneigh,it,types)
                # b) Calculate Valence of atoms
                valence = self.get_valence(conn,bndlst,bondorder)
                # c) Calculate Hybridization
                sp = self.get_hybridization(valence,nxtneigh)
                # d) Atom type assignment
                atypes = self.assign_atom_types(nxtneigh,sp,conn)
                # for next iteration
                types = atypes
            # Correct uff types for special cases
            atypes,is_changed = self.correct_atom_types(atypes,conn, sp)     # RS HACK pass sp as well to change it if needed (sideeffect!!)     
            if is_changed and self.verbose:
                print("\n Atom types changed to:")
                print(" ------------------------------------")
                for iat,(elemi,ufftype) in enumerate(zip(self.mol.elems, atypes)):
                    print("  atom %i element %s => %s" % (iat,elemi,ufftype))

        self.atypes = atypes
        self.sp = sp
        self.bondorder = bondorder

        #
        # 3. Do parameter assigment based on atom types
        #

        self.assign_vdw_param(atypes)
        self.assign_bnd_param(atypes,bonds,bondorder,rings)
        self.assign_ang_param(atypes,bondorder,bndlst,rings)
        self.assign_dihed_param(atypes,sp,bndlst,bondorder,rings)
        self.assign_oop_param(atypes,rings)
        
        #RS
        # 4. Set proper defaults for UFF
        self.mol.ff.settings["vdwtype"] = "lj"
        #self.mol.ff.settings["vdwtype"] = "lj_mdf"
        self.mol.ff.settings["chargetype"] = "point"
        self.mol.ff.settings["vdw12"] = 0.0
        self.mol.ff.settings["vdw13"] = 0.0
        self.mol.ff.settings["vdw14"] = 1.0
        self.mol.ff.settings["coul12"] = 0.0
        self.mol.ff.settings["coul13"] = 0.0
        self.mol.ff.settings["coul14"] = 1.0
        self.mol.ff.settings["radrule"] = "geometric"
        self.mol.ff.settings["epsrule"] = "geometric"
        self.mol.ff.settings["radfact"] = 0.5  # in UFF the sigma is twice the "radius"
        self.mol.ff.settings["chargegen"] = "zero"

        self.timer.report()
        return


    @timer('get init types')
    def get_init_types(self, nxtneigh):
        """
        Get initial uff typing. Can/should later be refined 
        Only the information on next neighbours is used for assigning the types

        :Return:
        - ufftypes : list of assigned UFF atom types
        """
        ufftypes = []
        for iat,(elemi,nxtnei) in enumerate(zip(self.mol.elems,nxtneigh)):
            e = elemi.capitalize()
            ufftype = e 
            if len(ufftype) == 1:
                ufftype += "_"
            if nxtnei == 2:
               if e == "H":
                  ufftype = "H_b"
               elif pse_group[e] == 14:
                  ufftype += "1"
            elif nxtnei == 3:
               if pse_group[e] == 15:
                  ufftype += "3"
               else: 
                  ufftype += "2"
            elif nxtnei == 4:
               if pse_group[e] == 14:
                  ufftype += "3"
               else:
                  ufftype += "4"
            elif nxtnei == 6:
                  ufftype += "6"
            ufftypes.append(ufftype)
        return ufftypes

    @timer('get bond order')
    def get_bond_order(self, nxtneigh, it, ufftypes):
        """
        Get bond order 
        Returns the pre-computed bond orders as well as a bond list

        :Return:
        - bondorder : list of assigned bond orders
        - bndlst : list of bonds
        - it : iteration in refinement
        - uffttpes : current UFF type assignment
        """
        num_all_bonds = len(self.mol.ff.ric.bnd)
        bondorder = np.zeros(num_all_bonds,dtype=float)
        bndlst = np.zeros((self.mol.natoms,self.mol.natoms),dtype=int)
        for ij,bnd in enumerate(self.mol.ff.ric.bnd):
            bo = getbo(self.mol.elems,nxtneigh,bnd,it,ufftypes)
            bondorder[ij] = bo
            iat = min(bnd[0],bnd[1])
            jat = max(bnd[0],bnd[1])
            bndlst[iat][jat] = ij
            bndlst[jat][iat] = ij
        return bondorder,bndlst

    @timer('get bond order_obabel')
    def get_bond_order_obabel(self):
        num_all_bonds = len(self.mol.ff.ric.bnd)
        bondorder = np.zeros(num_all_bonds,dtype=float)
        bndlst = np.zeros((self.mol.natoms,self.mol.natoms),dtype=int)
        n = self.mol.clone()
        n.set_xyz(n.apply_pbc(n.get_xyz()))
        n.make_nonperiodic()
        n.addon("obabel")
        bnd_obabel, bo_obabel = n.obabel.get_bond_order()
        aromatic = n.obabel.get_aromatic()
        arom_rings = n.obabel.get_aromatic_ringsystems()
        for ij,bnd in enumerate(self.mol.ff.ric.bnd):
            iat = min(bnd[0],bnd[1])
            jat = max(bnd[0],bnd[1])
            id_j = bnd_obabel[iat].index(jat)
            if all(at in aromatic for at in [iat, jat]) and any(all(at in ring for at in [iat, jat]) for ring in arom_rings):
                bo = 1.5
            else:
                bo = bo_obabel[iat][id_j]
            bondorder[ij] = bo
            bndlst[iat][jat] = ij
            bndlst[jat][iat] = ij
        rings = [list(r.atoms) for r in n.obabel.rings]
        return bondorder,bndlst,aromatic,rings
    
    @timer('get valence')
    def get_valence(self, conn, bndlst, bondorder):
        """
        Calculates the valences for the different atoms

        :Return:
        - valence : list of valences for the different atoms
        """
        valence = np.zeros(self.mol.natoms,dtype=float)
        for iat,conni in enumerate(conn):
            # get neighbours
            for jat in conni:
                i = min(iat,jat)
                j = max(iat,jat)
                ijbnd = bndlst[i][j]
                valence[iat] += bondorder[ijbnd]
        if self.verbose:
            for iat,(elemi,val) in enumerate(zip(self.mol.elems, valence)):
                print("  atom %i element %s => valence = %i" % (iat,elemi,val)) 
        return valence   
 
    @timer('get hybridization')
    def get_hybridization(self, valence, nxtneigh):
        """
        Determines the hybridization for the atom list 

        :Return:
        - sp : list of the hybridizations
        """
        sp = np.zeros(self.mol.natoms,dtype=int)
        for iat, (neighi,elem) in enumerate(zip(nxtneigh,self.mol.elems)):
            e = elem.capitalize()
            sp[iat] = int(math.ceil(neighi + (valence_electrons[e] - valence[iat])/2.0)-1.0)
            # Catch unrealistic cases:
            if sp[iat] > 3:
              if e in metals:
                 sp[iat] = 3
        if self.verbose:
            print(" Following hybridization identified:")
            print(" ------------------------------------")
            for iat,(elemi,spn) in enumerate(zip(self.mol.elems, sp)):
                print("  atom %i element %s => sp%i" % (iat,elemi,spn)) 
        return sp

    @timer('atom type assign')
    def assign_atom_types(self, nxtneigh, sp, conn, aromatic=[], rings=[]):
        """
        Determines the different UFF atomtypes based on the neighbour list and the hybridization

        :Return:
        - atypes : list of UFF atom types
        """
        atypes = []
        groups = getgroup(self.mol.elems)
        for iat,elemi in enumerate(self.mol.elems):
            is_assigned = False
            nxtnei = nxtneigh[iat]
            ufftype = elemi.capitalize()
            if len(ufftype) == 1:
                ufftype += "_"
            # special treatment for hydrogen
            if nxtnei == 2:
                if ufftype == 'H_':
                    partner = []
                    for jat in conn[iat]:
                        ej = self.mol.elems[jat].capitalize()
                        partner.append(ej)
                    if "B" in partner:
                       ufftype += 'b' 
                    atypes.append(ufftype)
                    continue
            if ufftype == 'H_':
                atypes.append(ufftype)
                continue
  
            # Quick hack
            if ufftype == "Zn":
                #ufftype = "Zn2f2"  # "Zn3+2  "Zn4+2"  "Zn3f2" "Zn1f1" "Zn2f2"
                ufftype = "Zn4+2"  # "Zn3+2  "Zn4+2"  "Zn3f2" "Zn1f1" "Zn2f2"
                atypes.append(ufftype)
                continue 
            if ufftype == "Cu":
                #ufftype = "Zn2f2"  # "Zn3+2  "Zn4+2"  "Zn3f2" "Zn1f1" "Zn2f2"
                ufftype = "Cu4+2"  # "Zn3+2  "Zn4+2"  "Zn3f2" "Zn1f1" "Zn2f2"
                atypes.append(ufftype)
                continue 
            
            ei = elemi.capitalize()
            if self.use_obabel == False:
                # resonance detection
                # check if it contains C or N, is sp2 and part of a ring
                ring_size = 0
                if (ei in ["C","N"] and sp[iat] <= 2) or (ei in ["N","O"] and sp[iat] == 3):
                    for jat in conn[iat]:
                        moldg = gt.Graph(self.mol.graph.moldg)    # Deep copy
                        vertices = moldg.get_vertices()
                        vi = vertices[iat]
                        vj = vertices[jat]
                        eij = moldg.edge(iat,jat)
                        if eij:
                            # filter edge between vertices vi and vj 
                            moldg.clear_filters()
                            filt = moldg.new_edge_property("bool")
                            filt.set_value(True)
                            filt[eij]=False
                            moldg.set_edge_filter(filt)
                            vertex_list = gt.topology.shortest_path(moldg,vi,vj)
                            moldg.clear_filters()
                            if vertex_list[0] and vertex_list[1]:
                                ring_size = len(vertex_list[0])
                        moldg.clear_filters()
                if ring_size in [5,6,7,8]:
                    if elemi.capitalize() in ["O","N"]:
                       ufftype += "2"
                       sp[iat] = 2
                    else:
                       ufftype += "R"
                    atypes.append(ufftype)
                    continue
            else:
                if iat in aromatic:
                    #if ei in ["O","N"]:
                    #   ufftype += "2"
                    #   sp[iat] = 2
                    #else:
                    #   ufftype += "R"
                    ufftype += "R"
                    atypes.append(ufftype)
                    continue 
                if ei in ["O","N"]:
                    ring_3 = False
                    for ring_i in rings:
                        if iat in ring_i and len(ring_i) == 3:
                            ring_3 = True
                            break
                    if ring_3 == True:
                        ufftype += "R"
                        atypes.append(ufftype)
                        continue

            # Quick hack
            if ei in ["C","N","O"] and sp[iat] >= 4: 
                ufftype += "3"
                atypes.append(ufftype)
                continue

            # the rest
            ufftype += str(sp[iat]) 
        
            atypes.append(ufftype)
       
        if self.verbose: 
            print("\n Following atom types are assigned:")
            print(" ------------------------------------")
            for iat,(elemi,ufftype) in enumerate(zip(self.mol.elems, atypes)):
                print("  atom %i element %s => %s" % (iat,elemi,ufftype))
        self.uff_types = atypes
        return atypes 
    
   
    @timer('Correct atom types')
    def correct_atom_types(self, atypes, conn, sp): 
        """
        I don't like this routine. It re-assigns atom types as ad hoc repair for certain cases

        RS in order to make things consistent we have to change the hybridization is sp along with this correction.
            ==> it might be worth to test the mechanisms available in obabel .. depending on how fast that work and 
                how bulletproof it is 

        """
        is_changed = False
        new_atypes = atypes.copy()
        for iat,ufftype in enumerate(atypes):
            if ufftype == "O_3":
                is_assigned = False
                for jat in conn[iat]:
                    if atypes[jat] == "C_R":
                        newtype = "O_2"
                        new_atypes[iat] = newtype
                        is_changed = True
                        sp[iat] = 2 # change hybridization of this atom to sp2 (O_2)
                        break
        return new_atypes,is_changed    
    
    ########################################################################################## 
    # atom specific data (charge, non-bonded interactions)
    # Note:
    #      parind is the parameter idenitifer
    #      par    is the dictionary with parind keys containing the parameter
    ########################################################################################## 
    @timer('assign vdw param')
    def assign_vdw_param(self, atypes):
        """
        Assigns the van-der-waals parameters for the UFF potential terms 

        """

        self.mol.ff.parind["vdw"] = []  # re-init again
        self.mol.ff.parind["cha"] = []
        for uff_typei in atypes:
            xi = self.uff_data[uff_typei][2] # tabulted vdW distance (of well) in Angstr
            Di = self.uff_data[uff_typei][3] # tabulated well depth (aka epsilon)
            # Zi = self.uff_data[uff_typei][5]    
    
            sigi = xi * x2sigma

            parind_vdw = "lj->(" + uff_typei + ")|UFF"   # GS TODO Lammps definition different than UFF paper
            
            self.mol.ff.parind["vdw"].append([parind_vdw])
            self.mol.ff.par["vdw"][parind_vdw] = ("lj" ,[sigi, Di])
            
            parind_cha = "point->(" + uff_typei + ")|UFF"       
            self.mol.ff.parind["cha"].append([parind_cha])
            self.mol.ff.par["cha"][parind_cha] = ("point",[0.0]) # We put here zero charge
    
    ########################################################################################## 
    #
    # ric specific data 
    #
    ########################################################################################## 
    
    # bonds
    @timer('assign bnd param')
    def assign_bnd_param(self, atypes, bonds, bondorder, rings=[]):
        """
        Assigns the bond potential parameters for the UFF potential terms 

        """
        for ijbnd, bnd in enumerate(bonds):
            iat = bnd[0]
            jat = bnd[1]
            uff_typei = atypes[iat]
            uff_typej = atypes[jat]
            atom_types = sorted([uff_typei,uff_typej])
            parind_bnd = f"harm->({atom_types[0]},{atom_types[1]})_{ijbnd}|UFF"
            #
            # calculate r_ij, k_ij
            #
            Zi = self.uff_data[uff_typei][5] 
            Zj = self.uff_data[uff_typej][5] 
            bo = bondorder[ijbnd]   
            rij = get_rij(self.uff_data,uff_typei,uff_typej,bo) 
   
            kij = 664.12 * Zi*Zj / rij**3  

            ring_3 = False
            ring_4 = False
            for ring_i in rings:
                if iat in ring_i:
                    if len(ring_i) == 3 and jat in ring_i:
                        ring_3 = True
                    elif len(ring_i) == 4 and jat in ring_i:
                        ring_4 = True
                    break
            if ring_3 == True:
                #kij *= 1.0 + (bo + 1.0) * 0.33333333
                kij *= bo
            elif ring_4 == True:
                #kij *= bo
                kij *= 1.0 + (bo + 1.0) * 0.66666667
                #kij *= 2 * bo

            # conversion needed since molsys harm is assumed to be in mdyne/A 
            mdyn2kcal = 143.88 
            kij = kij / mdyn2kcal

            if self.mol.ff.parind["bnd"][ijbnd] == None: 
                self.mol.ff.parind["bnd"][ijbnd] = [] 
            self.mol.ff.parind["bnd"][ijbnd].append(parind_bnd)
            self.mol.ff.par["bnd"][parind_bnd] = ("harm",[kij,rij])
        
        if self.debug_ric == True:
            print(self.mol.ff.parind["bnd"])
            print(self.mol.ff.par["bnd"])
    
    # angles
    @timer('assign ang param')
    def assign_ang_param(self, atypes, bondorder, bndlst, rings=[]):
        """
        Assigns the angle parameters for the UFF potential terms 

        """
        for iang,ang in enumerate(self.mol.ff.ric.ang):
            # we assume following connectivity
            #  
            #    j
            #  /  \
            # i    k
            #
            iat = ang[0] #ang[0] 
            jat = ang[1] #ang[1] 
            kat = ang[2] #ang[2]
            uff_typei = atypes[iat]
            uff_typej = atypes[jat]
            uff_typek = atypes[kat]
            atom_types = [uff_typei,uff_typej,uff_typek]
            if uff_typei > uff_typek:
                atom_types.reverse()
            Zi = self.uff_data[uff_typei][5] 
            Zk = self.uff_data[uff_typek][5] 
            theta0 =  math.radians(self.uff_data[uff_typej][1]) # TODO Correct value?
            boij = bondorder[bndlst[iat][jat]]
            boik = bondorder[bndlst[iat][kat]]
            bojk = bondorder[bndlst[jat][kat]]
            rij = get_rij(self.uff_data,uff_typei,uff_typej,boij)
            rjk = get_rij(self.uff_data,uff_typej,uff_typek,bojk)
            rik = math.sqrt(rij * rij + rjk * rjk - 2.0 * rij * rjk * math.cos(theta0))
            beta = 664.12 / (rij*rjk)
            kijk = beta * (Zi * Zk / rik**5) * rij*rjk * ( 3.0 * rij * rjk * (1.0 - math.cos(theta0)**2) - rik**2*math.cos(theta0))
            
            #get uff center definition
            if uff_typej == 'H_':  #hack vanessa
                coord = 1
            else:
                center = uff_typej[2]
                if center == "R":
                    coord = 2
                elif center == "b":
                    coord = 1
                else:
                    coord = int(center)
      
     
            # LAMMPS types:
            # -------------
            #
            #  cosine/periodic : 2.0/n**2 * C [1-B(-1)**n cos(n theta)] C (energy), B = 1 or -1, n = 1,2,3,4,...
            #
            #  fourier         : K * [C0 + C1*cos(theta) + C2*cos(2*theta)] K (energy), C0,C1,C2 real
            #
    
            is_cosine_periodic = False
            if coord == 1:
                # sp - linear case (sign error in rappe paper)
                is_cosine_periodic = True
                # -> k * (1.0 + cos(theta))
                C = kijk * 0.5
                B = 1
                n =  1
            else:
                ring_3_center = False
                ring_3_all = False
                ring_4_center = False
                ring_4_all = False
                ring_else = False
                for ring_i in rings:
                    if jat in ring_i:
                        if len(ring_i) == 3:
                            ring_3_center = True
                            if iat in ring_i and kat in ring_i:
                                ring_3_all = True
                        elif len(ring_i) == 4:
                            ring_4_center = True
                            if iat in ring_i and kat in ring_i:
                                ring_4_all = True
                        else:
                            if iat in ring_i and kat in ring_i:
                                ring_else = True
                if coord == 2 or coord == 4 or coord == 6:
                    #  sp2:  trigonal planar, equatorial plane of trigonal bipyramidal
                    #  square planar
                    #  octahedral
                    # -> k * (1 - cos(n*theta))
                    is_cosine_periodic = True
                    if coord == 2:
                        if ring_3_center == True:
                            #kijk *= 0.5
                            if ring_3_all == True:
                                n = 2
                                B = -1
                                #kijk *= 0.44444444
                                kijk *= 0.33333333
                            elif ring_else == True:
                                #n = 3
                                #B = -1
                                n = 1
                                B = 1
                                kijk *= 0.5
                            else:
                                n = 1
                                B = 1
                                #kijk *= 0.11111111
                                kijk *= 0.5
                        elif ring_4_center == True:
                            if ring_4_all == True:
                                n = 2
                                B = -1
                                #kijk *= 0.33333333
                                kijk *= 3.0
                            elif ring_else == True:
                                #n = 3
                                #B = -1
                                n = 1
                                B = 1
                                kijk *= 0.5
                            else:
                                n = 1
                                B = 1
                                kijk *= 0.5
                        else:
                            n = 3
                            B = -1
                    else:
                        n =  4
                        B =  1
                    C = kijk * 0.5 
                else:
                    # general sp3 case
                    # k * (c0 + c1*cos(theta) + c2*cos(2*theta))
                    C2 = 1.0 / (4.0*math.sin(theta0)**2)
                    if ring_3_center == True:
                        if ring_3_all == True:
                            kijk *= 0.33333333
                        else:
                            kijk *= 0.5
                    elif ring_4_center == True:
                        if ring_4_all == True:
                            #kijk *= 0.33333333
                            kijk *= 0.2
                        else:
                            kijk *= 0.5
                    C1 = -4.0 * C2*math.cos(theta0)
                    C0 = C2*(2.0*math.cos(theta0)**2+1.0)

            astring = ",".join(atom_types)
            if is_cosine_periodic:
                parind_ang = f"cosine_periodic->({astring})_{iang}|UFF" 
            else:
                parind_ang = f"fourier->({astring})_{iang}|UFF" 
    
            if self.mol.ff.parind["ang"][iang] == None:
                self.mol.ff.parind["ang"][iang] = [] 
            self.mol.ff.parind["ang"][iang].append(parind_ang)
            if is_cosine_periodic:
                self.mol.ff.par["ang"][parind_ang] = ("cosine_periodic",[C,B,n]) 
            else:
                self.mol.ff.par["ang"][parind_ang] = ("fourier_uff",[kijk,C0,C1,C2]) 
    
     
    # torsions (dihedrals)
    @timer('assign dihed param')
    def assign_dihed_param(self, atypes, sp, bndlst, bondorder, rings=[]):
        """
        Assigns the dihedral parameters for the UFF potential terms 

        """
        remove = []
        for idih,dih in enumerate(self.mol.ff.ric.dih):
            iat = dih[0] 
            jat = dih[1] 
            kat = dih[2]
            lat = dih[3]
            uff_typei = atypes[dih[0]]
            uff_typej = atypes[dih[1]]
            uff_typek = atypes[dih[2]]
            uff_typel = atypes[dih[3]]
            atom_types = [uff_typei,uff_typej,uff_typek,uff_typel]
            if uff_typej > uff_typek:
                atom_types.reverse()
    
            parind_dih = "harm->(" + ",".join(atom_types) + f")_{idih}|UFF" 
    
            V_tor = 1.0 
    
            bojk = bondorder[bndlst[jat][kat]] 

            # Note: UFF torsional terms not for all elements
            ei = self.mol.elems[iat].capitalize()
            ej = self.mol.elems[jat].capitalize()
            ek = self.mol.elems[kat].capitalize()
            el = self.mol.elems[lat].capitalize()
            exclusion = ["Zn","Cu"]
            do_default = (ei in exclusion) or (ej in exclusion) or (ek in exclusion) or (el in exclusion)

            if sp[jat] == 3 and sp[kat] ==  3:
                #V_tor = math.sqrt(torsional_barriers[uff_typek] * torsional_barriers[uff_typek]) 
                V_tor = math.sqrt(torsion[uff_typej][0] * torsion[uff_typek][0])
                order = 3
                cosTerm = -1  # phi0=60
    
                # special case for single bonds between group 6 elements:
                if bojk == 1.0 and pse_group[self.mol.elems[jat].capitalize()] == 6 and pse_group[self.mol.elems[kat].capitalize()]:  
                  Vj = 6.8
                  Vk = 6.8
                  if self.mol.elems[jat].capitalize() == "O": 
                    Vj = 2.0
                  
                  if self.mol.elems[kat].capitalize() == "O": 
                    Vk = 2.0
                  
                  V_tor = math.sqrt(Vj * Vk)
                  order = 2
                  cosTerm = -1  # phi0=90
            # TODO: utilize linear specials in dih finder and remove these hacks for linear cases
            #linear case:
            elif sp[jat] == 1 or sp[kat] == 1:
                V_tor = 0.0
                order = 2
                cosTerm = 1
            elif sp[jat] == 2 and sp[kat] == 2:
                # other linear case:
                boij = bondorder[bndlst[iat][jat]] 
                bokl = bondorder[bndlst[kat][lat]] 
                if bojk == 2.0 and (boij == 2.0 or bokl == 2.0):
                    V_tor = 0.0
                    order = 2
                    cosTerm = 1
                else:
                    Uj = self.uff_data[uff_typej][7] # RS use param table, makes get_U obsolete
                    Uk = self.uff_data[uff_typek][7]
                    V_tor = 5.0 * math.sqrt(Uj * Uk) * (1. + 4.18 * math.log(bojk)) 
                    order = 2
                    cosTerm = 1
            else:
                V_tor = 1.0
                order = 6
                cosTerm = 1  # phi0 = 0
                if bojk == 1:
                  # special case between group 6 sp3 and non-group 6 sp2:
                  jsp3 = (sp[jat] == 3)
                  j_in_group_6 = pse_group[self.mol.elems[jat].capitalize()] == 6
                  k_in_group_6 = pse_group[self.mol.elems[jat].capitalize()] == 6
                  if (sp[jat] == 3 and j_in_group_6 and (not k_in_group_6)) or (jsp3 and k_in_group_6 and (not j_in_group_6)):
                    Uj = self.uff_data[uff_typej][7] # RS use param table, makes get_U obsolete
                    Uk = self.uff_data[uff_typek][7]                    
                    V_tor = 5.0 * math.sqrt(Uj * Uk) * (1. + 4.18 * math.log(bojk)) 
                    order = 2
                    cosTerm = -1   # phi0 = 90;
                  
                # special case for sp3 - sp2 - sp2
                # (i.e. the sp2 has another sp2 neighbor, like propene)
                elif sp[lat] == 2: 
                    V_tor = 2.0
                    order = 3
    
            # UFF potential:
            # -------------
            #
            # E = 1/2 * V_tor [ 1 - cos(n*phi0) * cos(n*phi) ]
    
            # LAMMPS type:
            # -------------
            #
            #  harmonic  : K * [1 + d*cos(n*phi)] K (energy), d = +1 or -1, n = order 
            #
            k = 0.5 * V_tor
            # TODO: recheck these once modified dihedral potential is implemented
            ring_3_jk = False
            ring_3_ij = False
            ring_3_kl = False
            ring_3_ijk = False
            ring_3_jkl = False
            ring_4_jk = False
            ring_4_ij = False
            ring_4_kl = False
            ring_4_ijk = False
            ring_4_jkl = False
            ring_4_ijkl = False
            for ring_i in rings:
                if len(ring_i) == 3:
                    if jat in ring_i:
                        if kat in ring_i:
                            ring_3_jk = True
                            if iat in ring_i:
                                ring_3_ijk = True
                            elif lat in ring_i:
                                ring_3_jkl = True
                        elif iat in ring_i:
                            ring_3_ij = True
                    elif kat in ring_i:
                        if lat in ring_i:
                            ring_3_kl = True
                if len(ring_i) == 4:
                    if jat in ring_i:
                        if kat in ring_i:
                            ring_4_jk = True
                            if iat in ring_i:
                                ring_4_ijk = True
                                if lat in ring_i:
                                    ring_4_ijkl = True
                            elif lat in ring_i:
                                ring_4_jkl = True
                        elif iat in ring_i:
                            ring_4_ij = True
                    elif kat in ring_i:
                        if lat in ring_i:
                            ring_4_kl = True
                break
            if ring_3_jk == True:
                if ring_3_ijk == True:
                    if sp[kat] == 2:
                        k = 0.0
                    else:
                        k *= 0.33333333
                elif ring_3_jkl == True:
                    if sp[jat] == 2:
                        k = 0.0
                    else:
                        k *= 0.33333333
                elif sp[jat] == 2 or sp[kat] == 2:
                    k = 0.0
                else:
                    k = 0.5
            elif ring_3_ij == True:
                if sp[jat] == 2:
                    k = 0.0
                else:
                    k *= 1.0
            elif ring_3_kl == True:
                if sp[kat] == 2:
                    k = 0.0
                else:
                    k *= 1.0
            if ring_4_jk == True:
                if ring_4_ijk == True:
                    if ring_4_ijkl == True:
                        k *= 2.0
                    elif sp[kat] == 2:
                        k = 0.0
                    else:
                        k *= 0.33333333
                elif ring_4_jkl == True:
                    if sp[jat] == 2:
                        k = 0.0
                    else:
                        k *= 0.33333333
                elif sp[jat] == 2 or sp[kat] == 2:
                    k = 0.0
                else:
                    k = 0.5
            elif ring_4_ij == True:
                if sp[jat] == 2:
                    k = 0.0
                else:
                    k *= 1.0
            elif ring_4_kl == True:
                if sp[kat] == 2:
                    k = 0.0
                else:
                    k *= 1.0
            if self.mol.ff.parind["dih"][idih] == None:
                self.mol.ff.parind["dih"][idih] = [] 
            self.mol.ff.parind["dih"][idih].append(parind_dih)
            self.mol.ff.par["dih"][parind_dih] = ("harmonic",[k,-cosTerm,order])
    
    
    # inversion/oop (improper)
    @timer('assign oop param')
    def assign_oop_param(self, atypes, rings=[]):
        """
        Assigns the out-of-plane bending parameters for the UFF potential terms 

        """
        for ioop,oop in enumerate(self.mol.ff.ric.oop):
            iat = oop[0] # the central atom is the first one in the list 
            uff_typei = atypes[oop[0]]
            uff_typej = atypes[oop[1]]
            uff_typek = atypes[oop[2]]
            uff_typel = atypes[oop[3]]
            atom_types = [uff_typei,uff_typej,uff_typek,uff_typel]
    
            parind_oop = "fourier->(" + ",".join(atom_types) + f")_{ioop}|UFF" 
 
            # Non-zero cases:
            #  C_R - O_2 - wild - wild
            #  C_R - wild - O_2 - wild
            #  C_R - wild - wild - O_2
            #  C_2 - O_2 - wild - wild
            #  C_2 - wild - O_2 - wild
            #  C_2 - wild - wild - O_2
            #  C_R - wild - wild - wild
            #  C_2 - wild - wild - wild
            #  N_2 - wild - wild - wild
            #  N_R - wild - wild - wild 
            #  O_2 - wild - wild - wild
            #  O_R - wild - wild - wild   
 
            elemi = self.mol.elems[iat].capitalize()
            # TODO C_2 C_R
            if elemi in ["C","N","O"]:
                C0 =  1.0
                C1 = -1.0
                C2 =  0.0
                kijkl = 6.0
                # check is C is bound to O_2 # TODO check if correct?
                if elemi == "C" and "O_2" in [uff_typej,uff_typek,uff_typel]:
                    kijkl = 50.0
            elif elemi in ["P","As","Sb","Bi"]:
                # Thanks to rdkit
                w0 = math.pi / 180.0
                if elemi == "P":
                  w0 *= 84.4339
                elif elemi == "As":
                  w0 *= 86.9735
                elif elemi == "Sb":
                  w0 *= 87.7047
                elif elemi == "Bi":
                  w0 *= 90.0
                C2 = 1.0
                C1 = -4.0 * math.cos(w0)
                C0 = -(C1 * math.cos(w0) + C2 * math.cos(2.0 * w0))
                kijkl = 22.0 / (C0 + C1 + C2)
            else:
                C0 =  1.0
                C1 = -1.0
                C2 =  0.0
                kijl = 0.0
            kijkl /= 3.0 # side comment in UFF paper
            if self.mol.ff.parind["oop"][ioop] == None:
                self.mol.ff.parind["oop"][ioop] = []
            self.mol.ff.parind["oop"][ioop].append(parind_oop)
            self.mol.ff.par["oop"][parind_oop] = ("fourier",[kijkl,C0,C1,C2,0])
    
    def get_uff_types(self):
        """
        Helper function for the interface to lammps. Returns the UFF atom types 

        """
        if self.uff_types != None:
            return self.uff_types
        else:
            raise ValueError("UFF types not (yet) determined!")

    def get_unique_elements(self):
        """
        Helper function for the interface to lammps. Returns a list of the unique atoms 

        """
        ff_type = {}
        count = 0
        for elem,ufftype in zip(self.mol.elems,self.get_uff_types()):
            if not ufftype in ff_type:
                count += 1
                type = count
                ff_type[ufftype] = type
                self.unique_atom_types[ufftype] = elem.capitalize()
        return self.unique_atom_types.values() 
