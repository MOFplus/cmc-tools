# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:25:43 2017

@author: rochus


        addon module FF to implement force field infrastructure to the molsys
        
        contains class ric
        contains class ff

        Revisions:
        - 03/24 RS write ff as yaml file (par file is yaml and also fpar!)

"""
def string2bool(s):
    return s.lower() in ("True", "true", "yes")

def bool2string(b):
    return str(b)

# RS (2024) do we need this here? why and for what?
# seems this is used in assign FF but that part should go from the addon into a utility script
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

import numpy as np
# RS (2024) do we need this here? wha and for what?
# np.seterr(invalid='raise')

import uuid
import molsys
from molsys.util.timer import timer, Timer
from molsys.util import elems
from molsys.util.ff_descriptors import desc as ff_desc  # RS i think this can go ... with yaml format and lmps_pots 
from molsys.util.aftypes import aftype, aftype_sort
from molsys.util.ffparameter import potentials, varpar, api_cache # RS which do we really need? 
from molsys.addon import base

import itertools
import copy
import string

import json

from pylmps import lammps_pots
import ruamel.yaml as ryaml

import logging
logger = logging.getLogger("molsys.ff")
logger.setLevel(logging.ERROR)

class ic(list):
    """
    Class inherited from flist which accepts attributes.
    Non-existing attributes return None instead of raising an error.
    """

    def __init__(self, *args, **kwargs):
        list.__init__(self,*args)
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.used = False
        return
        
    def __getattr__(self, name):
        """
        if name is not an attribute return None instead of raising an error
        """
        if not name in self.__dict__:
            return None
        else:
            return self.__dict__[name]
        

    def to_string(self, width=None, filt=None, inc=0):
        """
        Method to generate a string representation of the ic.

        Kwargs:
            width (int): Defaults to None. Width of the string representation
            filt (str): Defaults to None. 
            inc (int): Defaults to 0.
        
        Returns:
            str: string representation of the ic
        """
        form = "%d "
        if width: form = "%%%dd " % width
        attrstring = ""
        if filt:
            for k in self.__dict__:
                if filt == "all" or k in filt:
                    if self.__dict__[k] != None:
                        attrstring += " %s=%s" % (k, self.__dict__[k])
        if inc!=0:
            values = tuple(np.array(list(self))+inc)
        else:
            values = tuple(self)
        return ((len(self)*form) % values) + attrstring


class ric:
    """
    Class to detect and keep all the (redundant) internal coordinates of the system.

    Args:
        mol(molsys.mol): Parent mol object in which the rics should be detected.
    """

    def __init__(self, mol):
        self.timer = Timer("RIC")        
        self._mol      = mol
        #self.conn      = mol.conn
        #self.natoms    = mol.natoms
        #self.xyz       = mol.xyz
        self.aftypes   = []
        for i, a in enumerate(mol.get_atypes()):
            self.aftypes.append(aftype(a, mol.fragtypes[i])) # RS (2024) these aftypes are always complete (no wildcard/truncations)
        return

    @property
    def natoms(self):
        return self._mol.natoms

    @property
    def conn(self):
        return self._mol.conn
        
    @property
    def xyz(self):
        return self._mol.xyz
    
    @property
    def elems(self):
        return self._mol.elems

    #@property
    #def aftypes(self):
    #    aftypes   = []
    #    for i, a in enumerate(self._mol.get_atypes()):
    #        self.aftypes.append(aftype(a, self._mol.fragtypes[i]))
    #    return aftypes


    def find_rics(self, specials={"linear": [], "sqp":[], "noop": []}, smallring = False, for_uff=False,
                  exclude_3rings=False, exclude_4rings=False):
        """
        Method to find all rics of the system

        Kwargs:
            specials(dict): Defaults to {"linear": [], "sqp":[]}. Dictionary of special atom types.
                which has to be included in the search.
            smallring(bool): Defaults to False. If true a smallring check for is performed.


        RS (2024) important change:

        during the detection of the rics we sort the atom indices (they are found in random order)
        in a way that they adhere to the sorting of the correpsonding (full) aftypes.
        This means that also for potentials with a specific order of the parameters we can be sure that
        the ordering in the ric is the same as in the pname of the potential.
        An exception are truncated and wildcard aftypes, since here the ordering might be differnt.
        We will therefore need to test on changes in order of params when truncated pnames are used in the database.
        The simplest and probably useful method is to exclude any terms with ordered params when truncated or wildcarded
        aftypes are used (these are typically cross terms ... not really reasonable to use them for truncated or wildcarded
        aftpyes.

        """
        self.bnd    = self.find_bonds()
        self.ang    = self.find_angles()
        self.oop    = self.find_oops(for_uff, exclude_3rings, exclude_4rings, **specials)
        self.dih    = self.find_dihedrals(**specials)
        #self.rea    = self.find_reactives()
        if smallring: self.check_smallrings()
        self.report()
        # self.timer.report()
        return


    def set_rics(self, bnd, ang, oop, dih, sanity_test=False):
        """
        Method to set the rics from outside. Args has to be a properly sorted lists of 
        ic objects as if they are supplied by find_rics.

        Args:
            bnd(list): list of all bonds
            ang(list): list of all angles
            oop(list): list of all out-of planes
            dih(list): list of all dihedrals

        Kwargs:
            sanity_test(bool): Defaults to True. If True a sanity check is performed
                if the supplied RICs belong to the current system.
        """
        self.bnd = bnd
        self.ang = ang
        self.oop = oop
        self.dih = dih
        if sanity_test:
            # find bonds and check if they are equal ... this should be a sufficent test that the rest is the same, too
            
            mol_bnd = self.find_bonds()
            if sorted(mol_bnd) != sorted(bnd):
                raise ValueError("The rics provided do not match the mol object!")
        return

    def sort_bond(self, idx):
        """
        Method used for sorting bonds, according to their
        aftypes.
        
        Args:
            idx (list): list of indices defining a bond

        Returns:
            list: sorted list of indeces defining the bond
        """
        if self.aftypes[idx[0]] > self.aftypes[idx[1]]:
            idx.reverse()
        elif idx[0] > idx[1] and self.aftypes[idx[0]] == self.aftypes[idx[1]]:
            idx.reverse()
        return idx


    @timer("find bonds")
    def find_bonds(self):
        """
        Method to find all bonds of a system.
        
        Returns:
            list: list of internal coordinate objects defining all bonds
        """
        bonds=[]
        for a1 in range(self.natoms):
            for a2 in self.conn[a1]:
                if a2 > a1:
                    if self.aftypes[a1] <= self.aftypes[a2]: 
                        bonds.append(ic([a1, a2]))
                    else:
                        bonds.append(ic([a2, a1]))
        return bonds

    def sort_angle(self, idx):
        """
        Method used for sorting angles, according to their
        aftypes.
        
        Args:
            idx (list): list of indices defining an angle
        
        Returns:
            list: sorted list of indeces defining the angle
        """
        if self.aftypes[idx[0]] > self.aftypes[idx[2]]:
            idx.reverse()
        elif idx[0] > idx[2] and self.aftypes[idx[0]] == self.aftypes[idx[2]]:
            idx.reverse()
        return idx

    @timer("find angles")
    def find_angles(self):
        """
        Method to find all angles of a system

        Returns:
            list: list of internal coordinate objects defining all angles
        """
        angles=[]
        for ca in range(self.natoms):
            apex_atoms = self.conn[ca]
            naa = len(apex_atoms)
            for ia in range(naa):
                aa1 = apex_atoms[ia]
                other_apex_atoms = apex_atoms[ia+1:]
                for aa2 in other_apex_atoms:
                    if self.aftypes[aa1] < self.aftypes[aa2]:
                        angles.append(ic([aa1, ca, aa2]))
                    elif self.aftypes[aa1] == self.aftypes[aa2]:
                        if aa1 < aa2:
                            angles.append(ic([aa1, ca, aa2]))
                        else:
                            angles.append(ic([aa2, ca, aa1]))
                    else:
                        angles.append(ic([aa2, ca, aa1]))
        return angles

    @timer("find oops")
    def find_oops(self, for_uff=False, exclude_3rings=False, exclude_4rings=False,  linear = [], sqp = [], noop = []):
        """
        Method to find all oops of a system
        
        Args:
            for_uff (boolean): In case if use for UFF, we need to create permutations of the oop
            exclude_3rings (boolean): If True, exclude oop terms at three-membered ring centers
            exclude_4rings (boolean): If True, exclude oop terms at four-membered ring centers

        Returns:
            list: list of internal coordinate objects defining all oops

        RS (2024) also for oops we sort the atom indices in a way that they adhere to the sorting of the correpsonding (full) aftypes.
        """
        oops=[]
        # there are a lot of ways to find oops ...
        # we assume that only atoms with 3 partners can be an oop center
        for ta in range(self.natoms):
            if ((len(self.conn[ta]) == 3) and (self.aftypes[ta] not in noop)): # RS (2024) we can exclude some atom types from being oops by adding the aftypes to noop in specials
                # ah! we have an oop .. need to sort the last three atoms by aftype
                aft = [self.aftypes[i] for i in self.conn[ta]]
                a1, a2, a3 = [self.conn[ta][i] for i in sorted(range(3), key=lambda k: aft[k])]
                if exclude_3rings:
                    # catch 3-membered rings
                    if a1 in self.conn[a2] or a1 in self.conn[a3] or a2 in self.conn[a3]:
                        # no we don't!
                        continue
                if exclude_4rings:
                    # catch 4-membered rings
                    neighbors = [a1, a2, a3]
                    if any((s in self.conn[aj] for ai in neighbors for s in [nb for nb in self.conn[ai] if nb !=ta] for aj in [nb for nb in neighbors if nb != ai])):
                        # no we don't!
                        continue
                if for_uff:
                    oops.append(ic([ta, a1, a2, a3])) # RS (2024) in this way the last two atoms for uff should be also sorted
                    oops.append(ic([ta, a2, a1, a3]))
                    oops.append(ic([ta, a3, a1, a2]))
                    ### FIX THIS .. babak's version:
                    # oops.append(ic([ta, a1, a2, a3]))
                    # oops.append(ic([ta, a2, a1, a3]))
                    # oops.append(ic([ta, a3, a1, a2]))
                else:
                    oops.append(ic([ta, a1, a2, a3]))
        return oops

    @timer("find dihedrals")
    def find_dihedrals(self, linear = [], sqp = [], noop=[]):
        """
        Method to find all dihedrals of a system

        Kwargs:
            linear(list): Defaults to []. List of linear atom types.
            sqp(list): Defaults to []. List of square planar atom types.

        Returns:
            list: list of internal coordinate objects defining all dihedrals
        """
        dihedrals=[]
        for a2 in range(self.natoms):
            for a3 in self.conn[a2]:
                # avoid counting central bonds twice
                if a3 > a2:
                    endatom1 = list(self.conn[a2])
                    endatom4 = list(self.conn[a3])
                    endatom1.remove(a3)
                    endatom4.remove(a2)
                    ### check if a3 or a2 is a linear one
                    lin = False
                    stubb = False
                    while self.aftypes[a2] in linear:
                        assert len(endatom1) == 1
                        lin = True
                        a2old = a2
                        a2 = endatom1[0]
                        endatom1 = list(self.conn[a2])
                        endatom1.remove(a2old)
                        ### we have now to check for stubbs
                        if len(endatom1) == 0:
                            stubb = True
                            break
                    if stubb: continue
                    while self.aftypes[a3] in linear:
                        assert len(endatom4) == 1
                        lin = True
                        a3old = a3
                        a3 = endatom4[0]
                        endatom4 = list(self.conn[a3])
                        endatom4.remove(a3old)
                        ### we have now to check for stubbs
                        if len(endatom1) == 0:
                            stubb = True
                            break
                    if stubb: continue
                    for a1 in endatom1:
                        con1 = list(self.conn[a1])
                        for a4 in endatom4:
                            ring = None
                            if a1 == a4: continue
                            if con1.count(a4):
                                ring = 4
                            else:
                                con4 = list(self.conn[a4])
                                for c1 in con1:
                                    if con4.count(c1):
                                        ring = 5
                                        break
                            # sort d first for coupling term purposes
                            if self.aftypes[a2] > self.aftypes[a3]:
                                d = ic([a4,a3,a2,a1], ring = ring)
                            elif self.aftypes[a2] == self.aftypes[a3]:
                                if self.aftypes[a1] > self.aftypes[a4]:
                                    d = ic([a4,a3,a2,a1], ring = ring)
                                else:
                                    d = ic([a1,a2,a3,a4], ring = ring)
                            else:
                                d = ic([a1,a2,a3,a4], ring = ring)
                            #d = ic([a1,a2,a3,a4], ring = ring)
                            if lin:
                                ### in case of a dihedral due to dihedral shifts,
                                ### it has to be checked if we have this dihedral already
                                if d not in dihedrals:
                                    dihedrals.append(d)
                            elif self.aftypes[a2] in sqp:
                                # calculate angle a1 a2 a3
                                if abs(self.get_angle([a1,a2,a3])-180.0) > 10.0:
                                    dihedrals.append(d)
                            elif self.aftypes[a3] in sqp:
                                # calculate angle a2 a3 a4
                                if abs(self.get_angle([a2,a3,a4])-180.0) > 10.0:
                                    dihedrals.append(d)
                            else:
                                dihedrals.append(d)
        return dihedrals
    
    def find_reactives(self): #@KK 
        """Find reactive ZIF moieties (M-NCC)

        Returns:
            list: list of participating atom ids
        """
        reactives = []
        allowed_metals = ["zn"]
        for atom_id in range(self.natoms):
            if self.elems[atom_id].lower() != "n":
                continue
            if not (len(self.conn[atom_id]) == 3):
                continue

            conn_elems = [self.elems[x].lower() for x in self.conn[atom_id]]
            if conn_elems.count("c") != 2:
                continue
            
            if any(x in allowed_metals for x in conn_elems):
                c_id = [i for i,x in enumerate(conn_elems) if x == "c"]
                m_id = [x for x in self.conn[atom_id] if x not in c_id][0]
                reactives.append(ic([atom_id, m_id, c_id[0], c_id[1]]))
        return reactives

    @timer("find smallrings")
    def check_smallrings(self):
        """
        Method needed to check if smallrings are present and to mark the
        corresponding bonds and angles.
        """
        for d in self.dih:
            if d.ring:
                # bnds are sorted also by aftypes
                self.bnd[self.bnd.index(self.sort_bond(d[0:2]))].ring = d.ring
                self.bnd[self.bnd.index(self.sort_bond(d[1:3]))].ring = d.ring
                self.bnd[self.bnd.index(self.sort_bond(d[2:4]))].ring = d.ring
                # angles are sorted in respect to aftypes, we have to
                # check both possibilities
                ang = self.sort_angle(d[0:3])              
                self.ang[self.ang.index(ang)].ring = d.ring
                ang = self.sort_angle(d[1:4])
                self.ang[self.ang.index(ang)].ring = d.ring
        return

    def get_distance(self,atoms):
        """
        Computes distance between two atoms.
        
        Parameters:
            atoms (list): list of atomindices

        Returns:
            float: atom dinstance
        """
        xyz = self._mol.apply_pbc(self.xyz[atoms],fixidx = 0)
        apex_1 = xyz[0]
        apex_2 = xyz[1]
        return np.linalg.norm(apex_1-apex_2)

    def get_angle(self,atoms):
        """
        Computes the angle between three atoms
        
        Parameters:
            atoms (list): list of atomindices

        Returns:
            float: angle in degree
        """
        xyz = self._mol.apply_pbc(self.xyz[atoms], fixidx = 0)
        #xyz = self.xyz[atoms]
        apex_1 = xyz[0]
        apex_2 = xyz[2]
        central = xyz[1]
        r1 = apex_1 - central
        r2 = apex_2 - central
        s = np.dot(r1,r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))
        if s < -1.0: s=-1.0
        if s >  1.0: s=1.0
        phi = np.arccos(s)
        return phi * (180.0/np.pi)

    def get_multiplicity(self, n1, n2):
        """
        Routine to estimate the multiplicity of a dihedral from the local topology
        
        Parameters:
            n1 (int): number of connections of central atom 1
            n2 (int): number of connections of central atom 2
        
        Returns:
            int: multiplicity
        """
        assert type(n1) == type(n2) == int
        if   set([n1,n2])==set([5,5]): return 4
        elif set([n1,n2])==set([6,6]): return 4
        elif set([n1,n2])==set([3,6]): return 4 
        elif set([n1,n2])==set([4,4]): return 3
        elif set([n1,n2])==set([2,4]): return 3
        elif set([n1,n2])==set([3,4]): return 3
        elif set([n1,n2])==set([3,3]): return 2
        elif set([n1,n2])==set([2,3]): return 2
        elif set([n1,n2])==set([2,2]): return 1
        else:                          return None



    def get_dihedral(self, atoms,compute_multiplicity=True):
        """
        Computes dihedral angle between four atoms
        
        Args:
            atoms (list): list of atomindices

        Kwargs:
            compute_multiplicity (bool): Defaults to True. If True multiplicity
                is returned together with the angle.

        Returns:
            tuple: tuple of angle in degree and mutliplicity
        """
        xyz = self._mol.apply_pbc(self.xyz[atoms], fixidx = 0)
        apex1 = xyz[0]
        apex2 = xyz[3]
        central1 = xyz[1]
        central2 = xyz[2]
        b0 = -1.0*(central1-apex1)
        b1 = central2-central1
        b2 = apex2-central2
        n1 = np.cross(b0,b1)
        n2 = np.cross(b1,b2)
        nn1 = np.linalg.norm(n1)
        nn2 = np.linalg.norm(n2)
        if (nn1 < 1.0e-13) or (nn2 < 1.0e-13):
            arg = 0.0 
            # TBI: this indicates a linear unit -> skip dihedral at al
        else:
            arg = -np.dot(n1,n2)/(nn1*nn2)
        if abs(1.0-arg) < 10**-14:
            arg = 1.0
        elif abs(1.0+arg) < 10**-14:
            arg = -1.0
        phi = np.arccos(arg)
        ### get multiplicity
        if compute_multiplicity : 
            m = self.get_multiplicity(len(self._mol.conn[atoms[1]]),
                len(self._mol.conn[atoms[2]]))
            return (phi * (180.0/np.pi), m)
        else:
            return (phi * (180.0/np.pi),0)

    def get_oop(self,atoms):
        """
        Dummy function to compute the value of an oop. It returns always 0.0.
        
        Parameters:
            atoms (list): list of atomindices

        Returns:
            float: always 0.0 is returned
        """
        return 0.0

    def compute_rics(self):
        """
        Computes the values of all rics in the system and attaches 
        them to the corresponding ic
        """
        for b in self.bnd: b.value = self.get_distance(list(b))
        for a in self.ang: a.value = self.get_angle(list(a))
        for d in self.dih: d.value = self.get_dihedral(list(d))
        for o in self.oop: o.value = self.get_oop(list(o))
        return

    def report(self):
        """
        Method to reports all found rics by the help of
        the logger object
        """
        logger.info("Reporting RICs")
        logger.info("%7d bonds"     % len(self.bnd))
        logger.info("%7d angles"    % len(self.ang))
        logger.info("%7d oops"      % len(self.oop))
        logger.info("%7d dihedrals" % len(self.dih))
        return



class ff(base):
    """Class to administrate the FF parameters and to assign them.

    The ff class is used to administrate the assignment of FF parameters
    following our specific assignment approach and to vary the parameters
    for FFgen runs.
    
    Args:
        molsys (molsys.mol): mol object for which parameters should be
            assigned.
    
    Kwargs:
        par(potentials): Defaults to None. External potentials object
            which should be used in this ff class. Only needed for multistructure
            fits with FFgen.

    Attr:
        ric_type()
        par()
        par_ind()
    """

    def __init__(self, mol, par = None, special_atypes = {}, smallring = False):
        super(ff,self).__init__(mol)
        self.timer = Timer("FF")
        self.ric = ric(mol)
        # RS (2024) find rics here already during init
        self.ric.find_rics(specials = special_atypes, smallring = smallring)
        self.interaction_types = ["bnd", "ang", "dih", "oop", "cha", "vdw"]
        # defaults RS this needs major overhaul .. what is really part of the FF and what is a setting to be changed. 
        # not everything that is default should show up in yaml.
        self.settings =  {
            "radfact": 1.0,
            "radrule": "arithmetic", 
            "epsrule": "geometric",
            "coul12" : 1.0,
            "coul13" : 1.0,
            "coul14" : 1.0,
            "vdw12"  : 0.0,
            "vdw13"  : 0.0,
            "vdw14"  : 1.0,
            "chargetype": "gaussian",
            "cutoff" : 12.0, 
            "vdwtype": "exp6_damped",
            "cut_lj_inner": 10.0,
            "cut_lj" : 12.0,
            "coreshell": False,
            "tailcorr" : False,
            "chargegen": None,
            "topoqeq_par": None
       }
        self.settings_formatter = {
            "radfact": float,
            "radrule": str,
            "epsrule": str,
            "cut_lj_inner": float,
            "cut_lj": float,
            "cut_coul": float,
            "coul12"  : float,
            "coul13"  : float,
            "coul14"  : float,
            "vdw12"   : float,
            "vdw13"   : float,
            "vdw14"   : float,
            "chargetype": str,
            "vdwtype": str,
            "coreshell": string2bool,
            "cutoff": float,
            "tailcorr": string2bool,
            "chargegen": str,
            "topoqeq_par": str
        }
        self.pair_potentials_initalized = False
        # RS (2024) add initialisation of par and parind here
        self._init_data()
        if par is not None: 
            assert type(par) == potentials
            self.par = par
        else:
            self.par = potentials({
                    "cha": {},
                    "vdw": {},
                    "vdwpr": {},
                    "chapr": {},
                    "reapr": {},
                    "bnd": {},
                    "ang": {},
                    "dih": {},
                    "oop": {},
                    })
        self.fit = False # RS do we need this flag?         
        # RS (2024) we should attach the variable here by default ... maybe the data structrues could added in this file.
        #
        # RS (2024) the following is also taken from the old assignement
        #   needs to be cleaned up and revised, but the datastructures like aftypes are always needed
        # NOTE timer is removed here for the moment .. not critical
        # as a first step we need to generate the fragment graph (RS 2024 really?? only for assigmenet)
        with self.timer("make full graph"): 
            # create full atomistic graph
            self._mol.addon("graph")
            self._mol.graph.make_graph()
        with self.timer("make fragment graph"):
            self._mol.addon("fragments")
            self.fragments = self._mol.fragments
            self.fragments.make_frag_graph(add_atom_map=True)
        with self.timer("make aftypes"):
            # now make a private list of atom types including the fragment name
            self.aftypes = []
            for i, a in enumerate(self._mol.get_atypes()):
                self.aftypes.append(aftype(a, self._mol.fragtypes[i]))
        with self.timer("identify molecules -> molid"):
           # add molid info to ic["vdw"]
           self._mol.graph.get_components()
           for i, at in enumerate(self.ric_type["vdw"]):
               at.molid = self._mol.graph.molg.vp.molid[i]
               self._mol.molid = self._mol.graph.molg.vp.molid.get_array()
        # topoqeq parameter are currently stored in a simple dict with key = atype and val = [sigma, Jii, Xi]
        self.topoqeq_par = {}
        logger.info("initialized the ff addon")
        self.par.FF ="FF"
        return

    def _init_data(self, cha=None, vdw=None):
        """
        Method to setup the internal data structres.

        Kwargs:
            cha(list): Defaults to None. If provided the
                indices in the list are setup as cha internal
                coordinates.
            vdw(list): Defaults to None. If provided the
                indices in the list are setup as vdw internal
                coordinates.
        """
        # make data structures . call after ric has been filled with data either in assign or after read
        # these are the relevant datastructures that need to be filled by one or the other way.
        if cha is None:
            cha = [ic([i]) for i in range(self._mol.natoms)]
        if vdw is None:
            vdw = [ic([i]) for i in range(self._mol.natoms)]
        self.ric_type = {
                "cha": cha,
                "vdw": vdw, 
                "bnd": self.ric.bnd, 
                "ang": self.ric.ang, 
                "dih": self.ric.dih, 
                "oop": self.ric.oop
                }
        self.parind = {
                "cha": [None]*self._mol.natoms,
                "vdw": [None]*self._mol.natoms,
                "bnd": [None]*len(self.ric.bnd),
                "ang": [None]*len(self.ric.ang),
                "dih": [None]*len(self.ric.dih),
                "oop": [None]*len(self.ric.oop)
                }
        return

    def report(self):
        self.timer.report()

    def fixup_refsysparams(self, var_ics = ["bnd", "ang", "dih", "oop"], cross_terms = [], refsysname=None):
        """
        Initialize the params for further fitting
        
        Revision: RS 2024 
        read basic info from ffa_pots.yaml (should be fixed content about how to handle pots)
        and ffa.yaml which is specific for your current problem .. but we might have a standard version for MOF-FF

        This should eventually go from ff addon into a script with interactive possibilities.
        
        OLD:
        Equivalent method to check consistency in the case that unknown parameters should
        be determined eg. fitted. The Method prepares the internal data structures for
        parameterization or sets default values for unkown params.
        :Parameters:
            - var_ics(list, optional): list of strings of internal coordinate names
            for which the fixup should be done, defaults to ["bnd", "ang", "dih", "oop"]
            - strbnd(bool, optional): switch for a forcing a fixup of strbnd terms, defauls to
            False
        """
        import ruamel.yaml as ryaml
        self.ric.compute_rics()                  # compute the geometric values of the rics
        self.par.attach_variables()              # attaches variables datastructure
        # if hasattr(self, "active_zone") == False:
        #     self.active_zone = []
        # defaults = {
        #     "bnd" : ("mm3", 2, "b", ["d","r"], [0.0, 8.0]),
        #     "ang" : ("mm3", 2, "a", ["d","r"], [0.0, 2.0]),
        #     "dih" : ("cos3", 3, "d", ["d","d","d"], [0.0, 15.0]),
        #     "oop" : ("harm", 2, "o", ["d",0.0], [0.0, 1.0]),
        #     "cha" : ("gaussian", 2, "c", ["d","d"], [0.0, 2.0]),
        #     "vdw" : ("buck6d", 2, "v", ["d","d"], [0.0, 2.0])}
        #
        # RS read in datastructures to control the setup from the yaml files (TODO how to find them from some basic env varibale)
        yaml = ryaml.YAML(typ="safe")
        with open("ffa_pots.yaml", "r") as f:
            ffa_pots = yaml.load(f)
        with open("ffa.yaml", "r") as f:
            ffa = yaml.load(f)
        # define which torsional cross terms exist
        # TODO
        cross_types = {}
        # cross_types["ang"] = [["strbnd"],[6]]
        # cross_types["dih"] = [["mbt","ebt","at","aat","bb13"],[4,8,8,3,3]]
        # 
        # refsysname must be set, if it is not a param of this method we use the name of the mol object
        if refsysname is None:
            self.refsysname = self._mol.name
        else:
            self.refsysname = refsysname

        for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
            pvn_count  = 0
            ric = self.ric_type[ic]
            par = self.par[ic]
            parind = self.parind[ic]
            for i, r in enumerate(ric):
                if parind[i] is None:
                    # REV RS 2024 removed active zone here for the moment ... 
                    # if ic == "cha" and i not in self.active_zone:
                    #     self.active_zone.append(i)
                    # not sure if we should sort here or not ... maybe not?
                    # REV RS 2024 remove sorting at all .. all RICs shoudl be sorted accoring to their aftype already.
                    parname = self.get_parname(r)
                    sparname = list(map(str, parname))
                    elemlist = [aft._atype_pure for aft in parname]
                    elemlist = "-".join(elemlist)
                    fullparlist = []
                    # Revision RS 2024
                    # We need to figure out which potential to use and what params to use for it.
                    ffa_ric = ffa[ic]
                    assert ffa_ric[-1][0] == "all" # make sure there is a default
                    for rulepair in ffa_ric:
                        if rulepair[0] == elemlist:
                            ffa_ric_pot = rulepair[1]
                            if len(rulepair) > 2:
                                ffa_ric_pot_cross = rulepair[2:]
                            break
                        elif rulepair[0] == "all":
                            ffa_ric_pot = rulepair[1] # that is the default ... TODO more clever rules to be used here
                            if len(rulepair) > 2:
                                ffa_ric_pot_cross = rulepair[2:]

                    ffa_ric_pot_par = ffa_pots[ic][ffa_ric_pot]
                    fullparlist.append(ffa_ric_pot + "->(" + ",".join(sparname) + ")|" + self.refsysname)
                    ### we have to set the variables here now
                    if not fullparlist[0] in par:
                        if ic in var_ics:
                            pvals = []
                            pfit = []
                            prange = []
                            pvname = []
                            # set up all params (get a value and add fit stuff if needed)
                            for j, p in enumerate(ffa_ric_pot_par):
                                do_fit =True
                                if type(p[0]) == type(0.0):
                                    # this is a number .. no fit
                                    pvals.append(p[0])
                                    do_fit = False
                                    prange.append(None)
                                elif p[0] == "d":
                                    # this is a param do the fit with the given range
                                    pvals.append(0.0)
                                    assert p[1] == "range"
                                    prange.append(p[2])
                                elif p[0] == "r":
                                    # this is a real value .. get it from the ric
                                    pvals.append(r.value)
                                    assert p[1] == "perc"
                                    prange.append([(100-p[2])/100*r.value, (100+p[2])/100*r.value])
                                else:
                                    raise ValueError("unknown option %s in ffa_pots.yaml" % p[0])
                                pfit.append(do_fit)
                                if do_fit:
                                    pvname.append("$%1s%i_%i" % (ic[0], pvn_count, j))
                                else:
                                    pvname.append(None)
                            par[fullparlist[0]] = [ffa_ric_pot, pvals] # enter the values here ... no fitting is set
                            pvn_count += 1
                            # now set up the fit stuff if needed
                            for k, do_fit in enumerate(pfit):
                                if do_fit:
                                    self.par.variables[pvname[k]] = varpar(self.par, name=pvname[k], val=pvals[k], range=prange[k])
                                    self.par.variables[pvname[k]].pos.append((ic, fullparlist[0], k))
                            # cross to be done
                            # hack for cross terms
                            if ic in cross_types.keys():
                              for ctid,ctype in enumerate(cross_types[ic][0]):
                                if ctype in cross_terms:
                                  fullparlist.append(ctype+"->("+",".join(sparname)+")|"+self.refsysname)
                                  count+=1
                                  vnames = list(map(lambda a: "$"+ctype+"%i_%i" % (count, a), range(cross_types[ic][1][ctid])))
                                  par[fullparlist[-1]] = (ctype, vnames)
                                  for idx,vn in enumerate(vnames):
                                      self.par.variables[vn] = varpar(self.par, name = vn)
                                      self.par.variables[vn].pos.append((ic,fullparlist[-1],idx))                                      
                        else:
                            # this needs further treatment .. currently we do not fit anything and put as many zeros as needed.
                            par[fullparlist[0]] = [ffa_ric_pot, len(ffa_ric_pot_par)*[0.0]]
                    parind[i] = fullparlist
        self.set_def_sig(range(self._mol.get_natoms()))
        self.set_def_vdw(range(self._mol.get_natoms()))
        # self.fix_strbnd()
        # self.fix_dih_cross()
        return

    def fix_strbnd(self):
        """
        Method to perform the fixup for strbnd potentials
        """
        ### get potentials to fix
        pots = self.par.variables.varpots
        dels = []
        for p in pots:
            pot, ref, aftypes = self.split_parname(p[1])
            if pot == "strbnd":
                # first check if apex atypes are the same
                if aftypes[0] == aftypes[2]:
                    dels.append(self.par["ang"][p[1]][1][1])
                    self.par["ang"][p[1]][1][1] = self.par["ang"][p[1]][1][0]
                # now distribute ref values
                apot  = "mm3->"+p[1].split("->")[-1] 
                spot1 = self.build_parname("bnd", "mm3", self.refsysname, aftypes[:2])
                spot2 = self.build_parname("bnd", "mm3", self.refsysname, aftypes[1:])
                s1 = self.par["bnd"][spot1][1][1]
                s2 = self.par["bnd"][spot2][1][1]
                a  = self.par["ang"][apot][1][1]
                # del variables
                dels.append(self.par["ang"][p[1]][1][3])
                dels.append(self.par["ang"][p[1]][1][4])
                dels.append(self.par["ang"][p[1]][1][5])
                # rename variables
                self.par["ang"][p[1]][1][3] = s1
                self.par["ang"][p[1]][1][4] = s2
                self.par["ang"][p[1]][1][5] = a
                # redistribute pots to self.variables dictionary
                self.par.variables[s1].pos.append(("ang", p[1],3))
                self.par.variables[s2].pos.append(("ang", p[1],4))
                self.par.variables[a].pos.append(("ang", p[1],5))
        for i in dels: del(self.par.variables[i])
        return

    def fix_dih_cross(self):
        """Method to perform the fixup for class2 dihedral cross term potentials
        """
        pots = self.par.variables.varpots
        dels = []
        crosspots = ["mbt","ebt","at","aat","bb13"]
        for p in pots:
            pot, ref, aftypes = self.split_parname(p[1])
            if pot in crosspots:
              # distribute ref values
              spot1   = self.build_parname("bnd", "mm3", self.refsysname, aftypes[:2])
              spot2   = self.build_parname("bnd", "mm3", self.refsysname, aftypes[2:])
              spotmid = self.build_parname("bnd", "mm3", self.refsysname, aftypes[1:3])
              s1      = self.par["bnd"][spot1][1][1]
              s2      = self.par["bnd"][spot2][1][1]
              smid    = self.par["bnd"][spotmid][1][1]
              apot1   = self.build_parname("ang", "mm3", self.refsysname, aftypes[:3])
              apot2   = self.build_parname("ang", "mm3", self.refsysname, aftypes[1:])
              a1      = self.par["ang"][apot1][1][1]
              a2      = self.par["ang"][apot2][1][1]
              if pot == "bb13":
                  # delete variables
                  dels.append(self.par["dih"][p[1]][1][1])
                  dels.append(self.par["dih"][p[1]][1][2])
                  # rename variables
                  self.par["dih"][p[1]][1][1] = s1
                  self.par["dih"][p[1]][1][2] = s2
                  # redistribute pots to self.variables dictionary
                  self.par.variables[s1].pos.append(("dih", p[1],1))
                  self.par.variables[s2].pos.append(("dih", p[1],2))
              elif pot == "mbt":
                  dels.append(self.par["dih"][p[1]][1][3])
                  self.par["dih"][p[1]][1][3] = smid
                  self.par.variables[smid].pos.append(("dih", p[1],3))
              elif pot == "ebt":
                  dels.append(self.par["dih"][p[1]][1][6])
                  self.par["dih"][p[1]][1][6] = s1
                  self.par.variables[s1].pos.append(("dih", p[1],6))
                  dels.append(self.par["dih"][p[1]][1][7])
                  self.par["dih"][p[1]][1][7] = s2
                  self.par.variables[s2].pos.append(("dih", p[1],7))
              elif pot == "at":
                  dels.append(self.par["dih"][p[1]][1][6])
                  self.par["dih"][p[1]][1][6] = a1
                  self.par.variables[a1].pos.append(("dih", p[1],6))
                  dels.append(self.par["dih"][p[1]][1][7])
                  self.par["dih"][p[1]][1][7] = a2
                  self.par.variables[a2].pos.append(("dih", p[1],7))
              elif pot == "aat":
                  dels.append(self.par["dih"][p[1]][1][1])
                  self.par["dih"][p[1]][1][1] = a1
                  self.par.variables[a1].pos.append(("dih", p[1],1))
                  dels.append(self.par["dih"][p[1]][1][2])
                  self.par["dih"][p[1]][1][2] = a2
                  self.par.variables[a2].pos.append(("dih", p[1],2))
              
                  
        for i in dels: del(self.par.variables[i])
        return


    def set_def_vdw(self,ind):
        """
        Method to set default vdw parameters for a given atom index
        :Parameters:
            - ind(int): atom index
        """
        elements = self._mol.get_elems()
        atypes   = self._mol.get_atypes()
        truncs   = [i.split("_")[0] for i in atypes]
        for i in ind:
            elem  = elements[i]
            at    = atypes[i]
            trunc = truncs[i]
            try:
                prm = elems.vdw_prm[at]
            except:
                try:
                    prm = elems.vdw_prm[trunc]
                except:
                    try:
                        prm = elems.vdw_prm[elem]
                    except:
                        prm = [0.0,0.0]
            self.par["vdw"][self.parind["vdw"][i][0]][1] = prm
        return


    def set_def_sig(self,ind):
        """
        Method to set default parameters for gaussian width for the the charges
        :Parameters:
            - ind(int): atom index
        """
        elements = self._mol.get_elems()
        for i in ind:
            elem = elements[i]
            try:
                sig = elems.sigmas[elem]
            except:
                sig = 0.0            
            self.par["cha"][self.parind["cha"][i][0]][1][1] = sig
        return
    
 
    def varnames2par(self):
        """
        Forces the paramters in the variables dictionary to be written in the internal
        data structures
        """
        if hasattr(self, 'do_not_varnames2par'): return
        self.par.variables(list(self.par.variables.keys()))
        return

    def remove_pars(self,identifier=[]):
        """Remove Variables to write the numbers to the fpar file
        
        Based on any identifier, variables can be deleted from the dictionary
        Identifiers can be regular expressions
        Usage:
            ff.remove_pars(['d'])  -- removes all dihedral variables
            ff.remove_pars(['a10'])  -- removes all entries of angle 10, e.g. a10_0 & a10_1
            ff.remove_pars(['b*_0'])  -- removes all bond entries zero e.g. b1_0 & b5_0, but keeps b1_1 and b5_1

        Keyword Arguments:
            identifier {list of strings} -- string identifiers, can be regexp (default: {[]})
        """
        import re
        for ident in identifier:
            re_ident = re.compile(ident)
            for k in list(self.par.variables.keys()):
                if k.count(ident) != 0:
                    del self.par.variables[k]
                    continue
                # check regexp
                re_result = re.search(re_ident,k)
                if re_result is None: continue
                if re_result.span()[-1] != 0:  #span[1] is zero if the regexp was not found
                    del self.par.variables[k]
                    continue
        return
    
    def break_bonds(self, pattern2del:str, bondeds = ("bnd", "ang", "dih", "oop")):
        """Breaks all bonds of a certain element or atomtype by deleting all related bonded params.
        This method does NOT alter the bondtable of the attached mol object.

        Args:
            pattern2del (str): Name of the element (e.g "zn") or full atomtype (e.g "c3_h1n2@imid")
            bondeds (tuple): All internal coordinates effected
        """
        pattern2del = pattern2del.lower()
        if "@" in pattern2del:
            atype = True
        else:
            assert "_" not in pattern2del, "Either give only an element or the full atom type like 'c3_h1n2@imid' "
            atype = False
        var_par_map = {"bnd":"b", "ang":"a", "dih":"d", "oop":"o"}
        ric_numbersource = self.enumerate_types()

        for bonded in bondeds:
            del_types = []
            ric_numbers = []
            for par in self.par[bonded]:
                bottom = par.find("(") + 1 
                top = par.find(")")
                moiety = par[bottom:top] # only look at the atom types

                atom_types = moiety.split(",")
                for atom_type in atom_types:
                    if not atype:
                        atom = atom_type.split("_")[0]
                    else:
                        atom = atom_type
                    if pattern2del in atom:
                        del_types.append(par)
                        ric_key = par.split("->")[1]
                        ric_number = ric_numbersource[bonded][ric_key]
                        ric_numbers.append(ric_number)

            for delete in del_types:
                del self.par[bonded][delete]
            
            if "variables" in vars(self.par).keys():
                for num in ric_numbers:
                    var_regex = var_par_map[bonded]+str(num)
                    self.remove_pars([var_regex])
                
            del_ids = []
            for i, par in enumerate(self.parind[bonded]):
                if par[0] in del_types:
                    del_ids.append(i)
            for _id in sorted(del_ids, reverse=True):
                del self.parind[bonded][_id]
                del self.ric_type[bonded][_id]   

        logger.info(f"Deleted all parameters for {bondeds} involving {pattern2del}")     
        return
    
    def mask_fitable_params(self, masked_ric_types = None, masked_patterns = None) -> None:
        """Disable parameters from fitting without having to alter the .par file

        Args:
            masked_ric_type (list[str], optional): name of ric types to be masked. Defaults to None.
            masked_pattern (list[str], optional): list of parameter names to be masked. Defaults to None.

        """
        if masked_ric_types is None:
            masked_ric_types = []
        if masked_patterns is None:
            masked_patterns = []

        self.backup_par = copy.deepcopy(self.par)
        new_par = copy.deepcopy(self.par)
        self.masked_vars = []

        for name, variable in self.par.variables.items():
            info = variable.pos
            delete = False
            for i, entry in enumerate(info):
                par_ric_type, par_name, par_id = entry
                if (par_ric_type in masked_ric_types) or (par_name in masked_patterns):
                    self.masked_vars.append(name)
                    delete = True
                    break
            if delete is True:
                del new_par.variables[name]

        self.par = new_par
        logger.info(f"Masked the following variables: {self.masked_vars}")
        return 
    
    def reset_fitmask(self) -> None:
        """Restores the entries masked from self.par.variables in `mask_fitable_params`.
        TODO/Disclaimer:
            This method merly fixes the file write out, it does not allow to chain masks together!
        """
        for var in self.masked_vars:
            self.par.variables[var] = self.backup_par.variables[var]
        return

    def setup_pair_potentials(self):
        """
        Method to setup the pair potentials based on the per atom type assigned parameters
        :Parameters:
            - radfact (int): factor to be multiplied during radius generation, default to 1.0
            - radrule (str): radiusrule, default to arithmetic
            - epsrule (str): epsilonrule, default to geometric
        """
        # one could improve this via an additional dictionary like structure using the vdwpr 
        self.vdwdata = {}
        self.chadata = {}
        self.types2numbers = {} #equivalent to self.dlp_types
        types = list(self.par["vdw"].keys())
        types.sort()
        for i, t in enumerate(types):
            if t not in self.types2numbers.keys():
                self.types2numbers[t]=str(i)
        ntypes = len(types)
        for i in range(ntypes):
            for j in range(i, ntypes):
                if "vdwpr" in self.par and len(self.par["vdwpr"].keys()) > 0:
                    poti,refi,ti = self.split_parname(types[i])
                    potj,refj,tj = self.split_parname(types[j])
                    assert poti == potj
                    assert refi == refj
                    parname = self.build_parname("vdwpr", poti, refi, [ti[0],tj[0]])
                    if parname in self.par["vdwpr"].keys():
                        # found an explicit parameter
                        par_ij = self.par["vdwpr"][parname]
                        self.vdwdata[types[i]+":"+types[j]] = par_ij
                        self.vdwdata[types[j]+":"+types[i]] = par_ij
                        continue
                # if "chapr" in self.par and len(self.par["chapr"].keys()) > 0:
                #     poti,refi,ti = self.split_parname(types[i])
                #     potj,refj,tj = self.split_parname(types[j])
                #     assert poti == potj
                #     assert refi == refj
                #     parname = self.build_parname("chapr", poti, refi, [ti[0],tj[0]])
                #     if parname in self.par["chapr"].keys():
                #         par_ij = self.par["chapr"][parname]
                #         self.chadata[types[i]+":"+types[j]] = par_ij
                #         self.chadata[types[j]+":"+types[i]] = par_ij
                #         continue
                par_i = self.par["vdw"][types[i]][1]
                par_j = self.par["vdw"][types[j]][1]
                pot_i =  self.par["vdw"][types[i]][0]
                pot_j =  self.par["vdw"][types[j]][0]
                if pot_i == pot_j:
                    pot = pot_i
                else:
                    raise IOError("Can not combine %s and %s" % (pot_i, pot_j))
                if pot == "buck6d":
                    if self.settings["radrule"] == "arithmetic":
                        rad = self.settings["radfact"]*(par_i[0]+par_j[0])
                    elif self.settings["radrule"] == "geometric":
                        rad = 2.0*self.settings["radfact"]*np.sqrt(par_i[0]*par_j[0])
                    else:
                        raise IOError("Unknown radius rule %s specified" % self.settings["radrule"])
                    if self.settings["epsrule"] == "arithmetic":
                        eps = 0.5 * (par_i[1]+par_j[1])
                    elif self.settings["epsrule"] == "geometric":
                        eps = np.sqrt(par_i[1]*par_j[1])
                    else:
                        raise IOError("Unknown epsilon rule %s specified" % self.settings["epsilon"])
                    par_ij = (pot,[rad,eps])
                elif pot == "buck":
                    if self.settings["epsrule"] == "geometric":
                        A = np.sqrt(par_i[0]*par_j[0])
                        C = np.sqrt(par_i[2]*par_j[2])
                    elif self.settings["epsrule"] == "arithmetic":
                        A = 0.5 * (par_i[0]+par_j[0])
                        C = 0.5 * (par_i[2]+par_j[2])
                    else:
                        raise IOError("Unknown epsilon rule %s specified" % self.settings["epsilon"])
                    if self.settings["radrule"] == "arithmetic":
                        B = 0.5*(par_i[1]+par_j[1])
                    elif self.settings["epsrule"] == "geometric":
                        B = np.sqrt(par_i[1]*par_j[1])
                    else:
                        raise IOError("Unknown radius rule %s specified" % self.settings["radrule"])
                    par_ij = (pot,[A,B,C])    
                elif pot == "lj_12_6":
                    if self.settings["radrule"] == "kong":
                        epsrule = "kong"
                        ri = par_i[0] * 2.0
                        rj = par_j[0] * 2.0
                        ei = par_i[1]
                        ej = par_j[1]
                        rad = (1.0/(2**13)*(ei * ri**12)/np.sqrt(ei*ri**6*ej*rj**6) * (1 + ((ej*rj**12)/(ei*ri**12))**(1.0/13))**13)**(1.0/6)
                    elif self.settings["radrule"] == "waldman":
                        epsrule = "waldman"
                        ri = par_i[0] * 2
                        rj = par_j[0] * 2
                        rad = ((ri**6 + rj**6)/2.0)**(1.0/6)
                    elif self.settings["radrule"] == "arithmetic":
                        rad = self.settings["radfact"]*(par_i[0]+par_j[0])
                    elif self.settings["radrule"] == "geometric":
                        rad = 2.0*self.settings["radfact"]*np.sqrt(par_i[0]*par_j[0])
                    else:
                        raise IOError("Unknown radius rule %s specified" % self.settings["radrule"])
                    if self.settings["epsrule"] == "kong":
                        eps = np.sqrt(ei * ri**6 * ej * rj**6)/rad**6
                    elif self.settings["epsrule"] == "waldman":
                        eps = 2.0 * np.sqrt(par_i[1]*par_j[1]) * ((ri**3 * rj**3)/(ri**6 + rj**6))
                    elif self.settings["epsrule"] == "arithmetic":
                        eps = 0.5 * (par_i[1]+par_j[1])
                    elif self.settings["epsrule"] == "geometric":
                        eps = np.sqrt(par_i[1]*par_j[1])
                    else:
                        raise IOError("Unknown epsilon rule %s specified" % self.settings["epsilon"])
                    par_ij = (pot,[rad,eps])
                elif pot == "lj":
                    # "standard" LJ potential ... allow for geometric combi to support UFF
                    if self.settings["epsrule"] == "arithmetic":
                        eps = 0.5 * (par_i[1] + par_j[i])
                    elif self.settings["epsrule"] == "geometric":
                        eps = np.sqrt(par_i[1]*par_j[1])
                    else:
                        raise NotImplementedError
                    if self.settings["radrule"] == "arithmetic":
                        rad = self.settings["radfact"]*(par_i[0]+par_j[0])
                    elif self.settings["radrule"] == "geometric":
                        rad = 2.0*self.settings["radfact"]*np.sqrt(par_i[0]*par_j[0])
                    else:
                        raise NotImplementedError
                    par_ij = (pot, [rad, eps])
                # all combinations are symmetric .. store pairs bith ways
                self.vdwdata[types[i]+":"+types[j]] = par_ij 
                self.vdwdata[types[j]+":"+types[i]] = par_ij   

                # self.chadata[types[i]+":"+types[j]] = par_ij
                # self.chadata[types[j]+":"+types[i]] = par_ij          
                #import pdb; pdb.set_trace()
        self.pair_potentials_initalized = True
        return




    def atoms_in_subsys(self, alist, fsubsys):
        """
        this helper function checks if all fragments of atoms (indices) in alist
        appear in the list of fragments (indices) in fsubsys
        :Parameters:
            - alist(list): list of atom indices
            - fsubsys(list): list of fragment indices 
        :Returns:
            - True or False
        """
        return all(f in fsubsys for f in map(lambda a: self._mol.fragnumbers[a], alist))

    def atoms_in_active(self, alist, subsys):
        """
        this helper function checks if any atom (indices) in alist
        appear in the list of active atoms (indices) in fsubsys

        Args:
            alist (list): list of atom indices
            subsys (list): list of atom indices
            
        Returns:
            bool: True or False
        """
        if subsys is None: return True
        return any(a in subsys for a in alist)

    def get_parname(self, alist):
        """
        helper function to produce the name string using the self.aftypes
        """
        return tuple(map(lambda a: self.aftypes[a], alist))

    def get_parname_equiv(self,alist, ic, refsys):
        """
        helper function to produce the name string using the self.aftypes
        and the information in the self.equivs dictionary
        :Parameters:
            - alist(list): list of atom indices
            - ic(str): corresponding internal coordinate
            - refsys(str): corresponding name of reference system
        :Returns:
            - parname(str): parname
        """
        assert type(ic) == type(refsys) == str
        ### first check if an atom in r is in the predifined active zone
        insides = []
        for i in alist:
            if self.active_zone.count(i) > 0: insides.append(i)
        ### now perform the actual lookup
        try:
            # check for equivs
            equivs = self.equivs[refsys][ic]
            # ok, got some --> try to apply
            # first check if for all insides an equiv is available
            for i in insides: 
                if i not in equivs.keys(): return None
            if len(insides) > 1: return None
            return list(map(lambda a: self.aftypes[a] if a not in equivs.keys() else equivs[a], alist))
        except:
            if len(insides) > 0: return None
            return list(map(lambda a: self.aftypes[a], alist))

    def get_parname_sort(self, alist, ic) -> tuple:
        """helper function to produce the name string using the self.aftypes

        Args:
            alist (list): atom ids for a ric
            ic (str): internal coordinate name

        Returns:
            tuple: _description_
        """
        l = list(map(lambda a: self.aftypes[a], alist))
        # l = [self.aftypes[a] for a in alist] @KK replace?
        return tuple(aftype_sort(l,ic))

    def split_parname(self,name):
        """
        Helper function to exploit the necessary information from the parname:
        :Parameters:
            - name(str): parname
        :Returns:
            - pot(str): potential type
            - ref(str): name of the reference system
            - aftypes(list): list of aftype names
        """
        pot = name.split("-")[0]
        ref = name.split("|")[-1]
        aftypes = name.split("(")[1].split(")")[0].split(",")
        return pot, ref, aftypes

    def build_parname(self, ic, pot, ref, aftypes):
        """
        Helper function to build the parname out of the necessary information
        :Parameters:
            - ic(str): name of the internal coordinate
            - pot(str): pot name
            - ref(str): name of the reference system
            - aftypes(list): list of the involved aftypes
        :Returns:
            - parname
        """
        sortedaft = aftype_sort(aftypes, ic)
        return pot + "->("+",".join(sortedaft)+")|"+ref
        

    def pick_params(self,aft_list,ic,at_list, pardir):
        """
        Hhelper function to pick params from the dictionary pardir using permutations for the given ic
        if len of aft_list == 1 (ic = vdw or cha) no permutations necessary
        :Parameters:
            -aft_list(list): list of aftypes
            -ic(str): name of internal coordinate
            -pardir(dict): dictionary holding the parameters
        :Returns:
            -parname(str): parname
            -params(list): list of the corresponding parameters
        """
        if aft_list is None: return (), None
        ic_perm = {"bnd": ((0,1), (1,0)),
                   "bnd5": ((0,1), (1,0)),
                   "ang": ((0,1,2), (2,1,0)),
                   "ang5": ((0,1,2), (2,1,0)),
                   "dih": ((0,1,2,3),(3,2,1,0)),
                   "oop": ((0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1))}
        if len(aft_list) == 1:
            parname = tuple(aft_list)
            if pardir[ic].index(parname, "wild_ft") >= 0:
                return parname, pardir[ic].__getitem__(parname, wildcards = True)
            else:
                return (), None
        else:         
            # we have to check here also for ring specific parameters
            if ic == "bnd" and at_list.ring == 5:
                ics = ["bnd5", "bnd"]
            elif ic == "ang" and at_list.ring == 5:
                ics = ["ang5", "ang"]
            else:
                ics = [ic]
            # now we have to loop over ics
            for cic in ics:
                perm = ic_perm[cic]
                for p in perm:
                    parname = tuple(map(aft_list.__getitem__, p))
                    if pardir[cic].index(parname, "wild_ft") >= 0:
                        # if we found a bnd5 or ang5 we have to modify
                        # the name of the parameter
                        if cic == "bnd5" or cic == "ang5":
                            param = copy.deepcopy(pardir[cic].__getitem__(parname, wildcards = True))
                            lparname = list(parname)
                            lparname.append("r5")
                            return tuple(lparname), param
                        else:
                            return parname, pardir[cic].__getitem__(parname, wildcards = True)
            # if we get to this point all permutations gave no result
            return (), None

    def enumerate_types(self) -> dict:
        """
        Helper function to assign a number to the type, needed for fileIO
        Return:
            dict: e.g {"bnd": {"(c3_h1n2@imid,h1_c1@imid)": 1, "(c3_h1n2@imid,c3_h1n2@imid)": 2, ...}, "ang": ...}
        """
        # dummy dicts to assign a number to the type
        par_types = {}
        keywords = [i for i in self.par.keys()]
        # check which keywords are avaialbale in the ff object
        #keywords_avail = []
        #for i in keywords:
        #    if i in 
        for ic in keywords:
            ptyp = {}
            i = 1
            for ind in self.par[ic]:
                # cut off the potential type --- if the rest is the same we use the same number
                rind = ind.split("->")[1]
                if not rind in ptyp: 
                    ptyp[rind] = i
                    i += 1
            par_types[ic] = ptyp
        return par_types
 

    def enumerate_types_atype_unique(self):
        """
        Helper function to assign a number to the type, needed for fileIO

        in special cases it is helpful to have an ordered list by just the atomtypes
        this implies that params are not distinguished by refsysnames

        we try this here and return a True/False depending on if that works

        """
        # dummy dicts to assign a number to the type
        OK = True
        par_types = {}
        keywords = [i for i in self.par.keys()]
        for ic in keywords:
            ptyp = {}
            with_ref = [i.split("->")[1] for i in self.par[ic]]
            no_ref = [i.split("|")[0] for i in with_ref]
            # make sure that there are no definitions that rely on the refsystem ... 
            if len(set(with_ref)) != len(set(no_ref)):
                OK = False
            i = 1
            with_ref.sort()
            for rind in with_ref:
                if not rind in ptyp: 
                    ptyp[rind] = i
                    i += 1
            par_types[ic] = ptyp
        return OK, par_types


    ################# IO methods #################################################

    ################# central read/write to ric/par ##############################

    def write(self, fname, fpar = True, try_without_ref=False, yaml=True, compact=False):
        """
        write the rics including the referencing types to an ascii file
        called <fname>.ric and the parameters to <fname>.par
        :Parameters:
            - fname(str): fname
            - fpar(bool): write an fpar file
            - try_without_ref(bool): experimental to write a par file without refsystem info
            - yaml(bool): experimental (new 2024) write par as a yaml output
        """
        if self.mpi_rank > 0:
            return
        # create a random hash to ensure that ric and par file belong together
        hash = str(uuid.uuid4())
        # dummy dicts to assign a number to the type
        if try_without_ref:
            OK, par_types = self.enumerate_types_atype_unique()
            if not OK:
                print ("WARNING: assigning without refnames failed!!")
                logger.info("WARNING: assigning without refnames failed!!")
                # redo regular
                par_types = self.enumerate_types()
        else:
            par_types = self.enumerate_types()
        # write the RICs first
        f = open(fname+".ric", "w")
        f.write("HASH: %s\n" % hash)
        logger.info("Writing RIC to file %s.ric" % fname)
        # should we add a name here in the file? the FF goes to par. keep it simple ...
        for ic in self.interaction_types:
            filt = None
            if ic == "dih" or ic == "bnd" or ic == "ang":
                filt = ["ring"]
            elif ic == "vdw":
                filt = ["molid"]
            else:
                pass
            ric = self.ric_type[ic]
            parind = self.parind[ic]
            ptyp = par_types[ic]
            f.write("%s %d\n" % (ic, len(ric)))
            for i,r in enumerate(ric):
                # we take only the first index and remove the ptype to lookup in ptyp dictionary
                pi = parind[i][0]
                ipi = ptyp[pi.split("->")[1]]
                f.write("%d %d %s\n" % (i+1, ipi, r.to_string(filt=filt, inc=1)))
            f.write("\n")
        f.close()
        # write the par file
        if yaml:
            # note that yaml par files contain fpar always ... there is no real difference between them
            # make the collection .. if that fails then the collect method will raise an error
            if hasattr(self.par, 'variables'): #@KK Take default values instaed of variables
                vals = self.par.variables.vals
                self.par.variables(vals)
            par_coll = {}
            par_coll["hash"] = hash # add RIC hash to guarantee that these two match
            par_coll = self.collect(par_coll, compact=compact, fpar=fpar)
            yaml = ryaml.YAML()
            yaml.width = 120
            yaml.indent(mapping=4, sequence=6, offset=2)
            yaml.default_flow_style = False
            yaml.prefix_colon = ' '
            f = open(fname + ".par", "w")
            # print (par_coll)
            yaml.dump(par_coll, f)
            f.close()
            logger.info("Writing parameter in yaml format to file %s.par" % fname)
            return
        else:
            if hasattr(self.par, 'variables') and fpar == True:
                # this is a fixed up refsystem for fitting
                f = open(fname+".fpar", "w")
                vals = self.par.variables.vals
                self.varnames2par()
                logger.info(f"Writing parameter to file {fname}.fpar")
            else:
                f = open(fname+".par", "w")             
                logger.info("Writing parameter to file %s.par" % fname)
        f.write("HASH: %s\n" % hash)
        f.write("FF %s\n\n" % self.par.FF)
        # write settings
        for k,v in self.settings.items():
            f.write("%-15s %s\n" % (k, str(v)))
        f.write("\n")
        # in py3 the dict keys are not reported as they are initialized. hence we sort here to 
        # restore the same sorting as before, i.e. the list below
        sorted_keys = self.interaction_types
        all_keys = self.par.keys()
        keywords = [x for x in sorted_keys if x in all_keys] + [x for x in all_keys if x not in sorted_keys]
        #keywords = [i for i in self.par.keys()]
        for ic in keywords:
            ptyp = par_types[ic]
            par = self.par[ic]
            f.write(ff_desc[ic])
            f.write("\n")
            f.write("%3s_type %d\n" % (ic, len(par)))
            ind = list(par.keys())
            ind.sort(key=lambda k: ptyp[k.split("->")[1]])
            for i in ind:
                ipi = ptyp[i.split("->")[1]]
                ptype, values = par[i]
                formatstr = " ".join(list(map(lambda a: "%15.8f" if type(a) != str else "%+15s", values)))
                sval = formatstr % tuple(values)
                #sval = (len(values)*"%15.8f ") % tuple(values)
                f.write("%-5d %20s %s           # %s\n" % (ipi, ptype, sval, i))
            f.write("\n")        
        if hasattr(self.par, 'variables') and fpar == True:
            self.par.variables(vals)
            if hasattr(self, 'active_zone'):
                active_zone = np.array(self.active_zone)+1
                f.write(("azone "+len(active_zone)*" %d"+"\n\n") % tuple(active_zone))
            if hasattr(self, 'refsysname'): 
                f.write("refsysname %s\n\n" % self.refsysname)
            f.write("variables %d\n" % len(self.par.variables))
            for k,v in self.par.variables.items():
                f.write("%10s %15.8f %15.8f %15.8f %3s %3s\n" % (v.name, v.val, v.range[0], v.range[1], v.bounds[0], v.bounds[1]))
        f.close()
        return

    def read(self, fname, fit=False):
        """
        read the ric/par files instead of assigning params
        :Parameters:
            - fname(str) : name of <fname>.par und <fname.ric> file
            - fit(bool, optional): specify if an fpar file should be read in, 
            holding fitting information, defaults to False
        """
        hash = None
        fric = open(fname+".ric", "r")
        ric_type = ["bnd", "ang", "dih", "oop", "cha", "vdw", "vdwpr", "chapr", "reapr"]
        ric_len  = [2    , 3    , 4    , 4    , 1    , 1    , 2      , 2      , 2      ]
        ric      = {}
        # read in ric first, store the type as an attribute in the first place
        stop = False
        assigned = []
        while not stop:
            line = fric.readline()
            if len(line)==0:
                # end of ric file
                stop = True
            sline = line.split()
            if len(sline)> 0:
                if sline[0] == "HASH:": 
                    hash = sline[1]
                elif sline[0] in ric_type:
                    curric = sline[0]
                    curric_len = ric_len[ric_type.index(curric)]
                    assigned.append(curric)
                    nric = int(sline[1])
                    rlist = []
                    for i in range(nric):
                        sline = fric.readline().split()
                        rtype = int(sline[1])
                        aind  = list(map(int, sline[2:curric_len+2]))
                        aind  = np.array(aind)-1
                        icl = ic(aind, type=rtype)
                        for attr in sline[curric_len+2:]:
                            atn,atv = attr.split("=")
                            icl.__setattr__(atn, int(atv))
                        rlist.append(icl)
                    ric[curric] = rlist    
        fric.close()
        logger.info("read RIC from file %s.ric" % fname)
        # now add data to ric object .. it gets only bnd, angl, oop, dih
        self.ric.set_rics(ric["bnd"], ric["ang"], ric["oop"], ric["dih"])
        # time to init the data structures .. supply vdw and cha here
        self._init_data(cha=ric["cha"], vdw=ric["vdw"])
        # self._init_pardata() ## RS part of __init__ now
        # check if molid has been read from ric file and set to molid in the parent mol object
        # NOTE: the ic is a list with attributes that never return an error .. non exisiting attributes are returned as None
        rt_vdw = self.ric_type["vdw"]
        if rt_vdw[0].molid is not None:
            self._mol.molid = np.array([x.molid for x in rt_vdw])
        # now open and read in the par file
        if fit:
            nkeys={}
            self.fit=True
            fpar = open(fname+".fpar", "r")
            # in the fit case we first screen for the variables block and read it ini
            self.par.attach_variables()
            #self.variables = varpars()
            line = fpar.readline()
            stop = False
            azone = False
            vars  = False
            refsysname = False
            while not stop:
                sline = line.split()
                if len(sline)>0:
                    if sline[0] == "azone":
                        self.active_zone = [int(x)-1 for x in sline[1:]]
                        azone = True
                    elif sline[0] == "variables":
                        nvar = int(sline[1])
                        for i in range(nvar):
                            sline = fpar.readline().split()
#                            nkey = self.par.variables[sline[0]] = varpar(self.par, sline[0], 
#                                          val = float(sline[1]), 
#                                          range = [float(sline[2]), float(sline[3])], 
#                                          bounds = [sline[4], sline[5]])
                            nkey = self.par.variables.__setitem__(sline[0], varpar(self.par, sline[0], 
                                          val = float(sline[1]), 
                                          range = [float(sline[2]), float(sline[3])], 
                                          bounds = [sline[4], sline[5]]))
                            if nkey != sline[0]:nkeys[sline[0]]=nkey
                        vars = True
                    elif sline[0] == "refsysname":
                        self.refsysname = sline[1]
                        refsysname = True
#                    if azone == vars == refsysname == True:
                    if vars == True:
                        fpar.seek(0)
                        stop = True
                        break
                line = fpar.readline()
                if len(line) == 0:
                    raise IOError("Variables block and/or azone in fpar is missing!")
        else:
            # first check if this is a yaml file format
            # 2024 RS: currently yaml files start with a version : X line (this could change and needs to be adapted, then)
            fpar = open(fname + ".par", "r")
            line = fpar.readline()
            sline = line.split()
            # HACK: use a lower case "hash" in case of yaml
            if sline[0] == "version" or sline[0] == "hash":
                # this is a yaml file it seems ... let#s rewind and read it. the use distribute (as oposed to collect) to distripute the data
                fpar.seek(0)
                # first we need to make sure that yaml works and that lammps pots are registered
                yaml = ryaml.YAML(typ="safe")
                par_coll = yaml.load(fpar) # load parameter collection from yaml
                fpar.close()
                # first check if there is a hash and if it matches:
                if "hash" in par_coll:
                    assert par_coll["hash"] == hash, f"\n\nHashes of RIC and PAR do not MATCH!!! \n\t.par: {par_coll['hash']} \n\t.ric: {hash}"
                else:
                    # what do we do if no hash? quench it?
                    print ("\n\nWARNING: no hash in PAR file to make sure that this matches with RIC!!!\n\n")
                self.distribute(par_coll) # and distribute it
                logger.info("read parameter from yaml par file %s.par" % fname)
                return
            # for yaml files that was it
            # fpar = open(fname+".par", "r")
            fpar.seek(0)
        stop = False
        found_hash = False
        while not stop:         
            line = fpar.readline()
            if len(line) == 0:
                stop = True
            sline = line.split()
            if len(sline)>0:
                if sline[0] == "HASH:":
                    found_hash = True
                    assert sline[1] == hash, "Hashes of ric and par file do not match!"
                if sline[0][0] == "#": continue 
                curric = sline[0].split("_")[0]
                if sline[0]=="FF":
                    self.par.FF = sline[1]
                elif sline[0] in self.settings.keys():
                    self.settings[sline[0]] = self.settings_formatter[sline[0]](sline[1]) #@KK breaks if chargegen is not defined yet!
                elif curric in ric_type:
                    par = self.par[curric]
                    t2ident = {} # maps integer type to identifier
                    ntypes = int(sline[1])
                    for i in range(ntypes):
                        sline = fpar.readline().split()
                        if sline[0][0] == "#": continue 
                        # now parse the line 
                        itype = int(sline[0])
                        ptype = sline[1]
                        ident = sline[-1]
                        param = sline[2:-2]
                        if self.fit:
                            newparam = []
                            # if we read a fpar file we need to test if there are variables
                            for paridx,p in enumerate(param):
                                if p[0] == "$":
                                    # check if variable name was overwritten
                                    if p in nkeys: p = nkeys[p]
                                    #import pdb; pdb.set_trace()
                                    if not p in self.par.variables:
                                        raise IOError("Varible %s undefiend in variable block" % p)
                                    # for multistruc fits the $ name are not anymore valid, it has
                                    # to be checked firtst if the variable is already defined under
                                    # a different name
                                    found = False
                                    for vname in self.par.variables.keys():
                                        if (curric, ident, paridx) in self.par.variables[vname].pos:
                                            found = True
                                            p = vname
                                            break
                                    if not found:
                                        self.par.variables[p].pos.append((curric,ident,paridx))
                                    newparam.append(p)
                                else:
                                    newparam.append(float(p))
                            param = newparam
                        else:
                            param = list(map(float, param))
                        if ident in par:
                            logger.warning('Identifier %s already in par dictionary --> will be overwritten' % ident)
                            raise ValueError("Identifier %s appears twice" % ident)
                        par[ident] = (ptype, param)
                        if itype in t2ident:
                            t2ident[itype].append(ident)
                        else:
                            t2ident[itype] = [ident]
                    # now all types are read in: set up the parind datastructure of the ric
                    if curric != "vdwpr" and curric != "chapr":
                        parind = self.parind[curric]
                        for i,r in enumerate(self.ric_type[curric]):
                            parind[i] = t2ident[r.type]
        fpar.close()
        # check if both fpar and ric file have hashes
        if hash is not None and found_hash == False:
            raise IOError("ric file has a hash, but par has not")
        logger.info("read parameter from file %s.par" % fname)
        ### replace variable names by the current value
        if self.fit: 
            self.par.variables.cleanup()
            self.par.variables()
        return

    ################# convert data to a dict/list strucutre and back for writing and reading to yaml ######

    def collect(self, par_collect, compact=False, fpar=False):
        """collect the parameter into a dict of dict/lists for yaml output
        """
        # first we need to make sure that yaml works and that lammps pots are registered
        if compact:
            assert fpar == False # add fit only for a non-compact output
        try:
            self.lmpspots.rics["bnd"]
        except AttributeError:
            # lmpspots is not there, yet ... get it
            self.lmpspots = lammps_pots.lpots()
            lammps_pots.register_defaults(self.lmpspots)
        # ok, lets get going
        par_collect["version"] = 1.0
        par_collect["settings"] = ryaml.CommentedMap(self.settings)
        # if fpar is True and we have varables then add them to the collection
        add_fit = False
        if hasattr(self.par, "variables") and fpar:
            add_fit = True 
        # convert the rics into dicts and add comments
        par_types = self.enumerate_types()
        for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw", "chapr", "vdwpr", "reapr"]: #@KK
            ic_par = {}
            icp_types = par_types[ic]
            for pname in self.par[ic]:
                p_par = {}
                par = self.par[ic][pname]
                pot, par = par # unpack into potential type and actual params
                p_par["ric"]:int = icp_types[pname.split("->")[1]]
                p_par["pot"] = pot
                if compact:
                    # dump params as a list in flow style (one line)
                    yaml_par = ryaml.CommentedSeq(par)
                    yaml_par.fa.set_flow_style()
                else:
                    # dump params as dict with keys as aparam names and units as comment
                    yaml_par = {}
                    yaml_fit = {}
                    lpt = self.lmpspots.rics[ic][pot]
                    for i, v in enumerate(par):
                        nm   = lpt.params[i]
                        yaml_par[nm] = float(v) # make sure that all values are float and not numpy scalars (yaml does not like it)
                        if add_fit:
                            vv, j = self.par.variables.finditem(ic, pname, i) # this is the name of the variable (string)
                            if vv:
                                vvv = self.par.variables[vv] # this is the varibale object itself (to access the attributes)
                                if j == 0:
                                    # this is the basic definition of the fit variable                                
                                    yaml_fit[nm] = ryaml.CommentedMap({"var": vv, "range": (float(vvv.range[0]), float(vvv.range[1]))}) # make sure this is float not np.float!
                                    if vvv.bounds is not None:
                                        yaml_fit[nm]["bounds"] = (vvv.bounds[0], vvv.bounds[1])
                                else:
                                    # this is a reference to the variable in a cross term, no need to redefine ranges or bounds
                                    yaml_fit[nm] = ryaml.CommentedMap({"var": vv})
                                yaml_fit[nm].fa.set_flow_style()
                    # add units as comments
                    yaml_par = ryaml.CommentedMap(yaml_par)
                    for i in range(lpt.nparams):
                        yaml_par.yaml_add_eol_comment("# " + lpt.units[i], key=lpt.params[i])
                p_par["par"] = yaml_par
                if len(yaml_fit) > 0:
                    p_par["fit"] = yaml_fit
                if add_fit:
                    # check if there is any variable for this potential
                    pass
                ic_par[pname] = p_par
            par_collect[ic] = ic_par
        # do the meta data at the end
        # par_collections["meta"] = 
        # add topoqeq parameter
        par_collect["topoqeq"] = ryaml.CommentedMap(self.topoqeq_par)
        # par_collect["topoqeq"].fa.set_flow_style()
        return par_collect

    def distribute(self, par_coll):
        """distribute data structure read from yaml to internal data 
        """
        # setup the lammps pots in order to map the params back in order 8and to check if units match (TODO)
        try:
            self.lmpspots["bnd"]
        except AttributeError:
            # lmpspots is not there, yet ... get it
            print ("DEBUG Setting up lammps_pots")
            self.lmpspots = lammps_pots.lpots()
            lammps_pots.register_defaults(self.lmpspots)
        # first update (fill) the settings dictionary (do we need to check for types? no!)
        # TODO settigns are a big mess. settings in ff addon have defaults and a formatter and only defautled settings can be read.
        #      all settings are later transfered to ff2lammps settings ... again only defaulted ... 
        #      this needs cleanup and maybe a yaml file with the defaults.
        self.settings.update(par_coll["settings"])
        # set fit to False .. if any param has a fit entry set back to true
        self.fit = False
        fit_par = {}
        for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw", "chapr", "vdwpr", "reapr"]:
            currpar = self.par[ic]
            if ic not in par_coll.keys():
                par_coll[ic] = {}
            inpar   = par_coll[ic]
            t2ident = {} # maps integer type to identifier
            for ident, val in inpar.items():
                param = val["par"]
                ptype = val["pot"]
                # convert param to a list .. make sure all subparams are there and the order is right
                try:
                    param = [param[nm] for nm in self.lmpspots.rics[ic][ptype].params]
                except KeyError:
                    print ("Failed to make param list")
                    print ("Check lammps_pots.py for ic = %s and potential type %s" % (ic, ptype))
                    print ("params in file:        ", param.keys())
                    print ("params in lammps_pots: ", self.lmpspots.rics[ic][ptype].params)
                    raise
                itype = val["ric"]
                currpar[ident] = (ptype, param) # make the entry
                if itype in t2ident:
                    t2ident[itype].append(ident)
                else:
                    t2ident[itype] = [ident]
                # check if there is a fit block
                if "fit" in val:
                    self.fit = True
                    # make an entry on each varable to be fitted
                    for k,v in val["fit"].items():
                        fit_par[(ic, ident, k)] = v
            # now all is parsed: set up the parind datastructure of the ric
            if ic not in ("vdwpr", "chapr", "reapr"): #@KK is "reapr" correct here?
                parind = self.parind[ic]
                for i,r in enumerate(self.ric_type[ic]):
                    parind[i] = t2ident[r.type]
        # if there is soemthing to fit print what we have for debug
        if self.fit:
            # add the variables datastructure to par
            self.par.attach_variables()
            # now generate the entries using the temporary dict fit_par
            # for each varpar a range needs to be defined. bounds can be omitted, then we use ["z", "i"] as a default
            # to make sure that all arpars have a range defined, we run over the fit_par dir twice. if any varpar does not have a range defined we
            # stop with an error
            # NOTE varable names do NOT have to start with "$" any more but if you want to write old fpar files this is still a condition
            varpar_ranges = {}
            for v in fit_par.values():
                if "range" in v:
                    if v["var"] in varpar_ranges:
                        # this varpar has been read before with ranges .. check if they differ
                        assert v["range"] == varpar_ranges[v["var"]], f"Error: the variable {v['var']} has been defined with a differnt range already!"
                    else:
                        varpar_ranges[v["var"]] = v["range"] # append the range for this varpar
            # now we can loop again. for each varpar a varpar_ranges entry must exist 
            for loc, data in fit_par.items():  # loc is the location as (ric, ricname without "pot->", parname -> gives pos via self.lmpspots.rics[ic][ptype].params.index(parname))
                ric = loc[0]
                ident = loc[1]
                pot = par_coll[ric][ident]["pot"]
                pos = self.lmpspots.rics[ric][pot].params.index(loc[2])
                vname = data["var"]
                assert vname in varpar_ranges, f"Error: the variable {vname} has not been defined with a range!"
                value = par_coll[ric][ident]["par"][loc[2]] # ok, this is a bit complex ... we pick up the value of the parameter from the original par_coll
                vrange = varpar_ranges[vname] 
                if "bounds" in data:
                    vbounds = data["bounds"]
                else:
                    vbounds =  ["z","i"]   
                # now generate the entry (in case it does not exist ... in this case we just add the position)
                if vname not in self.par.variables:
                    self.par.variables[vname] = varpar(self.par, vname, value, vrange, vbounds)
                self.par.variables[vname].pos.append((ric, ident, pos))
        # TODO handle meta data
        # TODO deal with vdwpr and chapr
        # read in the topoqeq parameter
        if "topoqeq" in par_coll:
            self.topoqeq_par = par_coll["topoqeq"]
        return



    ################# pack/unpack force field data to numpy arrays ##############################

    def pack(self):
        """
        pack the rics and parameters to numpy arrays (one list of strings is provided to regenerate the names)
        all the data will be packed into a single directory called data

        systems with variables can not be packed

        TODO: currently we only handle the ring attribute, needs to be more general if other attributes will be used in the future

        TBI: we need to pack vdwpr and chapr
        """
        #assert not hasattr(self.par, 'variables'), "Can not pack force field data with variables"
        # if hasattr(self.par, 'variables'): return {"FF": None}
        data = {}
        # dummy dicts to assign a number to the type
        par_types = self.enumerate_types()
        # pack the RICs first
        for ic in self.interaction_types:
            ric = self.ric_type[ic]
            n = len(ric)
            if n > 0:
                l = len(ric[0])+1
                filt = None
                # increment for attributes
                if ic == "dih" or ic == "vdw":
                    l += 1
                parind = self.parind[ic]
                ptyp = par_types[ic]
                ric_data = np.zeros([n,l], dtype="int32")
                for i,r in enumerate(ric):
                    # we take only the first index and remove the ptype to lookup in ptyp dictionary
                    pi = parind[i][0]
                    ipi = ptyp[pi.split("->")[1]]
                    line = [ipi]+list(r)
                    # add ring attribute if it is a dihedral
                    if ic=="dih":
                        if r.ring is not None:
                            line += [r.ring]
                        else:
                            line += [0]
                    # add molid attribute if it is a vdw (per atom)
                    if ic=="vdw":
                        if r.molid is not None:
                            line += [r.molid]
                        else:
                            # we need a -1 because molid=0 is possible
                            line += [-1]
                    ric_data[i] = np.array(line)
                data[ic] = ric_data
            # now pack PARams
            par = self.par[ic]
            npar = len(par)
            if npar > 0:
                ind = list(par.keys())
                ind.sort(key=lambda k: ptyp[k.split("->")[1]])
                # params are stored in a tuple of 2 lists and two numpy arrays
                #      list ptypes (string)  <- ptype
                #      list names (string)   <- i
                #      array npars(n,2) (int)      <- len(values), ipi
                #      array pars(n, maxnpar) (float) <- values
                # first round
                ptypes = []
                names  = []
                npars  = []
                for i in ind:
                    ipi = ptyp[i.split("->")[1]]
                    ptype, values = par[i]
                    ptypes.append(ptype)
                    names.append(i)
                    npars.append([len(values), ipi])
                npars = np.array(npars)
                maxnpar = np.amax(npars[:,0])
                pars = np.zeros([len(ind), maxnpar], dtype="float64")
                # second round .. pack params
                for j,i in enumerate(ind):
                    ptype, values = par[i]
                    pars[j,:npars[j,0]] = np.array(values)
                data[ic+"_par"] = (ptypes, names, npars, pars)
        # keep FF name
        data["FF"] = self.par.FF
        return data

    def unpack(self, data):
        """
        unpack data using exactly the structure produced in pack
        """
        ric_type = self.interaction_types
        ric      = {}
        for r in ric_type:
            rlist = []
            if r in data:
                rdata = data[r]
                nric = rdata.shape[0]                
                rlen  = rdata.shape[1]
                if r == "dih" or r == "vdw":
                    rlen -= 1 # in dih the ring attribute and for vdw the molid is stored as an additional column
                for i in range(nric):
                    rtype = rdata[i,0]
                    aind  = rdata[i,1:rlen]
                    if r == "dih":
                        if rdata[i,-1] != 0:
                            icl = ic(aind, type=rtype, ring=rdata[i,-1])
                        else:
                            icl = ic(aind, type=rtype)
                    elif r == "vdw":
                        if rdata[i,-1] >= 0:
                            icl = ic(aind, type=rtype, molid=rdata[i,-1])
                        else:
                            icl = ic(aind, type=rtype)
                    else:
                        icl = ic(aind, type=rtype)
                    rlist.append(icl)
            ric[r] = rlist    
        # now add data to ric object .. it gets only bnd, angl, oop, dih
        self.ric.set_rics(ric["bnd"], ric["ang"], ric["oop"], ric["dih"])
        # time to init the data structures .. supply vdw and cha here
        self._init_data(cha=ric["cha"], vdw=ric["vdw"])
        # check if molid has been unpacked from mfp5 ff data and set to molid in the parent mol object
        # NOTE: the ic is a list with attributes that never return an error .. non exisiting attributes are returned as None
        rt_vdw = self.ric_type["vdw"]
        if rt_vdw[0].molid is not None:
            self._mol.molid = np.array([x.molid for x in rt_vdw])
        # self._init_pardata() ## RS part of __init__ now
        # now do par part
        self.par.FF = data["FF"]
        for r in ric_type + ["vdwpr", "chapr"]:
            par = self.par[r]
            if r+"_par" in data:
                ptypes, names, npars, pars = data[r+"_par"]
                t2ident = {} # maps integer type to identifier
                ntypes = len(ptypes)
                for i in range(ntypes):
                    itype = npars[i,1]
                    ptype = ptypes[i].decode("utf-8")
                    ident = names[i].decode("utf-8")
                    param = list(pars[i,0:npars[i,0]]) # npars[i,0] is the number of params
                    if ident in par:
                        logger.warning('Identifier %s already in par dictionary --> will be overwritten' % ident)
                        raise ValueError("Identifier %s appears twice" % ident)
                    par[ident] = (ptype, param)
                    if itype in t2ident:
                        t2ident[itype].append(ident)
                    else:
                        t2ident[itype] = [ident]
                # now all types are read in: set up the parind datastructure of the ric
                parind = self.parind[r]
                for i,ri in enumerate(self.ric_type[r]):
                    parind[i] = t2ident[ri.type]
        return


    def update_molid(self):
        """helper function to update molid attribute in the vdw ics after things have changed in the parent mol's molid
        """
        for i, at in enumerate(self.ric_type["vdw"]):
            at.molid = self._mol.molid[i]
        return

# """  
# #  commented out becasue of string problems wih py3.12  (2024 RS)

#     def to_latex(self, refsys, ic, datypes = None):
        
#         Method to publish parameters as latex tables. This method needs pandas installed.

#         Args:
#             refsys (str): reference system for which the parameters should
#                 be published.
#             ic (str): Internal coordinate for which the parameters should
#                 be published
#             datypes (dict, optional): Defaults to None. Dictionary to 
#                 convert atomtypes to more readable types.
        
#         Returns:
#             str: latex code as str
        
   
#         import pandas as pd
#         mdyn2kcal=143.88
#         units = {
#             "cha": {},
#             "vdw": {},
#             "bnd": {"$k_b [\si{\kcal\per\mole\per\AA\squared}]$":mdyn2kcal},
#             "bnd5": {"$k_b [\si{\kcal\per\mole\per\AA\squared}]$":mdyn2kcal},
#             "ang": {"$k_a [\si{\kcal\per\mole\per\radian\squared}]$":mdyn2kcal},
#             "ang5": {"$k_a [\si{\kcal\per\mole\per\radian\squared}]$"},
#             "dih": {},
#             "oop": {"$k_o [\si{\kcal\per\mole\per\radian\squared}]$":mdyn2kcal},
#         }
#         potentials = {"cha": {"gaussian": ["$q [e]$", "$sigma [\si{\AA}]$"]},
#                 "vdw": {"buck6d": ["\text{$r^0_{ij} [\si{\AA}]$}", "$\epsilon_{ij}$"]},
#                 "bnd": {"mm3": ["$k_b [\si{\kcal\per\mole\per\AA\squared}]$", "\text{$r_b^0 [\si{\AA}]$}"]},
#                 "bnd5": {"mm3": ["$k_b [\si{\kcal\per\mole\per\AA\squared}]$", "\text{$r_b^0 [\si{\AA}]$}"]},
#                 "ang": {"mm3": ["$k_a [\si{\kcal\per\mole\per\radian\squared}]$", "\text{$\phi_a^0 [\si{\degree}]$} "]},
#                 "ang5": {"mm3": ["$k_a [\si{\kcal\per\mole\per\radian\squared}]$", "\text{$\phi_a^0 [\si{\degree}]$} "]},
#                 "dih": {"cos3": ["\text{$V_1 [\si{\kcal\per\mole}]$}", "\text{$V_2 [\si{\kcal\per\mole}]$}", "\text{$V_3 [\si{\kcal\per\mole}]$}",], 
#                 "cos3": ["\text{$V_1 [\si{\kcal\per\mole}]$}", "\text{$V_2 [\si{\kcal\per\mole}]$}", "\text{$V_3 [\si{\kcal\per\mole}]$}","\text{$V_4 [\si{\kcal\per\mole}]$}"]},
#                 "oop": {"harm": ["$k_o [\si{\kcal\per\mole\per\radian\squared}]$", "\text{$\gamma^0$ [\si{\degree}]}"]}}
#         sortlist = ["types", "potential",]
#         columnformat = "lcc"
#         d = {"types":[], "potential":[]}
#         for i,l in potentials[ic].items():
#             for j in l: 
#                 d[j]=[]
#                 sortlist.append(j)
#                 columnformat+="S"
#         for k,v in self.par[ic].items():
#             pot, ref, atypes = self.split_parname(k)
#             if pot not in potentials[ic].keys():continue
#             if ref == refsys:
#                 if datypes is not None: atypes = list(map(lambda a: datypes[a],atypes))
#                 d["types"].append("-".join(atypes))
#                 d["potential"].append(pot)
#                 for i, p in enumerate(v[1]):
#                     ptype = potentials[ic][pot][i]
#                     if ptype in units[ic].keys(): p *= units[ic][ptype]
#                     d[potentials[ic][pot][i]].append(round(p,3))
#         # remove empty columns
#         rkeys = []
#         for k,v in d.items():
#             if len(v)==0: rkeys.append(k)
#         for k in rkeys: 
#             del d[k]
#             sortlist.remove(k)
#         df = pd.DataFrame(data=d)
#         df = df[sortlist]
#         return df.to_latex(escape = False, column_format=columnformat)
# """
        

    def assign_params_from_key(self, fname):
        """
        Method used to setup an ff addon based on the legacy key file format.
        
        Args:
            fname (str): filename of the key file
        """


        def numberify_list(l, formatcode):
            if len(formatcode) == 0:
                return l
            elif formatcode[0] == "*":
                if formatcode[1] == "i":
                    try:
                        l = list(map(string.atoi,l))
                    except ValueError:
                        print ("Error converting %s to integer!!" % str(l))
                        raise ValueError
                elif formatcode[1] == "f":
                    l = list(map(string.atoi,l))
                elif formatcode[1] == "s":
                    pass
                else:
                    raise ValueError("unknown formatcode %s" % formatcode)
            else:
                for i in range(len(formatcode)):
                    if formatcode[i] == "i":
                        try:
                            l[i] = string.atoi(l[i])
                        except ValueError:
                            print ("Error converting %s to integer!!" % str(l[i]))
                            raise ValueError
                    if formatcode[i] == "f":
                        try:
                            l[i] = string.atof(l[i])
                        except ValueError:
                            print ("Error converting %s to float!!" % str(l[i]))
                            raise ValueError
                        except IndexError:
                            print ("Formatcode: %s" % formatcode)
                            print ("Data      : %s" % str(l))
                            raise IndexError('Maybe is the last opbend parameter missing? (not sure)')
                    if formatcode[i] == "o":
                        # optional float .. need to check if l is long enough ... o must be at the end!!
                        if (len(l) > i) :
                            l[i] = string.atof(l[i])
                        else:
                            l.append(0.0)
            return l
        # idea is to read the stuff first into self.par dictionary and afterwards
        # assign them
        # available potential keywords, keyword not yet implemented here are 
        # commented out
        pterms = {
            #"atom"        : (1, "so") , \
            "vdw"         : (1, "ff", "vdw", "buck6d")    ,\
            "vdwpr"       : (2, "ff", "vdwpr", "buck6d")    ,\
            "bond"        : (2,"ffo", "bnd", "mm3")   ,\
            #"bondq"       : (2,"ffff")   ,\
            #"bond5"       : (2,"ffo")   ,\
            #"bond4"       : (2,"ff")   ,\
            "angle"       : (3, "ff", "ang", "mm3") ,\
            #"angleq"       : (3, "ffff") ,\
            #"angle5"      : (3, "ffoo") ,\
            #"angle4"      : (3, "ffoo") ,\
            "anglef"      : (3, "ffioo", "ang", "fourier")  ,\
            #"anglef-2"    : (3, "ff")   ,\
            "strbnd"      : (3, "fff", "ang", "strbnd") ,\
            "opbend"      : (4, "fo", "oop", "harm")   ,\
            "torsion"     : (4, "fffo", "dih", "cos3") ,\
            #"torsion5"    : (4, "fffo") ,\
            #"torsion4"    : (4, "fffo") ,\
            "charge"      : (1, "ff", "cha", "gaussian") ,\
            #"chargemod"   : (2, "f") ,\
            #"chargeadd"   : (1, "f") ,\
            #"molname"     : (1, "*s") ,\
            #"virtbond"    : (2, "") ,\
            #"restrain-distance"  : (2, "ff"),\
            #"restrain-angle"     : (3, "ff"),\
            #"restrain-torsion"   : (4, "ff"),\
            }
        not_implemented = ["equivalent", "chargemod", "bond4", "bond5", 
            "angle5", "angle4", "restrain-distance","restrain-angle",
            "restrain-angle",]
        self._init_pardata()
        ptnames = list(pterms.keys())
        with open(fname, "r") as f:
            for line in f.readlines():
                sline = line.split()
                # jump over comments and empty lines
                if ((len(sline)>0) and (sline[0][0] != "#")):
                    keyword = sline[0]
                else:
                    continue
                # check for notes
                if sline.count("!!"):
                    sline = sline[:sline.index("!!")]
                # check if keyword is a potential keyword then read it in
                # ignore setttings at the first place
                if keyword in not_implemented:
                    raise NotImplementedError("%s not supported" % keyword)
                if ptnames.count(keyword):
                    natoms, formatcode, ic, pot = pterms[keyword]
                    # we have to get the atomkey and sort it
                    atomkey = aftype_sort([aftype(i,"leg") for i in sline[1:natoms+1]],ic)
                    # get actual params from numberify_list, stolen from pydlpoly.ff module
                    # we need to perform here some checks in order to distinct for example
                    # mm3 bnd from morse bnd potential
                    params = numberify_list(sline[natoms+1:], formatcode)
                    # we need to create a parname we use "leg" for legacy
                    # as name of the refsys and as framentnames to make it
                    # easy to use already implemented helper methods
                    # for the actual assignment
                    full_parname = pot+"->("+",".join(list(map(str,atomkey)))+")|leg"
                    if ic == "bnd" and params[2] > 0.0: pot = "morse"
                    if ic == "dih" and params[3] > 0.0: pot = "cos4"
                    self.par[ic][full_parname] = (pot, numberify_list(sline[natoms+1:], formatcode))
        # now we have to add the missing parameters for strbnd by looping over the bond and angle pots
        for k in self.par["ang"].keys():
            if self.par["ang"][k][0] == "strbnd":
                pot,ref, aftypes = self.split_parname(k)
                apot  = "mm3->"+k.split("->")[-1] 
                spot1 = self.build_parname("bnd", "mm3", "leg", aftypes[:2])
                spot2 = self.build_parname("bnd", "mm3", "leg", aftypes[1:])
                s1 = self.par["bnd"][spot1][1][1]
                s2 = self.par["bnd"][spot2][1][1]
                a  = self.par["ang"][apot][1][1]
                pot = ("strbnd", self.par["ang"][k][1]+[s1,s2,a])
                self.par["ang"][k]=pot                
        # now perform the actual assignment loop
        self.assign_params_offline(ref="leg", key = True)
        return




    
    def upload_params(self, FF, refname, dbrefname = None, azone = True, atfix = None, interactive = True):
        """
        Method to upload interactively the parameters to the already connected db.
        
        :Parameters:
            - FF(str): Name of the FF
            - refname(str): name of the refsystem for which params should be uploaded
            - dbrefname(str, optional): name of the refsystem in the db
            - azone(bool, optional): boolean flag indicating if an active zone entry in 
            the db should be created, defaults to True
            - atfix(list, optional): list of special atypes, which should be created 
            in the db for the reference system, defaults to None
            - interactive(bool, optional): specify if an interactive upload should be 
            done, defaults to True
        """
        assert type(refname) == str
        assert type(FF)      == str
        from mofplus import FF_api
        self.api = FF_api()
        if dbrefname is None: dbrefname = refname
        if atfix is not None:
            fixes = {}
            for i, at in enumerate(self._mol.atypes):
                if i in self.active_zone:
                    if at in atfix: fixes[str(i)]=at
        else: 
            fixes = None
        if azone:
            self.api.create_fit(FF, dbrefname, azone = self.active_zone, atfix = fixes)
        else:
            self.api.create_fit(FF, dbrefname)
        uploads = {
                "cha": {},
                "vdw": {}, 
                "bnd": {}, 
                "ang": {}, 
                "dih": {}, 
                "oop": {}}
        for ic,v in self.parind.items():
            for pl in v:
                for pn in pl: 
                    par = self.par[ic][pn]
                    pot, ref, aftypes = self.split_parname(pn)
                    if ref == refname:
                        if (tuple(aftypes), pot) not in uploads[ic].keys():
                            uploads[ic][(tuple(aftypes), pot)] = par[1]
        for ptype, upls in uploads.items():
            for desc, params in upls.items():
                # TODO: remove inconsitenz in db conserning charge and cha
                if ptype == "cha":
                    if interactive:
                        self.api.set_params_interactive(FF, desc[0], "charge", desc[1], dbrefname, params)
                    else:
                        self.api.set_params(FF, desc[0], "charge", desc[1], dbrefname, params)
                else:
                    if interactive:
                        self.api.set_params_interactive(FF, desc[0], ptype, desc[1], dbrefname, params)
                    else:
                        self.api.set_params(FF, desc[0], ptype, desc[1], dbrefname, params)
        return

