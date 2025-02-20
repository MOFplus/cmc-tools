# -*- coding: utf-8 -*-
from __future__ import absolute_import

import molsys.util.elems as elements
import molsys.util.rotations as rotations
import itertools
import copy
import numpy as np
from molsys.util.elems import vdw_radii as vdw



import logging
logger = logging.getLogger("molsys.bb")

organic_elements = ["h", "b", "c", "n", "o", "f", "si", "p", "s", "cl", "as", "se", "br", "i"]

class bb:

    def __init__(self,mol):
        """        '''
        initialize data structure for mol object so it can handle building block files

        Args:
            mol (molsys.mol): mol instance where this addon is added to
        """
        self.mol = mol
        #assert self.mol.bcond == 0, "BBs must be non-periodic. mol object has bcond=%d" % self.mol.bcond ### [RA] what about rods?
        self.connector = []            # list of connector atoms (can be dummies) used for orientation (TBI: COM of multiple atoms)
        self.connector_atoms = []       # list of lists: atoms that actually bond to the other BB
        self.connector_types = []       # list of integers: type of a connector (TBI: connectors_atype should contain the atype of the OTHER atom bonded to .. same layout as connector_atoms)
        self.connector_dummies = []       # list of integers: index of connectors that are dummies
        self.connector_atypes = None
        self.center_type = None
        self.mol.is_bb = True
        return

    def __mildcopy__(self, memo):
        """self.mol instance is kept the same, the rest is deepcopied
        __mildcopy__ is meant as an auxiliary method of mol.__deepcopy__
        to prevent recursion error. Deepcopying a bb instance works
        as usual because the bb.mol.__deepcopy__ stops the recursion
        with bb.mol.bb.__mildcopy__"""
        try: #python3
            newone = type(self)(self.mol.__class__())
        except: #python2
            newone = type(self)(bb, None)
        newdict = newone.__dict__
        newdict.update(self.__dict__)
        for key, val in newdict.items():
            if key != "mol":
                newdict[copy.deepcopy(key, memo)] = copy.deepcopy(val, memo)
        return newone

    def setup(self, connector,
                    connector_atoms=None,
                    connector_types= None,
                    connector_atypes=None,
                    center_type="coc",
                    center_xyz=None,
                    name=None,
                    rotate_on_z=False,
                    align_to_pax=False,
                    pax_use_connonly=False):
        """setup the BB data (only use this method to setup BBs!)

        TBI: possibility to have a sequence of indices for a connector -> set a flag
        
        Args:
            connector (list of ints): list of connector atoms
            connector_atoms (list of lists of ints, optional): actual atoms bonding. Defaults to None.
            connector_types (list of ints, optional): if present then special connectors exist. Defaults to None.
            connector_atypes (list of lists of strings, optional): atomtype of the atom to which the connector is bonded to
                                                 same layout as connector_atoms, if present then _types are generted. Defaults to None.
            center_type (string, optional): either "com" or "coc" or "special". if "special" center_xyz must be given. Defaults to "coc".
            center_xyz (numpy array, optional): coordinates of the center. Defaults to None.
            name (string, optional) : The name of the BB
        """
        assert not (rotate_on_z and align_to_pax) 
        self.connector = connector
        nc = len(connector)
        if connector_atoms is not None:
            assert len(connector_atoms)==nc
            self.connector_atoms = connector_atoms
        else:
            self.connector_atoms = [[i] for i in self.connector]
        if connector_types is not None:
            assert len(connector_types)==nc
            self.connector_types = connector_types
        else:
            if connector_atypes is not None:
                # if atypes are given then no types should be given .. this is determined
                assert len(connector_atypes)==nc
                for a,at in zip(self.connector_atoms,connector_atypes):
                    assert len(a)==len(at)
                self.connector_atypes = connector_atypes
                self.connector_types = []
                known_atypes = []
                for at in self.connector_atypes:
                    if at not in known_atypes:
                        known_atypes.append(at)
                    self.connector_types.append(known_atypes.index(at))
            else:
                self.connector_types = [0 for i in range(nc)]
        # detect connector dummies by element 'x'
        for i,cats in enumerate(self.connector_atoms):
            for j,cat in enumerate(cats):
                if self.mol.elems[cat-1].lower() == 'x':
                    self.connector_dummies.append(cat-1)
        assert center_type in ["com", "coc", "special"]
        self.center_type = center_type
        if self.center_type == "special":
            assert center_xyz is not None
            assert center_xyz.shape == (3,)
        elif self.center_type == "com":
            self.mol.set_real_mass()
            center_xyz = self.mol.get_com()
        else: # the coc case          
            center_xyz = self.mol.get_coc(idx=self.connector)
        self.mol.translate(-center_xyz)
        if name is not None:
            self.name = name
        if rotate_on_z:
            self.rotate_on_z()
        if align_to_pax:
            self.align_pax_to_xyz(use_connxyz=pax_use_connonly)
        # finally sort for rising type in order to allow proper wrting to mfpx
        self.sort_connector_type()
        return

    def setup_with_bb_info(self,conn_identifier = 'He',center_point='coc'):
        """ Converts a mol object with a given atom as conn_identifier label to a BB.
        It can currently only work with a single conn_identifier, which means, that no special_connectors can be defined in this way.        
        Args:
            conn_identifier (str, optional): Atom type of the Atom used as connector label The connected atoms of the conn_identified atom will become the new connectors. Defaults to 'He'.
            center_point (str, optional): Definition of which center to use in the BB. Defaults to 'coc'.
        """
        # get indices of atoms conencted to conn_identifier
        cident_idx = [i for i,e in enumerate(self.mol.elems) if e.lower() == conn_identifier.lower()]
        logger.debug(cident_idx)
        connector = []
        for i,c in enumerate(self.mol.conn):
            for j,ci in enumerate(c):
                if cident_idx.count(ci) != 0: # atom i is connected to to an identifier_atom
                    connector.append(i)
        logger.debug('connectors',self.mol.connectors)
        # remove identifier atoms
        for ci,c in enumerate(connector):
            offset = [True for cidx in cident_idx if cidx < c].count(True)
            connector[ci] -= offset
        self.mol.delete_atoms(cident_idx)
        self.setup(connector)
        return
    
    @property
    def connector_xyz(self):
        """ get the xyz coords of the connector atoms
        TBI: if a connector is two atoms compute the center of mass """
        return self.mol.get_xyz(idx=self.connector)

    @property
    def connector_dist(self):
        """ get the distance from the center (origin 0,0,0) to the connector positions
        """
        return np.linalg.norm(self.connector_xyz, axis=1)

    def rotate_on_z(self):
        """ especially if this is a linker (2 connectors) we want it to lie on the z-axis
        we always use the first connector (could also be a regular SBU!) to be on the z-axis """
        c1_xyz = self.mol.xyz[self.connectors[0]]
        z_axis = np.array([0.0,0.0,1.0],"d")
        theta = rotations.angle(z_axis,c1_xyz) # angle to rotate
        if (theta > 1.0e-10) and (theta < (np.pi-1.0e-10)):
            axis  = rotations.normalize(rotations.cross_prod(z_axis,c1_xyz)) # axis around which we rotate
            self.mol.xyz = rotations.rotate(self.mol.xyz, axis, -theta)
        return

    def align_pax_to_xyz(self,use_connxyz = False):
        """ Aligns the coordinates to match the three principal axes of the intertial tensor to the cartesian coordinate system.
        Does not yet use weights
        
        Args:
            use_connxyz (bool, optional): If True, use only the coordinates of the connectors. Defaults to False.
        """
        ### TODO has been tested for m-bdc and some others to work, test for others aswell! 
        if use_connxyz == False:
            xyz = self.mol.get_xyz()
            self.mol.set_xyz(rotations.align_pax(xyz))
            return
        xyz = self.connector_xyz
        eigval,eigvec = rotations.pax(xyz)
        eigorder = np.argsort(eigval)
        rotmat = eigvec[:,eigorder] #  sort the column vectors in the order of the eigenvalues to have largest on x, second largest on y, ... 
        self.mol.set_xyz(rotations.apply_mat(rotmat,self.mol.get_xyz()))
        return

    def sort_connector_type(self):
        """This internal method sorts the connectors according to a rsing type

        this is necessary for writing mfpx files, where the different types are seperated by a slash
        """
        order = np.argsort(self.connector_types)
        self.connector      = [self.connector[i] for i in order]
        self.connector_atoms = [self.connector_atoms[i] for i in order]
        self.connector_types = [self.connector_types[i] for i in order]
        if self.connector_atypes is not None:
            self.connector_atypes = [self.connector_atypes[i] for i in order]
        return

    def remove_carboxylates(self,co2mol=None,remove_carboxy_H=True):
        """this function is meant to be used to convert a molecule
            which is not yet a bb type to a bb type by removing 
            carboxylates and carboxylic acid groups and adding 
            connectors where the carboxylates were 
        
        Keyword Arguments:
            co2mol {molsys.mol} -- co2 mol object (default: {None})
            remove_carboxy_H {bool} -- if true, carboxylic hydrogen atoms are removed (default: {True})
        
        Returns:
            molsys.mol -- the mol object with removed carboxylates
        """        
        from molsys.util.atomtyper import atomtyper 
        import molsys
        # find carboxylates
        m = copy.copy(self.mol)
        if remove_carboxy_H is True:
            Hidxs = [i for i,x in enumerate(m.atypes) if ((x == 'h1_o1') and (m.elems[m.conn[i][0]][0:2]=='c3'))]
            m.delete_atoms(Hidxs)
            at = atomtyper(m); at()
        m = copy.copy(self.mol)
        at = atomtyper(m); at()        
        if co2mol is None:
            co2mfpxtxt = '# type xyz\n3\n  1 c      0.000000    0.000000    0.000000   c3_*1o2                 -1                  -1       2       3  \n  2 o      1.500000    0.000000    0.000000   o1_c1                   -1                  -1       1  \n  3 o     -1.500000    0.000000    0.000000   o1_c1                   -1                  -1       1  \n'
            co2mol = molsys.mol.from_string(co2mfpxtxt)
            co2mol.addon('graph')
            co2mol.graph.make_graph(hashes=False)
        m.addon('graph')
        m.graph.make_graph(hashes=False)
        idxs = m.graph.find_sub(co2mol.graph)
        carboxy_atoms = sorted([item for sublist in idxs for item in sublist])
        other_atoms = [x for x in range(m.natoms) if x not in carboxy_atoms]
        other_conn = [[x for xx in m.conn[x] if (carboxy_atoms.count(xx) != 0)] for x in other_atoms] # get atom indices connected to the carboxylates
        other_conn = sorted([item for sublist in other_conn for item in sublist]) # flatten list
        offset = [len(np.where(np.array(carboxy_atoms) <= x)[0]) for x in other_atoms]
        newmol = m.new_mol_by_index(other_atoms)
        newmol.set_real_mass()
        if not hasattr(newmol,'bb'): newmol.addon('bb')
        connector = [x-offset[other_atoms.index(x)] for x in other_conn]
        newmol.bb.setup(connector)
        del newmol.bb.mol
        newmol.bb.mol = newmol
        return newmol

    @property
    def is_organic(self):
        """
        Checks if the bb contains only organic elements
        """        
        elems = self.mol.elems
        num_organic_elements = np.sum([elems.count(organic_element) for organic_element in organic_elements])
        if num_organic_elements == self.mol.natoms:
            return True
        return False

    def contains_disorder(self, reject_covrad_fact=0.8):
        """        try to detect disordered parts of the structure based on heuristics:
            - unusual bond lengths
            - unusual coordination numbers
        
        Keyword Arguments:
            reject_covrad_fact {float} -- consider bonds smaller than sum of vdw * fact as disordered (default: {0.5})
        
        Returns:
            bool -- the answer to the question of the bb contains disorder
        """        
        from molsys.util.elems import cov_radii as cov
        m = self.mol
        for i in range(m.natoms):
            # carbon atoms should have <=4 bonds
            if m.elems[i] == 'c':
                if len(m.conn[i]) > 4:
                    return 'c conn'
            if m.elems[i] == 'h':
                if len(m.conn[i]) > 2:
                    return 'h conn'
        for i in range(m.natoms-1):
            for j in range(i+1,m.natoms):
                d,v,imgi = m.get_distvec(i,j)
                tgt_cov = cov[m.elems[i]] + cov[m.elems[j]]
                if d <= tgt_cov * reject_covrad_fact:
                    return 'cov fact'
        return False        


    ## RS Feb 2021 ##############################################
    #
    #   the follwoing methods a re used to generate smiles strings for BBs in order to generate MOF+IDs

    _first_conn_anumber = 87

    def _add_conatoms(self, nbb, connectors):
        """generate a mol object as a copy of the original BB with ficitious heavy atoms attached to the connectors

        Args:
            connectors (list of ints): list of connector atoms
        """
        natoms = nbb.get_natoms()
        # TBI check if any eleents are Fr or higher
        fbond = []
        tbond = []
        for i, c in enumerate(connectors):
            con_pos = nbb.get_xyz()[c]
            con_dist = np.linalg.norm(con_pos)
            new_pos = con_pos * (con_dist+1.0)/con_dist
            new_elem = elements.pse[self._first_conn_anumber+i] #  use Fr and higher to annotate connectors.
            nbb.add_atom(new_elem, "xx%i" % i, new_pos)
            fbond.append(c)
            tbond.append(natoms+i)
        nbb.add_bonds(fbond, tbond)
        return nbb

    def _remove_metal_bonds(self, nbb):
        del_bonds = []
        for i in range(nbb.get_natoms()):
            for j in nbb.conn[i]:
                if i < j:
                    if (nbb.elems[i] not in organic_elements) and (nbb.elems[j] not in organic_elements):
                        del_bonds.append((i,j))
        for b in del_bonds:
            nbb.delete_bond(b[0],b[1])
        return


    def get_BBsmile(self):
        import itertools
        # from tqdm import tqdm
        # we need the obabel addon
        all_smiles = []
        all_conn_seqs = []
        nconn = len(self.connector)
        assert nconn < 9
        for conn_seq in itertools.permutations(self.connector[1:]):
            nbb = self.mol.clone()
            if not self.is_organic:
                self._remove_metal_bonds(nbb)
            nbb = self._add_conatoms(nbb, [self.connector[0]]+list(conn_seq))
            nbb.addon("obabel")
            smi = nbb.obabel.cansmiles
            if smi not in all_smiles:
                all_smiles.append(smi)
                all_conn_seqs.append(conn_seq)
        seq = np.argsort(all_smiles)
        final_smi = all_smiles[seq[0]]
        final_seq = [self.connector[0]]+list(all_conn_seqs[seq[0]])
        for i in range(nconn):
            final_smi = final_smi.replace("[%s]" % elements.pse[self._first_conn_anumber+i].title(), "[X%s]" % (i+1))
        return final_smi, [self.connector.index(a) for a in final_seq]

    ## RS May 2022 ##############################################
    #
    #   the follwoing methods are used to permute/rotate BBs in order to generate MOF+IDs

    
    def get_conpermute(self, w, thresh=0.01, threshd=0.05):
        """
        this method checks if the requested permutation of the connectors w
        with v2 <- v1[w] is possible, and if so returns the rotation matrix

        Note: apply rotation matrix like this xyz@R.T 

        We then apply the rotation to the actual atomic coordiantes and 
        check if this maintans the atomistic structure

        TBI this works currently only without specific connectors
        (could be added by additional logic to prevent mapping conns of unequal type)
        """
        # v1 should be unit vectors
        v1 = self.connector_xyz/self.connector_dist[:,None]
        n = len(w)
        assert len(v1) == n
        # assert len(v1) > 3
        v2 = v1[w]
        # first check if this permutation conserves all angles (maintains shape)
        for i,j in itertools.combinations(range(n), 2):
            dp1 = np.dot(v1[i], v1[j])
            dp2 = np.dot(v2[i], v2[j])
            if np.abs(dp1 - dp2) > thresh:
                return None, False
        # compute rotation matrix for first two vectors that are not aligned
        for i,j in itertools.combinations(range(n), 2):
            dp1 = np.dot(v1[i], v1[j])
            if np.abs(dp1) < 0.95: # maybe even larger value is ok here, but connectors should have a sufficient angle in most cases
                break
        # take i and j as the pair to do the rotmat from .. we generate a vector perpendicular
        v1c = np.cross(v1[i], v1[j])
        v2c = np.cross(v2[i], v2[j])
        abc1 = np.column_stack((v1[i], v1[j], v1c))
        abc2 = np.column_stack((v2[i], v2[j], v2c))
        R = abc2 @ np.linalg.inv(abc1)
        # sanity test: is the determinant +1? (if -1 this is a mirror ... not shape conserving)
        rdet = np.linalg.det(R)
        assert np.allclose(rdet, 1.0)
        # now apply the rotation matrix R to v2 -> does it rotate on v1?
        # if not it is not a valid permutation
        rot_v1 = (R @ v1.T).T
        if not np.allclose(rot_v1, v2, atol=thresh):
            return None, False
        # at this point the rotation is allowed by the connectors and the rotation matrix is known
        rot_xyz = self.mol.xyz @ R.T
        # now check if this is the same structure
        match = True
        for i in range(self.mol.natoms):
            r = self.mol.xyz[i] - rot_xyz
            d = np.sqrt((r*r).sum(axis=1))
            short_i = np.argmin(d)
            short_d = d[short_i]
            if short_d > threshd:
                match = False
                break
            else:
                # verify of elements match
                if self.mol.elems[i] != self.mol.elems[short_i]:
                    match = False
                    break
        return R, match
