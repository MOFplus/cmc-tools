"""
This file implements a ric_fit class.
It is inheriting all features from RedIntCoords in ric.py
and adds features to load a respective reference structure and Hessian.
In addition, a weight matrix is held to evaluate various kinds of weighted mean
square deviations to be used as ingredients to fittness values
"""

import numpy as np
import copy

from base import base
from .ric_src import RedIntCoords


ricmapping = {"bnd": "str",
        "ang": "ibe",
        "oop": "obe",
        "dih": "tor"}


class ric(base,RedIntCoords):
    """
    class to compute redundant internal coordinates (ric)
    by using the inherited ric module.
    """
    
    def __init__(self, mol, npproj = True):
        """
        Init procedure, to setup the a ric instance.
        :Parameters:
            - mol(obj): mol class instance
            - npproj(bool): use numpy to do invert and project (default)
        """
        base.__init__(self, mol)
        ### check if ff is already initialized, else do
        if hasattr(self._mol,"ff") == False:
            self._mol.addon("ff")
            self._mol.ff.ric.find_rics()
        elif hasattr(self._mol.ff.ric, "bnd") == False:
            self._mol.ff.ric.find_rics()
        ### init RedIntCoords
        RedIntCoords.__init__(self)
        if npproj == True:
            self.invert_b_matrix = self.invert_b_matrix_np
            self.project_hessian = self.project_hessian_np
        self.val_dict = {"str": self.get_val_stretches,
                "ibe": self.get_val_in_bends,
                "obe": self.get_val_out_bends,
                "lbe": self.get_val_lin_bends,
                "tor": self.get_val_torsions,
                "eck": self.get_val_eckarts,
                "hes": self.get_ric_hessian}
        return

    def setup_rics(self, full = True, lindict={}):
        """
        Method to setup the rics for the given mol object.
        :Parameters:
            - full(bool): if full is equal to True the whole conn
            table is used to setup the rics, if False only those rics
            are used for which FF terms are used; defaults to True
            - lindict(dict): dictionary holding information about 
            the linear bends in the system. Keys are lists of 
            indices defining a linear angle and the value is the
            corresponding reference atom; defaults to {}
        """
        self._mol.ff.ric.compute_rics()
        ### bonds
        for i,r in enumerate(self._mol.ff.ric.bnd):
            if full or r.used:
                self.add_stretch(np.array(list(r))+1)
        ### angles
        for i,r in enumerate(self._mol.ff.ric.ang):
            if abs(r.value-180.0) < 2.0:
                r.lin = True                
            else:
                r.lin = False
            if full or r.used:
                if r.lin:
                    #pass
                    self.add_lin_bend_mod(r, lindict)
                else:
                    self.add_in_bend(np.array(list(r))+1)
        ### oop
        for i, r in enumerate(self._mol.ff.ric.oop):
            a,b,c,d = r
            if full or r.used:
                self.add_out_bend(np.array([a,b,c,d])+1)
                self.add_out_bend(np.array([a,c,b,d])+1)
                self.add_out_bend(np.array([a,d,c,b])+1)
        ### dihedrals
        # TODO so far no fragtors are available
        for i, r in enumerate(self._mol.ff.ric.dih):
            if full or r.used:
                self.add_torsion(np.array(list(r))+1)
        self.add_eckart()
        self._mol.set_real_mass()
        self.setup(masses = np.array(self._mol.get_mass()))
#        self.report_rics("rics.dat")
        return
    
    def add_lin_bend_mod(self, indices, lindict={}):
        """
        Method to add an linear bend
        :Parameters:
            - indices: indices of the linear bend
        """
        if tuple(indices) in lindict.keys():
            self.add_lin_bend(np.array(list(indices))+1, ref = lindict[tuple(indices)]+1)
            self.pprint("Using atom %s for lin bend %s as reference" % (lindict[tuple(indices)], indices))
        else:
            if len(self._mol.conn[indices[0]])>1:
                lref = copy.copy(self._mol.conn[indices[0]])
                lref.pop(lref.index(indices[1]))
                self.add_lin_bend(np.array(list(indices))+1, ref = lref[0]+1)
                self.pprint("Using atom %s for lin bend %s as reference" % (lref[0], indices))
            elif len(self._mol.conn[indices[2]])>1:
                lref = copy.copy(self._mol.conn[indices[2]])
                lref.pop(lref.index(indices[1]))
                self.add_lin_bend(np.array(list(indices))+1, ref = lref[0]+1)
                self.pprint("Using atom %s for lin bend %s as reference" % (lref[0], indices))
            else:
                raise ValueError("No reference atom found for linear bend %s" % indices)
        

    @property
    def first_str(self):
        return 0

    @property
    def first_ibe(self):
        return self.first_str + self.num_stretch

    @property
    def first_obe(self):
        return self.first_ibe + self.num_in_bend

    @property
    def first_tor(self):
        return self.first_obe + self.num_out_bend

    @property
    def first_lbe(self):
        return self.first_tor + self.num_torsion

    @property
    def all_rics(self):
        return {"str": self._stretches,
                "ibe": self._in_bends,
                "obe": self._out_bends,
                "tor": self.tor2dih(self._torsions),
                "lbe": self._lin_bends}

    @property
    def active_rics(self):
        all    = self.all_rics
        active = []
        for k in ["str", "ibe", "obe", "tor", "lbe"]:
            if len(all[k]) > 0: active.append(k)
        return active

    def map_ric(self, ric_type, ind, reversed = False):
        """
        Method to map a given ric to the corresponding indices in the internal
        data structures.
        :Parameters:
            - ric_type(str): the ric type, which hast to be either str, ibe, 
            obe, tor or lbe
            - ind(list): list of indices
        :Returns:
            - i(int): index in the local datastructure of the given ric_type
            - i_glob(int): index in the global datastructure of all rics
        """
        mapper = {"str": (self._stretches, self.first_str),
                "ibe": (self._in_bends, self.first_ibe),
                "obe": (self._out_bends, self.first_obe),
                "tor": (self._torsions, self.first_tor),
                "lbe": (self._lin_bends, self.first_lbe)
                }
        if (ric_type == "tor") and (len(ind) != 12): ind = self.dih2tor(ind)
        if mapper[ric_type][0].count(list(np.array(list(ind))+1)) != 0:
            i = mapper[ric_type][0].index(list(np.array(list(ind))+1))
            iglob = i + mapper[ric_type][1]
            return i, iglob
        elif reversed == False and ric_type in ["str", "ibe", "tor", "lbe"]:
            return self.map_ric(ric_type, ind[::-1], reversed = True)
        else:
            return None, None
    
    def dih2tor(self,ind):
        """
        Method to translate a dihedral consisting of four atom indices
        to a torsion ric definition consisting of 12 indices
        :Parameters:
            - ind(list): list of indices of the dihedral
        :Returns:
            - tor(list): list of indices of the torsion
        """
        assert len(ind) == 4
        mapping = [0,5,6,7]
        new_ind = 12*[-1]
        for i, idx in enumerate(ind): new_ind[mapping[i]] = idx
        return new_ind

    def tor2dih(self, ind):
        """
        Method to translate a torsion ric definition to a usual dihedral definition.
        :Parameters:
            - ind(list): list of indices defining the dihedral
        :Returns:
            - dih(list): list of indices of the dihedral
        """
        if len(ind) == 0: return np.array([])
        if type(ind[0]) == int:
            return np.array(ind)[0,5,6,7].tolist()
        else:
            return np.array(ind)[:,[0,5,6,7]].tolist()

    def report_rics(self, file = None):
        """
        Method to report the RICs which has been setup.
        :Parameters:
            - file(str): if set the rics are dumped to the given filename;
            defaults to None
        """
        buffer = ""
        count = 0
        rics = self.all_rics
        for k in self.active_rics:
            for i,r in enumerate(rics[k]):
                buffer+= ("%4d %3d %5s " % (count,i,k)+len(r)*" %4d")  % tuple(r)+"\n"
                count += 1
        if file is None:
            print(buffer)
        else:
            with open(file, "w") as f:
                f.write(buffer)


    def invert_b_matrix_np(self):
        """
        Invert B-matrix

        The order of RIC is described in the *construct_b_matrix* documentation.

        Parameters
        ----------
        None

        Returns
        -------
        bmat_inv: (3*N,M) array
                  Inverse B-matrix, where N is the number of atom in the system and M is
                  the number of RICs.
        rank: int
              Rank of the inverse B-matrix. If rank is smaller than the number of
              Cartesian coordinates (3*N), the set of RICs is incomplete.

        Example
        -------

        >>> bmat_inv, rank = ric.invert_b_matrix()
        """

        # Invert B-matrix
        # TODO Add parameters controlling SVD convergence to avoid numerical
        # instabilities.
        self._ric.bmat_inv = np.linalg.pinv(self._ric.bmat).T 
#        rank, stat = _ric_build.bmat_invert(self._ric.bmat, self._ric.bmat_inv)
    #    stat = self._ric.bmat_invert()
#        assert stat == 0

    #    return self._ric.bmat_inv, self._ric.rank
        return self._ric.bmat_inv

    def project_hessian_np(self, cart_hessian):
        """
        Project Hessian matrix from Cartesian coordinates to RICs.

        Parameters
        ----------
        cart_hessian: (3*N, 3*N) array
                      A symmetric Hessian matrix in Cartesian coordinates, where N
                      is the number of atoms in the system.

        Returns
        -------
        ric_hessian: (M, M) array
                     A symmetric Hessian matrix in RICs, where M is the number of
                     RICs.

        Example
        -------

        >>> ric_hess = ric.project_hessian(cart_hess)
        """

        # Check and assign Hessian matrix (Cartesian)
        assert isinstance(cart_hessian,np.ndarray)
        assert cart_hessian.dtype == np.float64
        assert cart_hessian.shape == self._ric.cart_hessian.shape
        #assert np.all(cart_hessian == np.transpose(cart_hessian)) # Slow!!!
        self._ric.cart_hessian[:,:] = cart_hessian
        # Project Cartesian Hessian to RICs
    #    import pdb; pdb.set_trace()
        self._ric.ric_hessian = np.dot(self._ric.bmat_inv.T,
                np.dot(self._ric.cart_hessian, self._ric.bmat_inv)) 
        return self._ric.ric_hessian

