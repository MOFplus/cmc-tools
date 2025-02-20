# -*- coding: utf-8 -*-
"""
           zmat

    addon to manipulate a molecular (!) zmat for a single molecule
    based on the chemcoord module

    this is a replacement of the original zmat.py (now zmat_old.py) which is kept
    for historic and sentimental reasons. 
    the main reason for the rewrite is that the api of chemcoord has substantially changed.

    Nov. 2021 RS RUB

"""

import chemcoord
import pandas
import numpy as np

class zmat:

    def __init__(self, mol):
        """zmat class

        This is a simple frontend for the chemcoord library (which needs to be
        installed including pandas for this to work)
        Usage: On instantiation or when you call the init() method, a chemcoord
        cartesian and zmat object will be generated from the mol objects xyz coords.
        Currently we have to rely that the z-mat is useful for your manipulations. 
        If this is not the case more specific things need to be implementend 
        (e.g. reorder z-matrix)

        You can then "label" an internal coordinate with a name (provided it is part of
        the z-matrix) by get_ic() and modify its value with set_ic().
        With the method update() the changed structure will be updated in the parent
        mol object
        
        Args:
            mol (molobject): parent mol object
        """
        self._mol = mol
        self.elems = [e.strip().capitalize() for e in self._mol.elems]
        self.elems = pandas.DataFrame(self.elems, columns=["atom"], dtype='str')
        self.init()
        self.ics = {}
        return

    def init(self, check_bonds=True):
        """(re)init the chemcoord classes

        this method copies the coordinates from the parent mol object
        to a cartesian object and its corresponding zmat

        this needs to be called any time the mol structure has been changed
        """
        mxyz= self._mol.get_xyz()
        xyz = pandas.DataFrame(self._mol.xyz, columns=['x','y','z'], dtype='float64')
        self.cart = chemcoord.Cartesian(pandas.concat([self.elems, xyz], axis=1))
        self.cart.index = range(1, self._mol.natoms+1)
        self.zmat = self.cart.get_zmat()
        if check_bonds:
            ccbonds = self.cart.get_bonds()
            for i in ccbonds:
                for j in ccbonds[i]:
                    assert (j-1) in self._mol.conn[i-1] 
        return

    def get_ic(self, name, atoms, ictype):
        """label an internal coordinate in the z-matrix with a name

        b, a and d are just single entries in the z-matrix.
        r is a rotation around a bond (all dihedrals around this central
        bond are altered with repect to the original structure)

        Args:
            name (string): name of the internal coordinate to ref it in set_ic()
            atoms (list of integers): atoms defining the IC
            ictype (string): type of the ic b, a, d, or r

        Returns:
            bool: False if the IC is not available in the z-mat.
        """
        assert ictype in ('b', 'a', 'd', 'r')
        if ictype == "b":
            i,j = atoms
            if self.zmat.loc[i, "b"] == j:
                self.ics[name] = (i, "bond")
                return True
            elif self.zmat.loc[j, "b"] == i:
                self.ics[name] = (j, "bond")
                return True
            else:
                pass
        elif ictype == "a":
            i,k,j = atoms
            if self.zmat.loc[i, "a"] == j:
                if self.zmat.loc[i, "b"] == k:
                    self.ics[name] = (i, "angle")
                    return True
            elif self.zmat.loc[j, "a"] == i:
                if self.zmat.loc[i, "b"] == k:
                    self.ics[name] = (i, "angle")
                    return True
            else:
                pass
        elif ictype == "d":
            i,k,l,j = atoms
            il = self.zmat.loc[i]
            jl = self.zmat.loc[j]
            if il["d"] == j:
                if il["b"] == k and il["a"] == l:
                    self.ics[name] = (i, "dihedral")
                    return True
            elif jl["d"] == i:
                if jl["b"] == l and jl["a"] == k:
                    self.ics[name] = (j, "dihedral")
                    return True
            else:
                pass
        else:
            # this is for a rotation around a central bond
            k, l = atoms
            zmlines = [] # collect all lines that contribute
            for i, bi in self.zmat.loc[:, ('b', 'a')].iterrows():
                if (bi["b"] == k and bi["a"] == l) or (bi["b"] == l and bi["a"] == k):
                    zmlines.append(i)
            if len(zmlines) > 0:
                # ok we found something .. go ahead
                refvals = []
                for i in zmlines:
                    refvals.append(self.zmat.loc[i, "dihedral"])               
                self.ics[name] = tuple([len(zmlines), "rot"] + zmlines +refvals)
                return True
        return False

    def set_ic(self, name, value):
        if self.ics[name][1] == "rot":
            # this is a rotation around a bond .. need to update all dihedrals
            par = self.ics[name]
            for ii in range(par[0]):
                i = par[2+ii]      # get the line index
                v = par[2+par[0]+ii]    # get the ref value
                self.zmat.safe_loc[i, "dihedral"] = v+value
        else:
            i,s = self.ics[name]
            self.zmat.safe_loc[i, s] = value
        return

    def update(self):
        """update the xyz coords in the parent mol
        """
        self.cart = self.zmat.get_cartesian()
        pxyz = self.cart.loc[:,['x', 'y', 'z']]
        xyz = pxyz.sort_index().to_numpy()
        self._mol.set_xyz(np.ascontiguousarray(xyz))
        return




    