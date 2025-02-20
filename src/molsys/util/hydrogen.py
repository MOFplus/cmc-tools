#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import numpy
import copy
from molsys.addon import base
import molsys.util.rotations as rot
class hydrogen(base):
    """
    Class offering methods to add hydrogens to experimental structures. Up to now
    only a method for sp2 carbons is implemented.
    """

    def add_sp2(self, idx):
        """
        Method to add a hydrogen to a sp2 carbon.

        :Parameters:
          - idx (int): index of the carbon atom where the  hydrogen should be added.
        """
        m = self._mol
        # get central atom
        ca = self._mol.xyz[idx]
        # check number of bonded partners
        assert len(self._mol.conn[idx]) == 2
        # get bonded atom pos, map them to the image of the first one
        #mapped = self._mol.map2image(self._mol.xyz[[idx, self._mol.conn[idx][0], self._mol.conn[idx][1]]])
        #vecs = mapped[1:]-ca
        d1,v1,imgi1 = m.get_distvec(idx,m.conn[idx][0])
        d2,v2,imgi2 = m.get_distvec(idx,m.conn[idx][1])
        vec = v1+v2
        nvec = -vec/np.linalg.norm(vec)
        # add hydrogen
        self._mol.add_atom('h','h', ca+1.098*nvec)
        self._mol.add_bonds(idx, self._mol.natoms - 1)
        return
    
    def add_sp3(self, idx):
        """
        Method to add a hydrogen to an sp3 carbon.

        :Parameters:
          - idx (int): index of the carbon atom where the  hydrogen should be added.
        """
        m = self._mol
        # get central atom
        ca = self._mol.xyz[idx]
        conn = self._mol.conn[idx]
        tangle = 109.5
        if len(conn)  == 1:
            # we have a terminal sp3 carbon here. add three hydrogens
            d,v,imgi = m.get_distvec(conn[0],idx) # v is the vector towards the terminal carbon atom
            vnorm = numpy.linalg.norm(v)
            #import pdb; pdb.set_trace() 
            rndvec = numpy.random.uniform(0,1,3)
            rndvec  /=  numpy.linalg.norm(rndvec)
            n = rot.cross_prod(v/vnorm,rndvec)
            n /= numpy.linalg.norm(n)
            vnew = rot.rotate_xyz_around_vector(v/vnorm*1.10,n,degrees=180-tangle) 
            vnew = vnew / numpy.linalg.norm(vnew) * 1.10
            newxyz1 = ca + vnew
            newxyz2 = ca + rot.rotate_xyz_around_vector(vnew,v/vnorm,degrees=tangle)
            newxyz3 = ca + rot.rotate_xyz_around_vector(vnew,v/vnorm,degrees=2*tangle)
            self._mol.add_atom('h','h', newxyz1)
            self._mol.add_bonds(idx, self._mol.natoms - 1)
            self._mol.add_atom('h','h', newxyz2)
            self._mol.add_bonds(idx, self._mol.natoms - 1)
            self._mol.add_atom('h','h', newxyz3)
            self._mol.add_bonds(idx, self._mol.natoms - 1)
        elif len(conn)==2:
            d1,v1,imgi1 = m.get_distvec(idx,conn[0])
            d2,v2,imgi2 = m.get_distvec(idx,conn[1])
            old = v2 / numpy.linalg.norm(v2) * 1.10
            newxyz1 = ca + rot.rotate_xyz_around_vector(old,v1,degrees=tangle) 
            newxyz2 = ca + rot.rotate_xyz_around_vector(old,v1,degrees=2*tangle) 
            self._mol.add_atom('h','h', newxyz1)
            self._mol.add_bonds(idx, self._mol.natoms - 1)
            self._mol.add_atom('h','h', newxyz2)
            self._mol.add_bonds(idx, self._mol.natoms - 1)
            
            
        else:
            print('TBI')

        return












