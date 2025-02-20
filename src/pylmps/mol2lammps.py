"""

@author: vanessa


              xyz2lammps
              
class to be instantiated with an exisiting mol object and paramters already assinged
it will write a data and a lamps input file              

"""

import numpy as np
import string
import copy

import molsys
import molsys.util.elems as elements
from molsys.addon import base


class mol2lammps(base):
    
    def __init__(self, mol,setup_FF=True):
        """
        setup system and get parameter 
        
        :Parameters:
        
            - mol: mol object with ff addon and params assigned
        """
        self._mol = mol
        return
        
    def write_data(self, filename="tmp.data", centered = False):
        self.data_filename = filename
        f = open(filename, "w")
        natypes = 3
        
        # write header 
        header = "LAMMPS data file for mol object with MOF-FF params from www.mofplus.org\n\n"
        header += "%10d atoms\n"      % self._mol.get_natoms()
        header += "%10d atom types\n\n"     %natypes 
        xyz = self._mol.get_xyz()
        if self._mol.bcond == 0:
            # in the nonperiodic case center the molecule in the origin
            # JK: Lammps wants a box also in the non-periodic (free) case.
            self._mol.periodic=False
            self._mol.translate(-self._mol.get_com())
            cmax = xyz.max(axis=0)+10.0
            cmin = xyz.min(axis=0)-10.0
            tilts = (0.0,0.0,0.0)
        elif self._mol.bcond<2:
            # orthorombic/cubic bcondq
            cell = self._mol.get_cell()
            cmin = np.zeros([3])
            cmax = cell.diagonal()
            tilts = (0.0,0.0,0.0)
        else:
            # triclinic bcond
            cell = self._mol.get_cell()
            cmin = np.zeros([3])
            cmax = cell.diagonal()
            tilts = (cell[1,0], cell[2,0], cell[2,1])
        #bcond: sets boundary conditions. 2 for cubic and orthorombic systems, 3 for triclinic systems
        if centered == False:       #hack: here systems which are centered at the origin are used.
            cmin = (-cmax)        
        if self._mol.bcond >= 0:
        #theres presumably no use for this part, it produces a non cubic cell
            header += '%12.6f %12.6f  xlo xhi\n' % (cmin[0], cmax[0])
            header += '%12.6f %12.6f  ylo yhi\n' % (cmin[1], cmax[1])
            header += '%12.6f %12.6f  zlo zhi\n' % (cmin[2], cmax[2])
            '''
            if self._mol.bcond >= 0:
                alle = []
                for i in range(3):
                    alle.append(abs(cmin[i]))
                    alle.append(abs(cmax[i]))
                header += '%12.6f %12.6f  xlo xhi\n' % (-max(alle), max(alle))
                header += '%12.6f %12.6f  ylo yhi\n' % (-max(alle), max(alle))
                header += '%12.6f %12.6f  zlo zhi\n' % (-max(alle), max(alle))
            '''
        if self._mol.bcond > 2:
            header += '%12.6f %12.6f %12.6f  xy xz yz\n' % tilts
        # NOTE in lammps masses are mapped on atomtypes which indicate vdw interactions (pair potentials)
        #   => we do NOT use the masses set up in the mol object because of this mapping
        
        header += "0.000000     0.000000     0.000000  xy xz yz\n"
        header += "\nMasses\n\n"        
        
        header += "1 1.0080\n" 
        header += "2 12.0107\n" 
        header += "3 15.9994\n" 
        
        f.write(header)
        # write Atoms
        f.write("\nAtoms\n\n")
        chargesum = 0.0
        for i in range(self._mol.get_natoms()):
            vdwt  = ''
            if self._mol.atypes[i] == 'h':
                atype = 1
            elif self._mol.atypes[i] == 'c':
                atype = 2
            elif self._mol.atypes[i] == 'o':
                atype = 3
            chrg = 0.0
            x,y,z = xyz[i]
            #   ind  atype chrg x y z # comment
            f.write("%10d %5d  %10.5f %12.6f %12.6f %12.6f # %s\n" % (i+1, atype, chrg, x,y,z, vdwt))
        f.write("\n")
        f.close()
        return
    