
#
# import ctypes for interface to xTB
#
import ctypes
from ctypes import c_int, c_double, c_bool

#
# import interface to xTB
#
try:
    import xtb
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_FULL, VERBOSITY_MINIMAL, VERBOSITY_MUTED 

except ImportError:
    print("ImportError: Impossible to load xTB")

from molsys import mpiobject

import numpy as np
import molsys
import sys

from molsys.util import mfp5io
import molsys.util.elems as elems

from molsys.util.constants import bohr

import math

#
# Class definition for xtb calcaulator
#
class xtb_calc(mpiobject):


   def __init__( self
               , mol
               , gfn: int = 0
               , charge: float = 0.0
               , pbc: bool = False
               , maxiter: int = 250
               , etemp: float = 300.0 
               , accuracy: float = 0.01
               , uhf: int = 0
               , verbose = 0
               , write_mfp5_file = False
               , write_frequency = 1
               , add_central_force = False
               , force_k = 0.01
               , central_force_freq = 500
               , mfp5file = None
               , restart=None
               , stage = None
               , mpi_comm=None
               , out = None
               ):

      super(xtb_calc, self).__init__(mpi_comm,out)
      #
      # Sanity check(s)
      #
      if gfn not in [-1,0,1,2]:
        raise NotImplementedError("Currently only gfn-FF, gfn0-xTB, gfn1-xTB and gfn2-xTP supported")

      if gfn in (-1, 2) and pbc == True:
        raise NotImplementedError("Currently PBC only supported for gfn0-xTB and gfn1-xTB")


      parameter_set = {  0 : Param.GFN0xTB
                      ,  1 : Param.GFN1xTB
                      ,  2 : Param.GFN2xTB
                      , -1 : Param.GFNFF
                      }
      #
      # Assign class attributes
      #
      self.gfn = gfn
      self.mol = mol
      self.pbc = pbc
      self.charge = charge
      self.param = parameter_set[gfn] 
      self.etemp = etemp
      self.maxiter = maxiter
      self.accuracy = accuracy
      self.uhf = uhf
      self.verbose = verbose

      # set up things for integration in MD driver
      self.nbondsmax = None
      self.write_mfp5_file = write_mfp5_file
      self.write_frequency = write_frequency
      self.write_counter = 1
      self.startup = True
      self.stage = stage
      self.use_own_writer = True
      # for adding central force periodically 
      self.add_central_force = add_central_force
      self.force_k = force_k
      self.central_force_freq = central_force_freq
      self.central_force_counter = 0
      if write_mfp5_file:
          self.mfp5 = mfp5io.mfp5io(mfp5file, ffe=self, restart=restart)
          # Get upper bound for bonds 
          if self.nbondsmax == None:
              self.nbondsmax = 0
              for e in self.get_elements():
                  self.nbondsmax += elems.maxbond[e]
              self.nbondsmax /= 2
          self.nbondsmax = int(self.nbondsmax)
      else:
          self.mfp5 = None
      # set up things that do not change during the calcultion
      self.numbers   = np.array(self.mol.get_elems_number(), dtype=c_int)
      self.calc = None
      return

   def get_natoms(self):
      return self.mol.get_natoms()

   def get_elements(self):
      return self.mol.get_elems()

   def get_xyz(self):
      return self.mol.get_xyz()

   def set_xyz(self, xyz):
      self.mol.set_xyz(xyz)
      return

   def get_cell(self):
      return self.mol.get_cell()

   def get_image(self):
      images = np.zeros((self.mol.natoms,3), dtype=np.int32)
      return images

   def get_bond_length(self,iat,jat):
      rij, rvec, closest = self.mol.get_distvec(iat, jat)
      return rij 

   def get_elements(self):
      return self.mol.get_elems()

   def set_stage(self,stage):
      self.stage = stage

   def activate_central_force(self):
      self.add_central_force = True

   def deactivate_central_force(self):
      self.add_central_force = False

   def calculate(self, xyz, cell):
      # update xyz in the mol object .. used for writing the frame later
      self.set_xyz(xyz)
      positions = np.array(xyz/bohr, dtype=c_double)
      if self.pbc == True:  
         cell      = np.array(cell/bohr, dtype=c_double)
         pbc       = np.full(3, True, dtype=c_bool)
      else:
         cell = None
         pbc       = np.full(3, False, dtype=c_bool)

      if self.calc is None:
         # make calc when it does not exist (first interation)
         self.calc = Calculator(self.param, self.numbers, positions, self.charge, uhf=self.uhf, lattice=cell, periodic=pbc)
         self.calc.set_verbosity(self.verbose)
         #
         # Set user options
         #
         self.calc.set_electronic_temperature(self.etemp)
         self.calc.set_max_iterations(self.maxiter)
         self.calc.set_accuracy(self.accuracy)
      else:
         # update calc (molecule obejct under the hood)
         try:
             self.calc.update(positions, lattice=cell)
         except:
             print("Problems updating coordinates:")
             print("poistions:")
             print(positions)
             print("Lattice:")
             print(cell)
             sys.exit()

      res = self.calc.singlepoint()

      results = { 'energy'    : res.get_energy()
                , 'gradient'  : res.get_gradient()
                # , 'bondorder' : res.get_bond_orders()
                }
      if self.gfn != -1:
         # for the non GFN-FF we can get bond orders
         results['bondorder'] = res.get_bond_orders()


      if self.add_central_force:
         self.central_force_counter += 1 
         if self.central_force_counter == self.central_force_freq:
             self.central_force_counter = 0
             force = np.zeros((self.get_natoms(),3))
             origin = [0.0,0.0,0.0]    
             #origin = self.mol.get_com()     
             self.mol.set_real_mass()      
             amass = self.mol.get_mass()
             for idx,(row,xyz) in enumerate(zip(force,self.get_xyz())):
                 row[0] = self.force_k * amass[idx] * (xyz[0] - origin[0])**2 * np.sign(xyz[0] - origin[0])
                 row[1] = self.force_k * amass[idx] * (xyz[1] - origin[1])**2 * np.sign(xyz[1] - origin[1])
                 row[2] = self.force_k * amass[idx] * (xyz[2] - origin[2])**2 * np.sign(xyz[2] - origin[2])
             results['gradient'] += force

      if self.write_mfp5_file and self.is_master:
         self.write_frame(results)

      return results

   def write_frame(self,results):

       def update_bond_order(wiberg_index,r,elemi,elemj):
           bohr = 0.529177
           # will be later moved somewhere else. Here we store the equilibirum bond info
           # Dictionary with element types e1_e2 (e1 and e2 in alphavetical order) 
           # followed by a list of tuples containing the tabulated data of Wiberg bond indices and equilibrium distances.
           # 1. we find the entry for e1_e2
           # 2. search list for tuple with the closet wiberg bond index (first entry) to given argument
           tabulated_data = { "c_h"  : [(0.99857405837699698,2.0723),(0.98377257130473217,2.0790) ] 
                            , "o_o"  : [(1.4999999999999820,2.7590)]
                            , "c_c"  : [(2.9979458557061029,2.2559),(2.0372552164634392,2.4876)]
                            , "c_o"  : [(1.0103530950900168,2.6576),(1.0042166955474088,2.6851), (1.8711324229829684,2.3046)]
                            , "h_o"  : [(0.92103774706803487,1.8350),(0.90664162691079231,1.8196)]
                            , "c_n"  : [(2.8038418306284827,2.2130),(1.1418104928922275,2.6377),(1.2686688800681707,2.5832) ]
                            , "h_n"  : [(0.94056613311717330,1.9257)]
                            , "h_h"  : [(1.0000000000000002,1.3984)]
                            , "na_o" : [(0.45407617747597667,3.7360)]
                            } 
           alpha = 0.3
           # decode bond info
           lstelem =  sorted([elemi,elemj], key=str.lower)
           identifier = str(lstelem[0]).lower() + "_" + str(lstelem[1]).lower()
           if identifier in tabulated_data:
               r_0 = 1.0
               dist = 999.99
               for data in tabulated_data[identifier]:
                   if abs(data[0] - wiberg_index) < dist:
                       r_0 = data[1] * bohr
                       dist = abs(data[0] - wiberg_index) 
           else:
               # default (probably not good in most cases)
               #print("WARNING: Could not find pre-tabulated data for bond order for identifier %s" % identifier) 
               r_0 = 1.0
           rij = r 
           bo = wiberg_index * math.exp(-abs(rij-r_0)/alpha) 
           return bo  
       #
       # Actual function body
       #
       if self.write_counter == self.write_frequency or self.startup:
           self.write_counter = 1  # Set back to zero
           self.startup = False
           self.mfp5.open()
           if self.stage in self.mfp5.h5file:
               st = self.mfp5.h5file[self.stage]
               traj = st["traj"]
               bond_order = results['bondorder']
               elems = self.get_elements()
               # Setup bondtab
               bond_tab = np.zeros((self.nbondsmax,2),dtype=int)
               bond_ord = np.full((self.nbondsmax),-1.0)
               bothres = 0.5
               natoms = self.get_natoms()
               nbnd = 0
               for iat in range(natoms):
                  for jat in range(0,iat+1):
                     # calculate bond length
                     bnd = self.get_bond_length(iat,jat)
                     bo = update_bond_order(bond_order[iat][jat],bnd,elems[iat],elems[jat])
                     #print("bond order %s %s %10.3f %10.3f" % (elems[iat],elems[jat],bo, bnd))
                     if bo > bothres:
                        bond_tab[nbnd][0] = iat+1 # We need the index starting from 1
                        bond_tab[nbnd][1] = jat+1 # We need the index starting from 1
                        bond_ord[nbnd] = bo
                        nbnd += 1
               if "bondord" not in traj:
                  # entry does not exist. Create it
                  self.mfp5.add_bondtab(self.stage, self.nbondsmax, bond_tab, bond_ord)
               # Add bondtab to mfp5 file
               traj_bondord = traj["bondord"]
               traj_bondord.resize((traj_bondord.shape[0] + 1),axis=0)
               traj_bondord[-1:] = bond_ord
               traj_bondtab = traj["bondtab"]
               traj_bondtab.resize((traj_bondtab.shape[0] + 1),axis=0)
               traj_bondtab[-1:] = bond_tab
               # Add trajectory info
               xyz_data = self.mol.get_xyz()
               #
               # add xyz data
               #
               # Lammps write the trajectory...
               if self.use_own_writer == True:
                  if "xyz" in traj:
                     trj_xyz = traj["xyz"]
                     trj_xyz.resize((trj_xyz.shape[0] + 1),axis = 0)
                     trj_xyz[-1:] = xyz_data 
                  else:
                     xyzshape = (1,) + xyz_data.shape
                     trj_xyz = traj.require_dataset("xyz",shape=xyzshape, dtype=xyz_data.dtype, maxshape=( (None,) + xyz_data.shape), chunks=xyzshape)
                     trj_xyz[...] = self.mol.get_xyz()
                  #velocities = False # TODO
                  # if velocities:
                  #    trj_vel = traj.require_dataset("vel", shape=vel.shape, dtype=vel.dtype)
                  #    #trj_vel[...] = vel #TODO
           self.mfp5.close() 
       else:
           self.write_counter += 1


