# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:26:02 CEST 2019

@author: rochus

                 expot _base

   base calss to derive external potentials for use with pylmps

   derive your class from this base class and overload the calc_energy_force and setup methods

   You can either set all params during the instantiation or within setup where you have access to the 
   mol object and its ff addon etc.
   Note that parameter fitting with ff_gen only works if you define params from the mol.ff object. 

"""

import numpy as np
from molsys import mpiobject
from lammps import lammps
from molsys.util.constants import kcalmol, electronvolt, bohr
from mpi4py import MPI
import time

from .xtb_calc import xtb_calc

class expot_base(mpiobject):

    def __init__(self, mpi_comm=None, out = None, local=False):
        super(expot_base, self).__init__(mpi_comm,out)
        self.name = "base"
        self.is_expot = True
        self.expot_time = 0.0
        # get xyz wrapped or unwrapped
        self.unwrapped = False
        # fix generates forces (this is the default but can be switched off)
        self.do_forces = True
        # if local use a local callback without any gather and scatter (need to get your stuff yourself)
        self.local = local
        if local:
            self.callback = self.callback_local
        else:
            self.callback = self.callback_full
        return

    def setup(self, pl):
        # keep access to pylmps object
        self.pl = pl
        # allocate arrays
        self.natoms = self.pl.get_natoms()
        self.energy = 0.0
        self.force  = np.zeros([self.natoms, 3], dtype="float64")
        self.step = 0
        return
    
    def calc_energy_force(self):
        """main class to be rewritten
        """
        # in base we do nothing ... in principle self.xyz exists
        return self.energy, self.force

    def calc_energy(self):
        # do nothing ... needs to be redefiend
        return self.energy

    def callback_full(self, lmps, vflag):
        """The callback function called by lammps .. should not be changed
        """
        tstart = time.time()
        lmps = lammps(ptr=lmps)
        # get the current atom positions
        self.xyz = self.pl.get_xyz(unwrapped=self.unwrapped)
        # get current cell
        self.cell = self.pl.get_cell()
        if self.do_forces:
            # calculate energy and force
            self.calc_energy_force()
            # distribute the forces back
            lmps.scatter_atoms("f", 2, 3, np.ctypeslib.as_ctypes(self.force))
        else:
            self.calc_energy()
        self.step += 1
        self.expot_time += time.time() - tstart 
        return self.energy

    def callback_local(self, lmps, com, fcom):
        """  callback function for a local fix

        """
        tstart = time.time()
        lmps = lammps(ptr=lmps) # not needed becasue we have access to it anyway
        self.calc_energy_force_local(com, fcom)
        self.step += 1
        self.expot_time += time.time() - tstart 
        return self.energy

    def callback_new(self,lmp, ntimestep, nlocal, tag, x, f):

        """The callback function called by lammps .. should not be changed
        """
        root = 0
        nprocs = self.mpi_comm.Get_size()

        # TODO Not really MPI parallel yet:
        # -> Forces need to be scattered to all slaves
        tstart = time.time()
        #
        # get the current atom positions
        # 
        #  -> gather atom positions from all nodes on master
        # 
        sendbuf_coord = np.ctypeslib.as_array(x)
        sendbuf_tags  = np.ctypeslib.as_array(tag)
        sendcounts_coord = np.array(self.mpi_comm.gather(sendbuf_coord.size, root))
        sendcounts2 = np.array(self.mpi_comm.gather(sendbuf_tags.size, root))
        if self.is_master:
            xyz  = np.empty(sum(sendcounts_coord), dtype=np.float64)
            tags = np.empty(sum(sendcounts2), dtype=np.int32)
        else:
            xyz  = None
            tags = None
        self.mpi_comm.Gatherv(sendbuf=sendbuf_coord, recvbuf=(xyz,  sendcounts_coord), root=root)
        self.mpi_comm.Gatherv(sendbuf=sendbuf_tags, recvbuf=(tags, sendcounts2), root=root)
        if self.is_master:
            # get current cell
            self.cell = self.pl.get_cell()
            self.xyz = np.ctypeslib.as_array(xyz)
            self.xyz.shape=(self.natoms,3)
            # Reorder according to tags
            idx = tags - 1
            self.xyz = self.xyz[idx]
            # calculate energy and force
            self.calc_energy_force()
            forces = np.ctypeslib.as_array(self.force)
            # prepare to scatter forces. Counts per worker should be the same as for the coordinates
            sendcounts_force = np.array(sendcounts_coord, copy=True)   
        else:
            forces = None
            sendcounts_force = np.zeros(nprocs, dtype=np.int)
        # for scatterv we need the counts on all workers
        self.mpi_comm.bcast(sendcounts_force, root=root)
        dspls = np.zeros(nprocs, dtype=np.int)
        dspls[0] = 0
        for i in range(1,nprocs):        
            dspls[i] = dspls[i-1] + sendcounts_force[i-1] 
        # scatter forces to nodes
        forces_local = np.zeros(nlocal*3,dtype=np.float64)
        #self.mpi_comm.Scatterv(sendbuf=forces, recvbuf=(forces_local, sendcounts_force), root=root)
        self.mpi_comm.Scatterv([forces, sendcounts_force, dspls, MPI.DOUBLE], forces_local, root=root) 
        forces_local.shape=(nlocal,3)
        # distribute the forces back
        for i in range(nlocal):
           f[i][0] = forces_local[i][0] 
           f[i][1] = forces_local[i][1] 
           f[i][2] = forces_local[i][2] 
        self.step += 1
        self.expot_time += time.time() - tstart 
        if self.is_master:
            # only master adds total energy to the fix
            lmp.fix_external_set_energy_global("expot_" + self.name, self.energy)
        return 

    def test_deriv(self, delta=0.0001, verbose=True):
        """test the analytic forces by numerical differentiation

        make shure to call the energy and force once before you call this function
        to make shure all is set up and self.xyz exisits and is initalized

        Args:
            delta (float, optional): shift of coords. Defaults to 0.0001.
            verbose (boolen, optiona): print while we go. Defaults to True

        """
        xyz_keep = self.xyz.copy()
        force_analytik = self.force.copy()
        force_numeric  = np.zeros(force_analytik.shape)
        for i in range(self.natoms):
            for j in range(3):
                self.xyz[i,j] += delta
                ep, f = self.calc_energy_force()
                self.xyz[i,j] -= 2.0*delta
                em, f = self.calc_energy_force()
                self.xyz[i,j] = xyz_keep[i,j]
                force_numeric[i,j] = -(ep-em)/(2.0*delta)
                if verbose:
                    print ("atom %5d dim %2d : analytik %10.5f numeric %10.5f (delta %8.2f)" % (i, j, force_analytik[i,j], force_numeric[i,j], force_analytik[i,j]-force_numeric[i,j]))
        return force_numeric


class expot_ase(expot_base):

    def __init__(self, atoms, idx):
        super(expot_ase, self).__init__()
        self.atoms = atoms
        self.idx = idx
        self.name = "ase"
        return

    def setup(self,pl):
        super(expot_ase, self).setup(pl)
        assert len(self.idx) <= self.natoms
        for i in self.idx:
            assert i < self.natoms
        self.pprint("An ASE external potential was added!")
        return

    def calc_energy_force(self):
        # we have to set the actual coordinates and cell to ASE
        self.atoms.set_cell(self.cell)
        self.atoms.set_positions(self.xyz[self.idx])
        # by default ase uses eV and A as units
        # consequently units has to be changed here to kcal/mol
        self.energy = self.atoms.get_potential_energy()*electronvolt/kcalmol
        self.force[self.idx] = self.atoms.get_forces()*electronvolt/kcalmol
        return self.energy, self.force

class expot_xtb(expot_base):

    def __init__(self, mol, periodic=None, gfn_param=0, etemp=300.0, accuracy=1.0, uhf=0, verbose=0, maxiter=250
                ,write_mfp5_file=False,write_frequency=100,mfp5file=None
                ,add_central_force=False,force_k=0.01,central_force_freq=500,restart=None,stage=None):
        """constructor for xTB external potential
        
        This will construct an external potential based on the xtb program to be used inside LAMMPS       
 
        Args:
            mol (molsys object): molecular system 
            periodic (boolean, optional): Sets if we have a periodic system. If this argument is set to None periodicity is extracted from the mol object
            gfn_param (int, optional): GFN parameterization. Possible values (-1,0,1,2) -1 is the GFN-FF force field
            etemp (float, optional): Electronic temperature. Defaults to 300 K
            accuracy (float, optional): accuracy setting. Defaults to 1.0 (default in xtb)
            uhf (int, optional): Number of unpaired electrons
            verbose (int, optional): Print level. Possible values 0,1,2. Defaults to 0 (silent mode)
            maxiter (int, optional): Maximum number of iterations in the self consisting charge cycles. Defaults to 250
            write_mfp5_file (boolean, optional): If a mfp5 should be written? Defaults to False 
            write_frequency (int, optional): If write_mfp5_file = True it determine how often the coordinates should be dumped
            mfp5file (mfp5file object, optional): The mfp5 file handeler used
            restart (boolean, optional): Determines if this is a restart
            stage (string, optional): Determines the current stage we are in 

        """
        super(expot_xtb, self).__init__()
        self.mol = mol
        self.gfn_param = gfn_param
        self.uhf = uhf
        self.etemp = etemp
        self.accuracy = accuracy
        self.verbose = verbose
        self.maxiter = maxiter     
        if periodic == None:
            self.periodic = mol.periodic
        else:
            self.periodic = periodic
        self.name = "xtb"
        self.bond_order = None
        self.write_mfp5_file = write_mfp5_file
        self.write_frequency = write_frequency
        self.mfp5file = mfp5file
        self.restart = restart
        self.stage = stage
        self.add_central_force = add_central_force
        self.central_force_freq = central_force_freq
        self.force_k = force_k
        return

    def setup(self,pl):
        super(expot_xtb, self).setup(pl)
        # create calculator and do gfn-xTB calculation
        self.gfn = xtb_calc( self.mol
                      , self.gfn_param
                      , pbc=self.periodic
                      , uhf=self.uhf
                      , accuracy=self.accuracy
                      , etemp=self.etemp
                      , verbose=self.verbose
                      , maxiter=self.maxiter
                      , write_mfp5_file=self.write_mfp5_file
                      , write_frequency=self.write_frequency
                      , mfp5file=self.mfp5file
                      , restart=self.restart
                      , stage=self.stage
                      , add_central_force=self.add_central_force
                      , force_k=self.force_k
                      , central_force_freq=self.central_force_freq
                      )
        self.pprint("An xTB external potential was added")
        print("Periodic Boundary Conditions:") 
        print(self.periodic)
        return

    def calc_energy_force(self):
        results = self.gfn.calculate(self.xyz, self.cell)
        #
        # xTB uses a.u. as units so we need to convert
        #
        self.energy  = results['energy'] / kcalmol
        self.force   = -results['gradient'] / kcalmol / bohr
        if self.gfn_param >= 0:
            self.bond_order = results['bondorder']
        return self.energy, self.force

    def get_bond_order(self):
        assert self.gfn_param >= 0, "No bond orders for GFN-FF (param = -1)"
        results = self.gfn.calculate(self.xyz, self.cell)
        self.bond_order = results['bondorder']
        return self.bond_order


    def set_stage(self,stage):
        self.gfn.set_stage(stage)
        return

class expot_central_force(expot_base):

    def __init__(self, mol, periodic=None, force_k=0.01,central_force_freq=500):
        """constructor for an artifical central force potential
        
        This will construct an external potential which applies a force central to the box         
 
        Args:
            mol (molsys object): molecular system 

        """
        super(expot_central_force, self).__init__()
        self.mol = mol
        if periodic == None:
            self.periodic = mol.periodic
        else:
            self.periodic = periodic
        self.central_force_counter = 0
        self.name = "central"
        self.add_central_force = add_central_force
        self.central_force_freq = central_force_freq
        self.force_k = force_k
        return

    def setup(self,pl):
        super(expot_central_force, self).setup(pl)
        self.pprint("A central force potential was added")
        print("Periodic Boundary Conditions:") 
        print(self.periodic)
        return

    def calc_energy_force(self):
        self.energy  = 0.0 
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
        self.force = force
        return self.energy, self.force

"""
   As an illustration this is a derived external potential that just fixes one interatomic distance by a harmonic potential
  
"""

class expot_test(expot_base):

    def __init__(self, a1, a2, r0, k):
        """a test external potential
        
        Args:
            a1 (int): atom1
            a2 (int): atom2
            r0 (float): refdist
            k (float): force constant
        """
        super(expot_test, self).__init__()
        self.name = "test"
        self.a1 = a1
        self.a2 = a2
        self.r0 = r0
        self.k  = k
        return

    def setup(self, pl):
        super(expot_test, self).setup(pl)
        # check if a1 and a2 are in range
        assert self.a1 < self.natoms
        assert self.a2 < self.natoms
        self.pprint("a test external potential between atoms %d (%s) and %d (%s)" % (self.a1, self.pl.mol.elems[self.a1], self.a2, self.pl.mol.elems[self.a2]))
        return

    def calc_energy_force(self):
        d = self.xyz[self.a1]-self.xyz[self.a2]
        r = np.sqrt(np.sum(d*d))
        rr = r-self.r0
        self.energy = 0.5*self.k*rr*rr
        dE = self.k*rr
        self.force[self.a1] = -d/r*dE
        self.force[self.a2] = +d/r*dE
        return self.energy, self.force


