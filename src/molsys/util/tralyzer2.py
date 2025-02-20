""" 
               tralyzer 2
               
       A trajectory analyzer working on and processing mfp5 files
       This is the second incarnation of tralyzer using a different design 
       principle:
       instead of itslef opening the mfp5 file and handling everything it 
       expects an already opened mfp5 file on instantiation.
       This is in order to work with the mfp5 interactive script.
       The downside: currently parallelization is not included
       
       to be done: 
           - add parallel operations using the multiprocessing/shmem lib of python
           - convert ops to cython to speed up
           
       R. Schmid  (RUB 2014)
       
"""

from mpi4py import MPI
import numpy as np
import numpy.linalg as la
#import pdlpio
import copy
from molsys.util import elems as elements

import time

# using arcesios new elements.py for the full PSE
atomicmass = elements.mass
atomicnumbers = elements.number
a2b = 1.0/0.529177


class progressrep:
    
    def __init__(self, name, ntot, nevery):
        self.name = name
        self.ntot = ntot
        self.nevery = nevery
        self.count = 0
        self.repcount = 0
        self.tstart = time.time()
        self.tottime = 0.0
        return
        
    def start(self):
        self.count = 0
        self.repcount = 0
        self.tstart = time.time()
        self.tottime = 0.0
        return
        
    def __call__(self):
        self.count += 1
        if (self.count%self.nevery) == 0:
            self.repcount += 1
            t = time.time()
            dt = t-self.tstart
            self.tstart = t
            self.tottime += dt
            at = self.tottime/self.repcount
            eta = (self.ntot-self.count)/self.nevery*at
            print("Processing %6d: time since last report [s]: %6.2f (av:%6.2f) ETA [s]: %8.2f" % (self.count, dt, at, eta))
        return
        


class tralyzer2:

    def __init__(self, pdpf, stage, tstep=None):
        """ mfp5f  is already open
            keep MPI stuff from tralyzer .. in case we ever need it """
        self.comm = MPI.COMM_WORLD
        self.node_id = self.comm.Get_rank()
        self.nodes   = self.comm.Get_size()
        # open file
        self.pdpf = pdpf
        self.version = 1.0
        if "version" in list(self.pdpf.h5file.attrs.keys()):
            self.version = self.pdpf.h5file.attrs["version"]
        # JK: it seems that tralyzer has its own data structure, redundant with the new mol
        # data structure. I'll use a mol and only link the attributes.
        self.mol = self.pdpf.get_mol_from_system()
        self.elems = self.mol.elems
        self.types = self.mol.atypes
        self.boundarycond = self.mol.bcond
        ctab = self.mol.get_ctab()
        self.natoms = self.mol.natoms
        # recover connectivity from ctab
        self.cnct = self.mol.conn
        # register types
        self.typedata = {}
        for t in self.types:
            if t not in self.typedata: self.typedata[t] = None
        self.ntypes = len(list(self.typedata.keys()))
        # molecule info - note that only a nonredundant set of info is in mfp5 and the
        # rest needs to be recovered
        self.pdpf.setup_molecules(self.mol)
        self.whichmol = self.mol.molecules.whichmol
        self.moltypes = self.mol.molecules.moltypes
        self.molnames = self.mol.molecules.molnames
        self.nmols = len(self.moltypes) 
        self.mols = []
        for i in range(self.nmols) : self.mols.append([])
        for i, m in enumerate(self.whichmol): self.mols[m].append(i)
        # set stage
        if not self.pdpf.has_stage(stage):
            raise IOError("The pdp file has no stage %s" % stage)
        self.stagename = stage
        self.traj = self.pdpf.h5file[stage]["traj"]
        self.rest = self.pdpf.h5file[stage]["restart"]
        if ("tstep" in list(self.traj.attrs.keys())) and (tstep == None):
            timestep = self.traj.attrs["tstep"]
        elif tstep == None:
            timestep = 0.001
        else:
            timestep = tstep
        # for version > 1.0 allow different steps
        if self.version > 1.0:
            nstep = self.traj.attrs["nstep"]
            if not np.isscalar(nstep):
                nstep = nstep.max()
#            if "nstep" in list(self.traj["xyz"].attrs.keys()):
#                nstep = self.traj["xyz"].attrs["nstep"]
        self.dt = timestep*nstep
        # self.nframes = self.xyz.shape[0]
        self.nframes = self.traj["xyz"].shape[0]
        self.pprint("set active stage to %s" % stage)
        self.pprint("number of frames    %10d" % self.nframes)
        self.pprint("time between frames %10.5f ps" % self.dt)
        if self.boundarycond > 0:
            if not list(self.traj.keys()).count("cell"):
                self.pprint("using static cellparams from restart data")
                self.cell = np.array(self.rest["cell"])
                self.fixed_cell = True
            else:
                self.pprint("using trajectory cellparams per frame")
                self.fixed_cell = False
                self.cell = np.array(self.traj["cell"])
                if "nstep" in list(self.traj["cell"].attrs.keys()):
                    self.dt_cell = timestep*self.traj["cell"].attrs["nstep"]
                else:
                    self.dt_cell = self.dt
        if list(self.traj.keys()).count("vel"):
            self.pprint("velocities are available")
            self.vel = self.traj["vel"]
        else:
            self.vel = None
        self.pprint("\n TRALYZER2 started up")
        return
           
    def pprint(self, string):
        if self.node_id==0:
            print(string)
        return
        
    def get_mollist(self, molname):
        if type(molname) == type([]):
            mols = []
            for m in molname:
                moltype = self.molnames.index(m)
                for i,n in enumerate(self.moltypes):
                    if n==moltype: mols.append(i)
            return mols
        else:
            moltype = self.molnames.index(molname)
            return [i for i,n in enumerate(self.moltypes) if n==moltype]

                
             
    ####  cell stuff for trajectories of NsT runs
        
    def get_cell_averages(self, first_frame=0, last_frame=None, write_frames=None):
        """ computes averages over cell params and volume """
        if self.fixed_cell:
            self.pprint("This makes sense only for trajectories with varying cell")
            return
        cell = np.array(self.cell[first_frame:last_frame], dtype="float64")
        V       = np.sum(cell[:,0]*np.cross(cell[:,1],cell[:,2]), axis=1)
        mean_V  = np.mean(V)
        stdd_V  = np.std(V)
        cellpar = np.sqrt(np.sum(cell*cell, axis=2))
        mean_cp = np.mean(cellpar, axis=0)
        stdd_cp = np.std(cellpar, axis=0)
        alpha   = np.arccos(np.sum(cell[:,1]*cell[:,2], axis=1)/(cellpar[:,1]*cellpar[:,2]))*(180.0/np.pi)
        beta    = np.arccos(np.sum(cell[:,0]*cell[:,2], axis=1)/(cellpar[:,0]*cellpar[:,2]))*(180.0/np.pi)
        gamma   = np.arccos(np.sum(cell[:,0]*cell[:,1], axis=1)/(cellpar[:,0]*cellpar[:,1]))*(180.0/np.pi)
        mean_alp= np.mean(alpha)
        stdd_alp= np.std(alpha)
        mean_bet= np.mean(beta)
        stdd_bet= np.std(beta)
        mean_gam= np.mean(gamma)
        stdd_gam= np.std(gamma)
        self.pprint("Analysis of cellparameters")
        self.pprint("==========================")
        self.pprint("Volume:   %12.6f A**3  (std-dev %12.6f)" % (mean_V, stdd_V))
        self.pprint("a     :   %12.6f A     (std-dev %12.6f)" % (mean_cp[0], stdd_cp[0]))
        self.pprint("b     :   %12.6f A     (std-dev %12.6f)" % (mean_cp[1], stdd_cp[1]))
        self.pprint("c     :   %12.6f A     (std-dev %12.6f)" % (mean_cp[2], stdd_cp[2]))
        self.pprint("alpha :   %12.6f deg   (std-dev %12.6f)" % (mean_alp, stdd_alp))
        self.pprint("beta  :   %12.6f deg   (std-dev %12.6f)" % (mean_bet, stdd_bet))
        self.pprint("gamma :   %12.6f deg   (std-dev %12.6f)" % (mean_gam, stdd_gam))
        if write_frames:
            self.pprint("Writing results for each frame to file %s" % write_frames)
            of = open(write_frames, "w")
            for i in range(len(cell)):
                of.write("%5d %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n" %\
                         (i+1, cellpar[i,0], cellpar[i,1], cellpar[i,2], alpha[i], beta[i], gamma[i], V[i]))
            of.close()
        return
                
        
    def compute_elastic_constants(self, T, first_frame=0, last_frame=None, h0=None):
        """ Compute elastic constants by strain fluctuation .. code basis from FX Coudert """
        if self.fixed_cell:
            self.pprint("This makes sense only for trajectories with varying cell")
            return
        h = np.array(self.cell[first_frame:last_frame], dtype="float64")
        # reference cell h0 (inverted and transposed) ... could be the equal cell or anything else
        if not h0: 
            h0 = np.mean(self.cell, axis=0)
            self.pprint("Using averaged cellparams as reference")
            self.pprint(str(h0))
        h0m1  = la.inv(h0)
        #h0m1 = la.inv(h[0])
        h0m1t = h0m1.transpose()
        # compute strain (deviation from reference cell)
        eps_mat = np.array([(np.dot(h0m1t, np.dot(hi.transpose(), np.dot(hi, h0m1))) - np.identity(3))/2.0 for hi in h])
        # Voigt_map = [[0,0], [1,1], [2,2], [2, 1], [2, 0], [1, 0]]
        eps_voigt = np.empty([6,eps_mat.shape[0]], "d")
        # eps_voigt_std = np.empty([6,eps_mat.shape[0]], "d")
        eps_voigt[0] = eps_mat[:,0,0]
        eps_voigt[1] = eps_mat[:,1,1]
        eps_voigt[2] = eps_mat[:,2,2]
        eps_voigt[3] = eps_mat[:,2,1]
        eps_voigt[4] = eps_mat[:,2,0]
        eps_voigt[5] = eps_mat[:,1,0]
        eps_voigt_mean = np.mean(eps_voigt, axis=1)
        # eps_voigt_std = np.std(eps_voigt, axis=0)
        Smat = np.zeros([6,6], "d")
        # Smat_std = np.zeros([6,6], "d")
        for i in range(6):
            for j in range(i+1):
                fij = np.mean(eps_voigt[i]*eps_voigt[j])
                Smat[i,j] = fij-(eps_voigt_mean[i]*eps_voigt_mean[j])
                if i != j: Smat[j,i] = Smat[i,j]
                # fix this for error propagation!!
                # Smat_std[i,j] = np.std(eps_voigt[i]*eps_voigt[j], axis=1)-eps_voigt_std[i]*eps_voigt_std[j]
                # if i != j: Smat_std[j,i] = Smat_std[i,j]
        # finish it (use mean Volume and T)
        V = np.mean(np.sum(h[:,0]*np.cross(h[:,1],h[:,2]), axis=1))
        Smat *= (V*1.e-30)/(1.3806488e-23*T)
        # Cmat in GPa
        Cmat = la.inv(Smat)*1.0e-9
        self.pprint('\nStiffness matrix C (GPa):')
        self.pprint(np.array_str(Cmat, precision=2, suppress_small=True))
        # Eigenvalues
        self.pprint ('\nStiffness matrix eigenvalues (GPa):')
        self.pprint ((6*'% 8.2f') % tuple(np.sort(la.eigvals(Cmat))))
        return
        
    ####  COM stuff
        
    def compute_com(self, molname, store=False):
        mollist = self.get_mollist(molname)
        nmols = len(mollist)
        self.pprint("\ncomputing the COM of %d molecules with name %s" % (nmols,molname))
        self.pprint("Molecules are unwrapped .. this works only if the molecule is smaller then half the unit cell!!")
        atomlist = []
        for m in mollist:
            atomlist.append(self.mols[m])
        # since all molecules are equal we need to get masses for only one representative
        mass = []
        for a in atomlist[0]:
            mass.append(atomicmass[self.elems[a]])
        # make it the right shape to multiply with xyz coords
        mass = np.array(mass, dtype="float64")
        summass = np.sum(mass)
        mass = mass[:,np.newaxis]
        # allocate the array for the COMs
        # NOTE we could use the dtype of the xyz dataype here 
        xyz = self.traj["xyz"]
        com = np.zeros([self.nframes,nmols,3],dtype=xyz.dtype)
        if self.fixed_cell:
            celldiag = self.cell.diagonal()
            half_celldiag = celldiag/2.0
        #if self.nodes>1:
            #frames_per_node = self.nframes/self.nodes
            #wrapup          = self.nframes%self.nodes
            #first_frame = self.node_id*frames_per_node
            #last_frame  = first_frame+frames_per_node
            #if self.node_id == self.nodes-1:
                #last_frame += wrapup
        #else:
            #first_frame=0
            #last_frame =self.nframes
        report = progressrep("com", self.nframes, self.nframes/20)
        for f in range(self.node_id, self.nframes, self.nodes):
            if not self.fixed_cell:
                celldiag = self.cell[f].diagonal()
                half_celldiag = celldiag/2.0
            cxyz = xyz[f]
            for i,m in enumerate(atomlist):
                mxyz = cxyz[m]
                # unwrap the molecule, the reference image is the first atom:
                #         if any other atom is farther then the half cell we need to unwrap it
                #         NOTE this works only for molecules smaller then half the cell!!!
                dmxyz = mxyz[1:]-mxyz[0]
                mxyz[1:] += np.where(np.less_equal(dmxyz, -half_celldiag), 1.0, 0.0)*celldiag
                mxyz[1:] -= np.where(np.greater   (dmxyz,  half_celldiag), 1.0, 0.0)*celldiag
                ccom = np.sum(mxyz*mass, axis=0)/summass
                # now wrap ccom back into the box if it is outside (box goes from -halfcell to +halfcell)
                ccom += np.where(np.less   (ccom,-half_celldiag), celldiag, 0.0)
                ccom -= np.where(np.greater(ccom, half_celldiag), celldiag, 0.0)
                com[f,i,:] = ccom
            report()
        if self.nodes>1:
            buf = np.zeros([self.nframes,nmols,3],dtype=self.xyz.dtype)
            self.comm.Allreduce(com, buf, MPI.SUM)
            com = buf
        if store:
            dataname = molname+"_COM"
            self.pprint("storing COM info in dataset %s" % dataname)
            if self.node_id ==0:
                self.com_h5 = self.traj.require_dataset(dataname, shape=com.shape, dtype=com.dtype)
                self.com_h5[...] = com
        return com
        
    def load_com(self, molname):
        if molname+"_COM" in list(self.traj.keys()):
            self.com_h5 = self.traj[molname+"_COM"]
            return np.array(self.com_h5)
        else:
            return None
        
    #### 3D Maps (COM probability distributions)
        
    def map3D_com(self, com, gridsize):
        self.pprint("\nTRALYZER2")
        self.pprint("Computing a 3D map of a com")
        nframes = com.shape[0]
        gridsize = np.array(gridsize,dtype="int32")
        grid = np.zeros(gridsize+np.array([1,1,1]), dtype="d")
        #grid = np.zeros(gridsize, dtype="d")
        self.pprint("WARING: Map currently works only for constant volume ensembles!!")
        if not self.fixed_cell:
            raise ValueError("map_com works currently only for fixed cell sizes")
        if not ((self.boundarycond==1)or(self.boundarycond==2)):
            raise ValueError("Can't map for these boundaryconditions")
        # shift com by half a cell +h/2 up to get all coordinates above zeros
        celldiag = self.cell.diagonal()
        h = celldiag/gridsize
        com += (celldiag+h)/2.0
        int_com = (com/h).astype("int32")
        entries = 0
        for f in range(self.node_id, nframes, self.nodes):
            for ind in int_com[f].tolist():
                grid[tuple(ind)]+=1.0
                entries += 1
        ## comunicate 
        #if self.nodes > 1:
            ## first allreduce (also number of entries need to be allreduced)
            #buf = np.zeros(grid.shape, grid.dtype)
            #self.comm.Allreduce(grid, buf, MPI.SUM)
            #grid = buf
            #entries = self.comm.allreduce(entries)
        # fix boundary conditions (values at the "faces" need to be identical)
        grid[0,:,:] += grid[-1,:,:]
        grid[:,0,:] += grid[:,-1,:]
        grid[:,:,0] += grid[:,:,-1]
        grid[-1,:,:] = grid[0,:,:]
        grid[:,-1,:] = grid[:,0,:]
        grid[:,:,-1] = grid[:,:,0]
        # normalize
        grid /= (float(entries)*h.prod())
        #if store:
            #dataname = name+"_MAP"
            #if self.nodes>1:
                #if replace:
                    #self.pprint("trying to replace the dataset %s" % dataname)
                    #del(self.traj[dataname])
                #self.pprint("storing map in dataset %s" % dataname)
                #self.map_h5 = self.traj.require_dataset(dataname, shape=grid.shape, dtype=grid.dtype)
                #self.map_h5[...] = grid
        return grid

    #def load_map(self, mapname):
        #self.map_h5 = self.traj[mapname]
        #return np.array(self.map_h5)
        
    def write_map_as_cube(self, map3d, fname, geomstage, showmol):
        if not ((self.boundarycond==1)or(self.boundarycond==2)):
            raise ValueError("Can't map for these boundaryconditions")
        self.pprint("\nTRALYZER\nWriting cube file of map with geometry from stage %s\n" % geomstage)
        mollist = self.get_mollist(showmol)
        atomlist = []
        for m in mollist: atomlist += self.mols[m]
        elems = []
        for a in atomlist: elems.append(atomicnumbers[self.elems[a]])
        all_xyz = np.array(self.pdpf.h5file[geomstage]["restart"]["xyz"])
        mol_xyz = all_xyz.take(atomlist, axis=0)
        natoms = len(atomlist)
        N3 = list(map3d.shape)
        if len(N3) == 2:
            self.pprint("Writing 2D map as a pseudo 3D")
            N3 += [1]
        N3 = np.array(N3)
        celldiag = self.cell.diagonal()
        hcell = celldiag/2.0*a2b
        mol_xyz *= a2b
        h3 = (celldiag/N3)*a2b
        if self.node_id==0:
            f = open(fname+".cube","w")
            f.write("3DMAP as cube %s \n" % (fname))
            f.write("\n")
            f.write("%5d%12.6f%12.6f%12.6f\n" % (natoms, -hcell[0]+h3[0]/2.0, -hcell[1]+h3[1]/2.0, -hcell[2]+h3[2]/2.0))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (N3[0], h3[0], 0.0, 0.0))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (N3[1], 0.0, h3[1], 0.0))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (N3[2], 0.0, 0.0, h3[2]))
            for i in range(natoms):
                xyz = mol_xyz[i]
                f.write("%5d%12.6f%12.6f%12.6f%12.6f\n" % (elems[i], 0.0, xyz[0], xyz[1], xyz[2]))
            lformat = N3[2]*"%e "+"\n"
            for x in range(N3[0]):
                for y in range(N3[1]):
                    ld = map3d[x,y]
                    if N3[2] == 1: ld = [ld]
                    f.write(lformat % tuple(ld))
            f.close()
        return

# NOT working properly ... why??
#    def map_symmetrize(self, map3d):
#        """ currently only a cubic symmetrization is performed """
#        new_map = (map3d+np.transpose(map3d, axes=(1,2,0))+np.transpose(map3d, axes=(2,0,1)))/3.0
#        return new_map

    def map_com(self, com, dims, dlen):
        dimcode = {"x": 0, "y": 1, "z": 2}
        self.pprint("\nTRALYZER2")
        self.pprint("Computing a map of a com")
        nframes = com.shape[0]
        dims.sort()
        if len(dims) != len(dlen):
            raise ValueError("You need to specify a length for all requested dimensions")
        gridsize = [1,1,1]
        for i,d in enumerate(dims):
            gridsize[dimcode[d]] = dlen[i]
        gridsize = np.array(gridsize,dtype="int32")
        grid = np.zeros(gridsize, dtype="d")
        self.pprint("WARING: Map currently works only for constant volume ensembles!!")
        if not self.fixed_cell:
            raise ValueError("map_com works currently only for fixed cell sizes")
        if not ((self.boundarycond==1)or(self.boundarycond==2)or(self.boundarycond==6)):
            raise ValueError("Can't map for these boundaryconditions")
        # shift com by half a cell +h/2 up to get all coordinates above zeros
        celldiag = self.cell.diagonal()
        h = celldiag/gridsize
        com += celldiag/2.0
        int_com = (com/h).astype("int32")
        entries = 0
        for f in range(self.node_id, nframes, self.nodes):
            for ind in int_com[f].tolist():
                grid[tuple(ind)]+=1.0
                entries += 1
        # normalize
        grid /= float(entries)
        # flatten
        new_shape = []
        for i in grid.shape:
            if i > 1: new_shape.append(i)
        grid.shape = new_shape
        return grid
        

    def com_msd(self, com, molname, dt_max=None, store=True):
        """ computes the mean square diffusion for a given COM 
            we keep the MSD per molecule and per dimension
            averaging is easy by taking sum of the array """
        if not ((self.boundarycond==1)or(self.boundarycond==2)or(self.boundarycond==6)):
            raise ValueError("Can't map for these boundaryconditions")
        self.pprint("\nComputing MSD from COM of molecule %s" % molname)
        # set up
        if dt_max:
            self.pprint("maximum deltat=%10.5f ps of a total of %10.5f ps" % (dt_max, self.nframes*self.dt))
            max_msd_frames = int(dt_max/self.dt)
            if max_msd_frames > self.nframes-1:
                raise ValueError("requested detla t is more then the avialbale sampling time")
        else:
            max_msd_frames = self.nframes
        if self.fixed_cell:
            celldiag = self.cell.diagonal()
            celldiag_half = celldiag/2.0
        else:
            celldiag = np.diagonal(self.cell, axis1=1, axis2=2)
        # before we can compute the MSD we have to "unfold" the com trajectory
        # we need an offset (multiples of the cell params) which needs to be incremented/decrementd
        # anytime the coordinate jumps by more then the cell param.
        offset = np.zeros(com.shape, dtype="int32")
        current_offset = np.zeros(com.shape[1:3], dtype="int32")
        self.pprint("Unfolding COM trajectory (non-parallel step!!!)")
        for i in range(1,self.nframes):
            if not self.fixed_cell:
                celldiag_half = celldiag[i]/2.0
            d = com[i]-com[i-1]
            negative_jump = np.less(d,-celldiag_half)
            positive_jump = np.greater(d,celldiag_half)
            current_offset += np.where(negative_jump, 1, 0)
            current_offset -= np.where(positive_jump, 1, 0)
            offset[i,...] = current_offset
        self.pprint("done with unfolding")
        if self.fixed_cell:
            com_unfold = com + offset*celldiag
        else:
            com_unfold = com + offset*celldiag[:,np.newaxis,:]
        # now lets do the msd
        msd = np.zeros(com.shape, dtype="float64")
        entries = np.zeros(com.shape[0], dtype="float64")
        entries[0] = 1.0
        if dt_max:
            # set the upper (unused) part of entries to one in order to avoid a division by zero
            entries[max_msd_frames+1:] = 1.0
        self.pprint("Now we compute the MSD up to a maximum deltat %10.5f ps" % (self.dt*max_msd_frames))
        for i in range(1+self.node_id,self.nframes,self.nodes):
            if (i%(self.nodes*100))==1: self.pprint("master node processing frame %d" % i)
            last_frame = i+max_msd_frames
            if last_frame > self.nframes: last_frame = self.nframes        
            r0 = com_unfold[i-1]
            rt = com_unfold[i:last_frame]
            dr = rt-r0
            msd[1:last_frame+1-i] += dr*dr
            entries[1:last_frame+1-i] += 1.0
        # now communicate and normalize
        if self.nodes>1:
            buf = np.zeros(msd.shape, msd.dtype)
            self.comm.Allreduce(msd, buf, MPI.SUM)
            msd = buf
            buf = np.zeros(entries.shape, entries.dtype)
            self.comm.Allreduce(entries, buf, MPI.SUM)
            entries = buf
        msd /= entries[:,np.newaxis,np.newaxis]
        if store:
            dataname = molname+"_MSD"
            self.pprint("storing MSD info in dataset %s" % dataname)
            if self.node_id == 0:
                self.msd_h5 = self.traj.require_dataset(dataname, shape=msd.shape, dtype=msd.dtype)
                self.msd_h5[...] = msd
            self.comm.Barrier()
        return msd

    def load_msd(self, msdname):
        self.msd_h5 = self.traj[msdname]
        return np.array(self.msd_h5)
        
    def write_msd(self, msd, fname):
        self.pprint("\nWriting MSD data to %s" % fname)
        if self.node_id == 0:
            f = open(fname, "w")
            smsd = np.sum(msd, axis=1)/float(msd.shape[1])
            tsmsd = np.sum(smsd, axis=1)/3.0
            # convert from A^2 to nm^2
            smsd  *= 0.01
            tsmsd *= 0.01
            for i in range(self.nframes):
                f.write("%10.5f %12.6f %12.6f %12.6f %12.6f\n" % (i*self.dt*0.001, smsd[i,0], smsd[i,1], smsd[i,2], tsmsd[i]))
            f.close()
        self.comm.Barrier()
        return
    
    def calc_diffc(self, msd, firstframe, lastframe):
        smsd = np.sum(msd, axis=1)/float(msd.shape[1])
        tsmsd = np.sum(smsd, axis=1)/3.0
        # convert from A^2 to nm^2
        smsd  *= 0.01
        tsmsd *= 0.01
        Diffadd = 0
        for i in range(firstframe, lastframe-1):
            if  i != 0:
                Diff = tsmsd[i]/(i*self.dt*0.001*2)
                Diffadd += Diff
            else:
                continue
        diffco = (Diffadd/((lastframe-1)-firstframe))
        return diffco
    
    def com_dipole(self, dip_molname, store=False):
        for i, item in enumerate(self.molnames):
            if item == dip_molname: molindex = i
        charges_dip = self.traj["charges"]
        coords_dip = self.traj["xyz"]
        # containers
        dipole_vec = 0
        q_storage = []
        target_mols = []
        coords_storage = []
        # get list of target molecules
        for i in range(self.nmols):
            if self.moltypes[i] == molindex: target_mols.append(i)
        # get list of coordinates of target molecule       
        for i in self.mols[target_mols[0]]:
            coords_storage.append(coords_dip[0][i])
        # convert into array to allow multiplication later on
        coords_storage = np.asarray(coords_storage)
        
        dipole_storage = np.zeros([self.nframes,len(target_mols)],dtype="float64")
        # calculate dipoles
        for j in range(self.nframes):
            cf_charges = charges_dip[j]
            for i in target_mols:
                aindeces = self.mols[i]
                for k in aindeces:
                    q_storage.append(cf_charges[k])
                #calculation dipole vector
                for k in range(len(q_storage)):
                    dipole_vec += coords_storage[k]*q_storage[k]
                #calculation dipole
                dipole = np.sqrt(dipole_vec[0]*dipole_vec[0]+dipole_vec[1]*dipole_vec[1]+dipole_vec[2]*dipole_vec[2])
                #add dipole to dipole_storage
                dipole_storage[j][i] = dipole
                #empty q_storage and reset dipole vector
                q_storage[:] = []
                dipole_vec = 0
        #calculation average dipole
        if store:
            dataname = dip_molname+"_DIPOLE"
            self.pprint("Storing dipole vector info in dataset %s" % (dataname))
            self.dipole_h5 = self.traj.require_dataset(dataname,shape=dipole_storage.shape, dtype=dipole_storage.dtype)
            self.dipole_h5[...] = dipole_storage/0.2082                 
        nentries = self.nframes*len(target_mols)
        avg_dipole = np.sum(dipole_storage)/nentries/0.2082
        print(avg_dipole)
        self.pprint("The average Dipole is %.3f" % (avg_dipole))
        return dipole_storage
    
    def histo_dip(self, dip_results, nbins, maxrange):
        self.pprint("Output written to dipoledistro.dat")
        f=open("dipoledistro.dat","w")
        dip_results = np.array(dip_results)
        hist, bin_edges = np.histogram(dip_results/0.2082, bins=nbins, range=(0,maxrange))
        dr=(maxrange/float(nbins))
        highest = float(max(hist))
        for i in range(nbins):
            k=(i+0.5)*dr
            f.write("%.3f %.9f\n" % (k, hist[i]/highest))
        f.close()
        return
    
    def dielec(self,temperature,store=False):
        #conversion factors
        ang_to_m = 1e-10
        e_to_c = 1.602177e-19
        eps_0 = 8.854187817e-12
        
        #storage for dipole vector and polarization
        dipvec_box = np.zeros([self.nframes,3],dtype="float64")
        if self.fixed_cell:
            celldiag_cf = self.cell.diagonal()
        else:
            celldiag = np.diagonal(self.cell, axis1=1, axis2=2)
        avg_p = 0
        avg_p2 = 0
        for i in range(self.nframes):
            #current frame values and temp storage for molecule's dipole vector 
            cf_charges = self.traj['charges'][i]
            cf_xyz = self.traj['xyz'][i]
            temp_dipvec_box = np.zeros([self.nmols,3],dtype="float64")
            if not self.fixed_cell:
                    celldiag_cf = celldiag[i]
            volume = celldiag_cf[0]*celldiag_cf[1]*celldiag_cf[2]
            for j in range(self.nmols):
                tar_indeces = self.mols[j]
                mol_xyz = cf_xyz[tar_indeces]
                mol_charges = cf_charges[tar_indeces]                                
                #unwrap
                dxyz = mol_xyz-mol_xyz[0]
                dxyz -= celldiag_cf*np.around(dxyz/celldiag_cf)
                temp_dipvec_box[j] = (dxyz*mol_charges[:,np.newaxis]).sum(axis=0)
            dipvec_box[i] = np.sum(temp_dipvec_box,axis=0)        
            avg_p += np.sum(dipvec_box[i]/volume)  
            avg_p2 += np.sum((dipvec_box[i]/volume)**2)
        avg_p /= self.nframes
        avg_p2 /= self.nframes
        #conversion
        volume = volume*ang_to_m**3
        avg_p = avg_p*e_to_c/ang_to_m**2
        avg_p2 = avg_p2*e_to_c**2/ang_to_m**4
 
        boltz = 1.38064852e-23*temperature
        eps = 1.0+(volume)/(3.0*boltz*eps_0)*(avg_p2-avg_p**2)
        
        if store:
            self.pprint("Storing system dipole vector info in dataset 'dipvec_box'")
            self.dipvec_box_h5 = self.traj.require_dataset("dipvec_box",shape=dipvec_box.shape, dtype=dipvec_box.dtype)
            self.dipvec_box_h5[...] = dipvec_box        
        self.pprint(eps)        
        return
    
    def dielec_precalc(self,temperature, volume):
        #conversion factors
        ang_to_m = 1e-10
        e_to_c = 1.602177e-19
        eps_0 = 8.854187817e-12

        avg_p = 0
        avg_p2 = 0
        dipoles = self.traj['dipole'][:]
        zero = (dipoles.sum(axis=0))/len(dipoles)
        for i in range(len(dipoles)):
            avg_p += np.sum((dipoles[i]-zero)/volume)  
            avg_p2 += np.sum(((dipoles[i]-zero)/volume)**2)
        avg_p /= len(dipoles)
        avg_p2 /= len(dipoles)
        #conversion
        volume = volume*ang_to_m**3
        avg_p = avg_p*e_to_c/ang_to_m**2
        avg_p2 = avg_p2*e_to_c**2/ang_to_m**4
        boltz = 1.38064852e-23*temperature
        eps = 1.0+(volume)/(3.0*boltz*eps_0)*(avg_p2-avg_p**2)
        self.pprint(eps)        
        return
    
    
    #CDP: work in progress, has to be merged into dielec() with flags
    def fld(self, field):
        #development
        #calculates dielectric constant for systems with an applied field
        ang_to_m = 1e-10
        e_to_c = 1.602177e-19
        eps_0 = 8.854187817e-12
        #storage for dipole vector and polarization
        dipvec_box = np.zeros([self.nframes,3],dtype="float64")
        pol_box = np.zeros([self.nframes,3],dtype="float64")
        if self.fixed_cell:
            celldiag_cf = self.cell.diagonal()
        else:
            celldiag = np.diagonal(self.cell, axis1=1, axis2=2)
        for i in range(self.nframes):
            #current frame values and temp storage for molecule's dipole vector 
            cf_charges = self.traj['charges'][i]
            cf_xyz = self.traj['xyz'][i]
            temp_dipvec_box = np.zeros([self.nmols,3],dtype="float64")
            if not self.fixed_cell:
                    celldiag_cf = celldiag[i]
            volume = celldiag_cf[0]*celldiag_cf[1]*celldiag_cf[2]
            for j in range(self.nmols):
                tar_indeces = self.mols[j]
                mol_xyz = cf_xyz[tar_indeces]
                mol_charges = cf_charges[tar_indeces]                                
                #unwrap
                dxyz = mol_xyz-mol_xyz[0]
                dxyz -= celldiag_cf*np.around(dxyz/celldiag_cf)
                temp_dipvec_box[j] = (dxyz*mol_charges[:,np.newaxis]).sum(axis=0)
            dipvec_box[i] = np.sum(temp_dipvec_box,axis=0) #list that contains the box dipole vector of all frames
            pol_box[i] = dipvec_box[i]/volume #list that contains the box polarization vector of all frames           
        avg_dip_x = np.sum(pol_box,axis=0)[0]/self.nframes
        avg_dip_y = np.sum(pol_box,axis=0)[1]/self.nframes
        avg_dip_z = np.sum(pol_box,axis=0)[2]/self.nframes
        
        f = open('dielec_const.dat','w')
        f.write("<P_x>:  %5f  e/Ang^2\n" % (avg_dip_x))
        f.write("<P_y>:  %5f  e/Ang^2\n" % (avg_dip_y))
        f.write("<P_z>:  %5f  e/Ang^2\n" % (avg_dip_z))
        
        #conversion
        avg_dip_x = avg_dip_x*e_to_c/ang_to_m**2
        avg_dip_y = avg_dip_y*e_to_c/ang_to_m**2
        avg_dip_z = avg_dip_z*e_to_c/ang_to_m**2
        field /= ang_to_m
 
        eps_x = 1.0+avg_dip_x/(eps_0*field)
        eps_y = 1.0+avg_dip_y/(eps_0*field)
        eps_z = 1.0+avg_dip_z/(eps_0*field)
        
        f.write("Eps_x: %5f\n" % (eps_x))
        f.write("Eps_y: %5f\n" % (eps_y))
        f.write("Eps_z: %5f\n" % (eps_z))
        
        self.pprint(eps_x)        
        return
        
        
    
    def com_avg_charge(self, atmtype, writeq, filename=None, nbins=None, minrange=None, maxrange=None):
        aindeces = []
        acharges = []
        charges = self.traj["charges"]
        for i in range(self.natoms):
            if self.types[i] == atmtype: aindeces.append(i)
        for j in range(self.nframes):
            cf_charges = charges[j]
            for i in range(self.natoms):
                if i in aindeces: acharges.append(cf_charges[i])
        avg_q = sum(acharges)/float(len(acharges))
        self.pprint("The average charge of %s is %.6f" % (atmtype, avg_q))
        if writeq:           
            fq = open(filename+".dat", "w")
            acharges = np.array(acharges)
            hist, bin_edges = np.histogram(acharges, bins=nbins, range=(minrange,maxrange))
            for i in range(nbins):
                k=minrange+i*((maxrange-minrange)/nbins)
                fq.write("%.3f %.9f\n" % (k, hist[i]/float(sum(hist))))
            fq.close
        return
    
    
    def track_mol_z(self,tarmol):
        z_container = []
        coords = self.traj["xyz"]
        #calculate average z position of all atoms in the molecule
        #print coords[0][self.mols[tarmol]][2]
        for i in range(self.nframes):
            cf_zcoords_mol = coords[i][self.mols[tarmol]][:,2] #z coordinates of all atoms belonging to the target molecule for current frame
            avg_z = sum(cf_zcoords_mol)/len(cf_zcoords_mol)
            z_container.append(avg_z)
        self.pprint("Writing results to z_coord_output")
        f = open("z_coord_output.dat", "w")
        for i in z_container:
            f.write("%.4f\n" % i)
        f.close
        return
    
    def snip(self, target, tarcut, ntarget, shuffle, fullmol, tarframe):
        counter = 1
        coords = self.traj["xyz"]
        #list of all indeces for atoms of the type 'target'
        target_indeces = []
        #celldiag for later unfolding
        if self.fixed_cell:
            celldiag_cf = self.cell.diagonal()
        else:
            celldiag = np.diagonal(self.cell, axis1=1, axis2=2)
        for i in range(self.natoms):
            if self.types[i] == target: target_indeces.append(i)
        if not target_indeces: 
            self.pprint("\n ***ERROR: This atom type does not appear in the trajectory file!***\n")
            return
        for i in range(self.nframes):
            if i%tarframe == 0:
                if not self.fixed_cell:
                    celldiag_cf = celldiag[i]
                cframe = coords[i]
                cf_indeces = []
                #if shuffle is set to true, it picks random clusters, else it picks the first x clusters
                if shuffle: np.random.shuffle(target_indeces)
                for j in range(ntarget):
                    cf_indeces.append(target_indeces[j])
                #for each selected target, check the distance to all other atoms
                for j in cf_indeces:
                    within_tarcut = []
                    dxyz = cframe[j]-cframe
                    dxyz -= celldiag_cf*np.around(dxyz/celldiag_cf)
                    #distance check
                    for k, kitem in enumerate(dxyz):
                        if np.linalg.norm(kitem) <= tarcut: within_tarcut.append(k)
                    if fullmol:
                        within_tarcut_full = []
                        for l in self.mols:
                            if set(within_tarcut) & set(l): within_tarcut_full.append(l)
                        #flatten list
                        within_tarcut = [x for y in within_tarcut_full for x in y]
                    o = open("frame_"+str(counter)+".xyz", "w")
                    o.write("%d\n" % len(within_tarcut))
                    o.write("%d's frame, cluster around %d\n" % (i, j))
                    for k in within_tarcut:
                        o.write("%s  %.6f  %.6f  %.6f\n" %(self.elems[k], dxyz[k][0], dxyz[k][1], dxyz[k][2]))
                    o.close
                    counter += 1
        return
    
    def get_keys(self):
        return list(self.traj.keys())
    
    def get_data(self, dataset):
        data = self.traj[dataset][:]
        return data
    
    ####  oriantation stuff
    
    def compute_orient(self, molname, atompair, store=True):
        mollist = self.get_mollist(molname)
        nmols = len(mollist)
        self.pprint("\ncomputing the orientation vector of %d molecules with name %s" % (nmols,molname))
        self.pprint("   using atoms %d and %d to compute the vector" % (atompair[0], atompair[1]))
        self.pprint("Molecules are unwrapped .. this works only if the molecule is smaller then half the unit cell!!")
        atomlist = []
        xyz = self.traj["xyz"]
        for m in mollist:
            atomlist.append(self.mols[m])
        # allocate the array for the orientation
        # NOTE we could use the dtype of the xyz dataype here 
        orient = np.zeros([self.nframes,nmols,3],dtype=xyz.dtype)
        if self.fixed_cell:
            celldiag = self.cell.diagonal()
            half_celldiag = celldiag/2.0
        for f in range(self.node_id,self.nframes, self.nodes):
            if (f%(self.nodes*100))==0: self.pprint("master node is processing frame %d" % f)
            if not self.fixed_cell:
                celldiag = self.cell[f].diagonal()
                half_celldiag = celldiag/2.0
            cxyz = xyz[f]
            for i,m in enumerate(atomlist):
                mxyz = cxyz[m]
                # unwrap the molecule, the reference image is the first atom:
                #         if any other atom is farther then the half cell we need to unwrap it
                #         NOTE this works only for molecules smaller then half the cell!!!
                dmxyz = mxyz[1:]-mxyz[0]
                mxyz[1:] += np.where(np.less   (dmxyz, -half_celldiag), 1.0, 0.0)*celldiag
                mxyz[1:] -= np.where(np.greater(dmxyz,  half_celldiag), 1.0, 0.0)*celldiag
                vect = mxyz[atompair[0]]-mxyz[atompair[1]]
                vect /= np.sqrt(np.sum(vect*vect))
                orient[f,i] = vect
        if self.nodes>1:
            buf = np.zeros(orient.shape,dtype=orient.dtype)
            self.comm.Allreduce(orient, buf, MPI.SUM)
            orient = buf
        if store:
            dataname = molname+"_ORIENT"
            self.pprint("storing orientation info in dataset %s" % dataname)
            if self.node_id==0:
                self.orient_h5 = self.traj.require_dataset(dataname, shape=orient.shape, dtype=orient.dtype)
                self.orient_h5[...] = orient
            self.comm.Barrier()
        return orient
        
    def load_orient(self, molname):
        self.orient_h5 = self.traj[molname+"_ORIENT"]
        return np.array(self.orient_h5)

    
        
    def orient_autocor(self, orient, molname, dt_max=None, store=True):
        """ computes the autocorrelation function of an orientation 
            assumes orient to be a (nframes, nmol, 3) array containing NORMALIZED orientation vectors"""
        self.pprint("\nComputing Orientational Autocorrelation Function (OACF) for molecule %s" % molname)
        # set up
        
        if dt_max:
            self.pprint("maximum deltat=%10.5f ps of a total of %10.5f ps" % (dt_max, self.nframes*self.dt))
            max_frames = int(dt_max/self.dt)
            if max_frames > self.nframes-1:
                raise ValueError("requested delta t is more then the avialbale sampling time")
        else:
            max_frames = self.nframes
        nmols = orient.shape[1]
        oacf = np.zeros([orient.shape[0],nmols], dtype="float64")
        c1 = np.zeros([max_frames,nmols])
        #c2 = np.zeros([max_frames,nmols])
        entries = np.zeros([orient.shape[0]], dtype="float64")
        entries2 = np.zeros(max_frames, dtype="float64")
        entries[0] = 1.0
        if dt_max:
            # set the upper (unused) part of entries to one in order to avoid a division by zero
            entries[max_frames+1:] = 1.0
        for i in range(1+self.node_id,self.nframes, self.nodes):
            if (i%(self.nodes*100))==1: self.pprint("master node processing frame %d" % i)
            last_frame = i+max_frames
            #if last_frame > self.nframes: last_frame = self.nframes        
            if last_frame > self.nframes: continue
            o0 = orient[i-1]
            ot = orient[i:last_frame]
            scalprod = np.sum(ot*o0,axis=2)
            c1[0:last_frame] += scalprod
            entries2[0:last_frame] += 1.0
            oacf[1:last_frame+1-i] += scalprod
            entries[1:last_frame+1-i] += 1.0
        # now normalize
        if self.nodes>1:
            buf = np.zeros(oacf.shape, oacf.dtype)
            self.comm.Allreduce(oacf, buf, MPI.SUM)
            oacf = buf
            buf = np.zeros(entries.shape, entries.dtype)
            self.comm.Allreduce(entries, buf, MPI.SUM)
            entries = buf
        oacf /= entries[:,np.newaxis]
        oacf[0] = 1.0
        c1 /= entries2[:,np.newaxis]
        c1[0] = 1.0
        if store:
            #dataname = molname+"_c1" #+str(int())
            #self.pprint("storing orientational autocorrelation function info in dataset %s" % dataname)
            #if self.node_id==0:
            #    self.oacf_h5 = self.traj.require_dataset(dataname, shape=c1.shape, dtype=c1.dtype)
            #    self.oacf_h5[...] = c1
            dataname = molname+"_OACF" 
            if self.node_id==0:
                self.oacf_h5 = self.traj.require_dataset(dataname, shape=oacf.shape, dtype=oacf.dtype)
                self.oacf_h5[...] = oacf
            self.comm.Barrier()
        return c1

    def get_relaxation_time(self,cf,ask=False):
        #fit A=exp(-t/tau) to extract tau (rotational relaxation time)
        def fun(t,tau):
            return np.exp(-t / tau)
        #def fun2(t,tau,A):
        #    return np.exp(-t / tau)
        try:
            from scipy.optimize import curve_fit
            import matplotlib.pyplot as plt
        except:
            raise ImportError('Scipy or matplotlib  not available!')
        len_cf  =len(cf)
        ### cf comes with shape (nsteps,1), 
        cf =  cf[:,0]
        x = np.linspace(0,self.dt*len_cf,len_cf)

        ### weighted fit: sigma corresponds to the weights
        #sigma  =1.0 - ( cf - np.min(cf))+0.05
        #sigma = cf - np.min(cf) + 0.01
        #sigma = np.sqrt(np.linspace(0.2,1,len(cf)))
        #fit = curve_fit(fun,x,cf,sigma=sigma,absolute_sigma=True)
        
        fit = curve_fit(fun,x,cf)
        
        relax_time = fit[0]
        print('RELAXATION TIME  t_relx = %8.5f ps' % (relax_time, ))
        plt.scatter(x,cf,marker='.')
        plt.plot(x,fun(x,relax_time),color='r')
        plt.xlabel('t / ps')
        plt.ylabel('autocorrelation')
        plt.legend(['simulation','fit'])
        plt.savefig('relaxation_time_fit.png',dpi=300)

        ###
        if ask == True:
            keyname = [i for i in list(self.traj.keys())if i.count('OACF') == 1][0]
            cf = self.traj[keyname]
            len_cf  =len(cf)
            cf =  cf[:,0]
            x = np.linspace(0,self.dt*len_cf,len_cf)
            fit = curve_fit(fun,x,cf)
            relax_time = fit[0]
            print('RELAXATION TIME  t_relx = %8.5f ps' % (relax_time, ))
            #plt.scatter(x,cf,marker='.')
            #plt.plot(x,fun(x,relax_time),color='r')
            #plt.xlabel('t / ps')
            #plt.ylabel('autocorrelation')
            #plt.legend(['simulation','fit'])
            #plt.savefig('relaxation_time_from_full_OACF_fit.png',dpi=300)
            plt.scatter(x[0:500],cf[0:500],marker='.')
            plt.plot(x[0:500],fun(x,relax_time)[0:500],color='b')
            plt.xlabel('t / ps')
            plt.ylabel('autocorrelation')
            plt.legend(['simulation','fit'])
            plt.savefig('relaxation_time_from_full_OACF_fit.png',dpi=300)

        return


        

    def load_oacf(self, molname):
        self.oacf_h5 = self.traj[molname+"_OACF"]
        return np.array(self.oacf_h5)


    def write_oacf(self, oacf, fname, all=False, dt_max=None):
        self.pprint("\nWriting OACF to dat file")
        if self.node_id==0:
            nframes = oacf.shape[0]
            if dt_max:
                self.pprint("maximum deltat=%10.5f ps of a total of %10.5f ps" % (dt_max, nframes*self.dt))
                max_frames = int(dt_max/self.dt)
                if max_frames > nframes-1:
                    raise ValueError("requested delta t is more then the avialbale sampling time")
            else:
                max_frames = nframes
            f = open(fname, "w")
            nmol = oacf.shape[1]
            soacf = np.sum(oacf, axis=1)/float(nmol)
            for i in range(max_frames):
                line = "%10.5f " % (i*self.dt*0.001)
                if all:
                    line += (nmol*"%10.5f ") % tuple(oacf[i])
                line += "%12.6f \n" % soacf[i]
                f.write(line)
            f.close()
        return


    def map_com_2D(self, com, gridsize, name, store=True, orient=None):
        """ this is "cell based" in contrast to the "vertex based" code for the 3D map
            the reason is that VMD does it different for isosurface and volumeslice.
            please use this cube file ONLY for volume slices .. there is no z-info anyway"""
        self.pprint("\nComputing a 2D map of a com for molecule %s in the xy plane" % name)
        nframes = com.shape[0]
        grid = np.zeros(gridsize, dtype="d")
        self.pprint("WARING: Map currently works only for constant volume ensembles!!")
        if not self.fixed_cell:
            raise ValueError("map_com works currently only for fixed cell sizes")
        if not ((self.boundarycond==1)or(self.boundarycond==2)):
            raise ValueError("Can't map for these boundaryconditions")
        # shift com by half a cell/2 up to get all coordinates above zeros
        celldiag = self.cell.diagonal()
        h = celldiag/(gridsize+[1])
        com += (celldiag)/2.0
        int_com = (com/h).astype("int32")
        entries = 0
        for f in range(self.node_id, nframes, self.nodes):
            for i, ind in enumerate(int_com[f].tolist()):
                if orient != None:
                    val = orient[f,i,2]
                    val *= val
                else:
                    val=1.0
                grid[tuple(ind[:2])]+=val
                entries += 1
        # comunicate 
        if self.nodes > 1:
            # first allreduce (also number of entries need to be allreduced)
            buf = np.zeros(grid.shape, grid.dtype)
            self.comm.Allreduce(grid, buf, MPI.SUM)
            grid = buf
            entries = self.comm.allreduce(entries)
        # normalize and convert to a probability density
        grid /= (float(entries)*h.prod())
        if store:
            dataname = name+"_2DMAP"
            if self.nodes>1:
                self.pprint("storing 2Dmap in dataset %s" % dataname)
                self.map_h5 = self.traj.require_dataset(dataname, shape=grid.shape, dtype=grid.dtype)
                self.map_h5[...] = grid
        return grid

        
    def remove_from_mfp5(self, dataset):
        self.pprint("\nRemoving dataset %s from group /%s/traj" % (dataset, self.stagename))
        del(self.traj[dataset])
        return

