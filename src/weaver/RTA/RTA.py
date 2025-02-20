import copy
import weaver
import pylmps
import subprocess
import os
import shutil
import sys
import string
import numpy
import pickle
from molsys import mpiobject
from . import genetic
import molsys.util.atomtyper as atomtyper



class RTA(mpiobject):
    def __init__(self,framework, orientfile=None, nproc = 1, ff = "MOF-FF",thresh_min = 0.1, 
            thresh_latmin = 0.15,opt_method=1, subdir = True, mpi_comm = None, out = None,use_cache=False):
        super(RTA,self).__init__(mpi_comm, out)
        self.f = framework
        self.nproc = nproc
        self.ff=ff
        self.thresh_min = thresh_min
        self.thresh_latmin = thresh_latmin
        self.opt_method = opt_method
        if orientfile is not None:
            self.f.read_orientations(orientfile)
            
        # TBI finish this stuff
        self.use_cache = use_cache
        self.cache_mol = None
        self.options = {}
        self.options["thresh_min"] = thresh_min 
        self.options["thresh_latmin"] = thresh_latmin
        # generate a mask to remove all orientations which do not change
        self.omask = []
        for i,n in enumerate(self.f.norientations):
            if n>1: self.omask.append(i)
        if subdir:
            self.subdir = "gaopt_"+self.f.name
        else:
            self.subdir = None
        return

    def orient2chrom(self,orient):
        chrom = numpy.array(orient).take(self.omask)
        return chrom.tolist()
        
    def chrom2orient(self,chrom):
        orient = numpy.zeros(len(self.f.norientations),"i")
        orient.put(self.omask, chrom)
        return orient.tolist()

    def opt_struc(self,p, ind):
        """ 
        generate and optimize the structure defined by the orientation p
        we use the integer ind to tag the files.
        if requested we do all of the work in subdirectory in order not to cluter things 
        """
        if self.subdir:
            if not os.path.isdir(self.subdir):
                if self.mpi_rank == 0: os.mkdir(self.subdir)
            self.mpi_comm.Barrier()
            os.chdir(self.subdir)
            path = "../"
        else:
            path = ""
        print(('opt',self.mpi_rank))
        sys.stdout.flush()
        tag = self.f.name+"_%06d" % ind
        # generate the atomstic mol object
        m = self.f.generate_framework(p)
        m.write(tag+".mfpx",ftype='mfpx')
        # set up force field
        at = atomtyper(m)
        at() # call is so great! we exactly know what happens here!
        done=False
        while not done:
            try:
                self.mpi_comm.Barrier()
                m.addon("ff")
                m.ff.assign_params(self.ff,cache=self.cache_mol)
                if self.mpi_rank == 0:m.ff.write(tag)
                if ((self.use_cache == True) and (self.cache_mol == None)):
                    self.cache_mol = copy.copy(m.ff.cache)
                done=True
            except Exception as e:
                print(('THERE WAS AN ERROR HERE!!!!!',e,self.mpi_rank))
                sys.stdout.flush()
                pass
        # TBI use other FF engine .. now we use lammps only
        pl = pylmps.pylmps(tag,out=tag+'.out')
        pl.setup(mol=m)
        # now optimize first the structure
        pl.MIN(self.options["thresh_min"])
        pl.LATMIN(self.options["thresh_latmin"], self.options["thresh_min"])
        # TBI some tests if it converged 
        pl.write(tag+'_latopt.mfpx')
        self.mpi_comm.Barrier()
        res = pl.get_eterm("epot")
        energies = pl.get_energy_contribs()
        enames = list(energies.keys())
        if self.mpi_rank == 0:
            efile = open(tag+'.energy','w')
            efile.write ((len(enames)*'%15s ' % tuple(enames))+'       TOTAL   \n')
            efile.write ((len(enames)*'%15.10f ' % tuple(energies.values())+ '%15.10f\n' % (res)))
            efile.close()
        if self.subdir:
            os.chdir("..")
        # check for memory leaks
        self.mpi_comm.Barrier()
        del pl
        return res
        

    def score(self, chrom, ind):
        orient = self.chrom2orient(chrom)
        self.pprint("evaluating %d with chromsome %s" % (ind, str(chrom)))
        sys.stdout.flush()
        e = self.opt_struc(orient, ind)
        return e

    def run(self, popsize = 16, maxgen = 100):
        relevant = self.orient2chrom(self.f.norientations)
        size = len(relevant)
        world = genetic.Environment(self.f.name, size=popsize, maxgenerations=maxgen, score_func=self.score,
                                    alleles = relevant,restart=False)
        world.run()
        # world.sort_register() #  that one seems not to work. did that ever run?
        return world
