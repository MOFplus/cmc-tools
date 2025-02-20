# new rotator class for weaver  now without knownledge about the real SBU!
from __future__ import absolute_import

# -*- coding: utf-8 -*-
#from numpy import *
import numpy
from scipy import optimize
from scipy.optimize import linear_sum_assignment
import itertools
import time
import copy
import random
from . import vector
from molsys.molsys_mpi import mpiobject
# try:
#    from . import frotator
# except ImportError:
#    from . import frotator3 as frotator

from . import frotator3 as frotator

import logging
logger = logging.getLogger("molsys.rotator")

try: 
    import sobol
    sobol_installed = True
except ImportError:
    logger.warning('sobol sequences not available: pip install sobol')
    sobol_installed = False



from .timing import timer, Timer

#try:
    #from munkres import Munkres
#except Import Error:
    #raise ImportError("install with 'sudo pip install munkres'")

def equal_orientations(o1,o2,small=1.0e-2):
    dev = abs(o1-o2)
    return all(numpy.less(dev,small))

numpy.set_printoptions(suppress=True)

class rotator(mpiobject):
    def __init__(self, v1, v2,bbinstance=None,use_sobol=True,mpi_comm=None,out=None):
        """initializes the rotator module to superimpose two vector sets
        
        The Rotator module solves the orientation problem within the context of the RTA.
        Technically two vector sets, v1 and v2 superimposed onto each other using a Monte-
        Carlo like approach: In n trials, the initial orientations are randomized and the
        assignment of which vector of set v1 corresponds to which vector in set v2 is found.
        Then the optimization given this particular assignment is performed numerically
        In the end a set of assignments + orientations + penalty can be obtained 
        
        Args:
            v1 (numpy.ndarray n x 3): vector set #1
            v2 (numpy.ndarray n x 3): vector set #2
            bbinstance (molsys.mol, optional): If given, BB informations are used. specific connectivity is defined there
            use_sobol (bool, optional): Defaults to True. If True, sobol pseudorandom numbers are used instead of a uniform distribution to achieve better sampling especially at small sample sizes.
        
        Raises:
            ValueError: v1 and v2 have to have the same shape. 
        """
        super(rotator,self).__init__(mpi_comm, out)
        # subsys stuff
        if len(v1)  != len(v2):
            raise ValueError("len(v1) != len(v2)")
        self.subsets=None
        if bbinstance!=None:
            if bbinstance.bb.specific_conn:
                self.set_subsys(bbinstance)
        if sobol_installed == True and use_sobol == True:
            self.use_sobol = True
        else:
            self.use_sobol = False
        self.counter = 0
        self.v1 = numpy.array(v1)
        self.v2 = numpy.array(v2)
        self.v1n = vector.normalize(self.v1)
        self.v2n = vector.normalize(self.v2)
        self.n = len(self.v1)
        self.which = numpy.array(range(self.n),dtype='int')     # convention? v2 is indexed always by 0,1,2,....,n-1
        self.R = numpy.zeros((3,3))
        self.pen = -1
        self.acos_mat = []
        self.timer = Timer()
        return
    
  ##### both, add subsys and set_specific conn should do both ############
  ##### set_specific_conn goes for the existing vectors and splits them into subsystems
  ##### add_subsys must be used when the respective vectors are not there yet
  ##### for weaver functionality, set_specific_conn will be most important, since bb_connector_xyz
  ##### that is passed already contains all the vectors. 
   
    def set_subsys(self,bb):
        self.nvert = bb.bb.nvertex
        self.subsets = len(bb.bb.specific_conn)
        self.conn_sets = bb.bb.connector_types
        ### If there is two different bbs belonging to the same specific conn this does nto work here!
        ### we may get something like this: [['1', '3'], '2'], 1&3 belong to 0, 2 belongs to 1
        sconn_index = {}
        for i,sc in enumerate(bb.bb.specific_conn):
            if hasattr(sc,'__iter__'):
                for sci in sc:
                    sconn_index[sci]=i
            else:
                sconn_index[sc] = i
        #self.vert_sets = [bb.specific_conn.index(i) for i in self.nvert]
        self.vert_sets = [sconn_index[i] for i in self.nvert]
        return
    
    #@timer('assign indices')
    def assign_indices(self,hungarian=True):
        """solves the assignment problem given the current v1n and v2n
        
        The assignment is the index map of which vector in v1 corresponds to which vector in v2 (denoted as self.which)
        in order to find a solution the vectors that are closest to each other are assigned. if that is not possible
        (i.e. if one of v1 has two small angle neighbors), the linear sum assignment is used to come up with a solution
            hungarian (bool, optional): Defaults to True. if false an exhaustive search is used. beware, O(n!)
        stores the assignment as self.which
        Takes care of specific connectivity if there is ome
        Returns:
            float: penalty of the best assignmend
        """
        if self.subsets:
            self.assign_indices_multi()
            return
        acos_mat = self.calc_acos_mat(self.v1n,self.v2n)
        smallest = numpy.argmin(acos_mat, axis=1).tolist() 
        if sorted(smallest) != range(self.n):
            if hungarian:
                self.hungarian(acos_mat)
            # do different schemes according to the problem size!
            else: # now extinct, since hungarian does the job much better!
                pen = self.exhaustive_i(acos_mat)
        else:
            self.which = numpy.array(smallest,dtype='int')  
        pen = self.calc_penalty(numpy.array([1.0,1.0,1.0]))
        return pen
    
    def assign_indices_multi(self):
        self.which = numpy.zeros((self.n,),dtype='int')
        for idxsub in range(self.subsets):
            ninset = self.conn_sets.count(idxsub)
            idxv = [e for e,i in enumerate(self.vert_sets) if i == idxsub] 
            idxc = [e for e,i in enumerate(self.conn_sets) if i == idxsub]
            vtmp = self.v1n[idxv,:]
            ctmp = self.v2n[idxc,:]
            acos_mat = self.calc_acos_mat(vtmp,ctmp)
            which = self.hungarian(acos_mat,return_which=True)
            for i,w in enumerate(which):
                self.which[idxv[i]] = idxc[w]
        self.which=numpy.array(self.which,dtype='int')
        return
    
    def hungarian(self,acos_mat,return_which=False):
        """ call linear sum assignment solver of scipy
        
        solves the linear sum assignment problem using the scipy implementation
        of the hungarian algorithm
        
        Args:
            acos_mat (numpy.ndarray): matrix to operate on
            return_which (bool, optional): Defaults to False. if true: return the which list
        
        Returns:
            list: if return_which is true, the which list is returned 
        """
        rows,cols = optimize.linear_sum_assignment(copy.copy(acos_mat))
        if return_which: return cols
        self.which = cols
    
    def exhaustive_i(self,acos_mat):
        """exhaustively find the best assignment
        
        Args:
            acos_mat (numpy.ndarray n x n): matrix to operate on
        
        Returns:
            float: penalty of best assignment
        """
        #acos_mat = self.acos_mat#1.0-numpy.sum(self.v1n[:,numpy.newaxis,:]*self.v2n[numpy.newaxis,:,:], axis=2)
        pmin = 1e10
        for p in itertools.permutations(range(self.n)):
            #pen=0.0
            #for i, j in enumerate(p): pen += acos_mat[i,j]
            pen = self.calc_penalty([1.0,1.0,1.0],which=p)
            if pen < pmin: pmin,p_keep =pen,p
        self.which = numpy.array(p_keep,dtype='int')
        return pmin

    def calc_acos_mat(self,v1,v2):
        n = v1.shape[0]
        return frotator.calc_acos_mat(n,v1.T,v2.T)
    
    
    #@timer("test superpose")
    def is_superpose(self,v1,v2,thresh,use_fortran=True):
        if len(v1) != len(v2): return False, 666.666
        if not use_fortran:
            rmsd=0.0
            for i in range(len(v1)):
                sxyz=v1[i]
                r = v2-sxyz
                d=numpy.sqrt(numpy.sum(r*r,axis=1))
                closest = numpy.argsort(d)[0]
                if d[closest] > thresh: return False, d[closest]
                rmsd += d[closest]**2
            rmsd = numpy.sqrt(numpy.sum(rmsd*rmsd))/float(len(v1))
            return True, rmsd
        else:
            bo,rmsd = frotator.is_superpose(self.n,v1.T,v2.T,thresh) 
            if bo== -1:
                bo=True
            else:
                bo=False
            return bo,rmsd
    
    def rotate(self,v,triple):  
        triple = numpy.array(triple,dtype='float64')
        frotator.rotate_by_triple(self.n,v.T,triple.T)
        return
    
    def rotate_matrix(self,v,rotmat):
        rotmat = numpy.array(rotmat,dtype='float64')
        v = numpy.dot(rotmat,v.T).T
        return
    
    def get_smallest(self):
        smallest = numpy.argmin(self.get_acos_mat(), axis=1).tolist() 
        return smallest
        
    def indices(self, lst, element):
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)
            
    def report(self):
        fx = numpy.zeros(len(self.stats))
        converged = 0
        for i in range(len(self.stats)):
            fx[i] = self.stats[i][2+1].fun
            if fx[i] < 0.1:
                converged += 1
            self.pprint(i, self.stats[i][1+1], self.stats[i][2+1].fun, numpy.array2string(self.stats[i][2+1].x % 1.0,precision=4),numpy.array2string(self.stats[i][2+1].x,precision=3))
        self.pprint(self.counter, numpy.array2string(numpy.array([numpy.min(fx), numpy.max(fx), \
                    numpy.mean(fx), numpy.median(fx), numpy.std(fx)]),precision=6),converged, self.timing)
        return [self.counter,converged, self.timing, numpy.min(fx), numpy.max(fx),numpy.mean(fx), numpy.median(fx), numpy.std(fx)]
    
    def bound(self,triple):
        triple=numpy.array(triple,dtype='float64')
        floor = float(numpy.floor(triple[0])) % 2.0
        triple[0:3] %= 1.0
        if floor >= 0.5: triple[0] = 1.0 - triple[0]
        return triple
    
    def calc_penalty_opt(self,triple,fact=1.0):
        self.counter += 1
        triple=self.bound(triple)
        return frotator.calc_penaltysq(self.n,self.v1n.T,self.v2n.T,self.which.T,triple.T) *fact
    
    def calc_grad(self,triple,delta=1e-6):
        triple=self.bound(triple)
        p,g = frotator.grad(self.n,self.v1n.T,self.v2n.T,self.which.T,triple.T,delta) 
        return g
    
    def calc_penalty(self, triple,which=[],fact = 1.0):
        triple=numpy.array(triple,dtype='float64')
        self.counter += 1
        if len(which) != 0:
            which = numpy.array(which,dtype='int')
            return frotator.calc_penalty(self.n,self.v1n.T,self.v2n.T,which.T,triple.T)*fact
        else:
            return frotator.calc_penalty(self.n,self.v1n.T,self.v2n.T,self.which.T,triple.T)*fact      
    
    def calc_penaltysq(self, triple,which=[],fact = 1.0):
        triple=numpy.array(triple,dtype='float64')
        self.counter += 1
        if len(which) != 0:
            which = numpy.array(which,dtype='int')
            return frotator.calc_penaltysq(self.n,self.v1n.T,self.v2n.T,which.T,triple.T)*fact
        else:
            return frotator.calc_penaltysq(self.n,self.v1n.T,self.v2n.T,self.which.T,triple.T)*fact             
        
    def optimize_rot2(self,initial_guess,method='SLSQP'):
        initial_guess = numpy.array(initial_guess,dtype='float64')
        self.opt = optimize.minimize(self.calc_penalty_opt,initial_guess,method=method,\
                    jac=False,tol=1e-6)#,options={'eps':1e-5,'gtol':1e-8,'maxiter':100}) #,'maxcor':50  'factr':1e8,
        return
    
    #@timer("optimize_rot")
    def optimize_rot(self,initial_guess,method='SLSQP'):
        #self.assign_indices()
        initial_guess = numpy.array(initial_guess,dtype='float64')
        self.opt = optimize.minimize(self.calc_penalty_opt,initial_guess,method=method,\
                    jac=None,tol=1e-10,options={'eps':1e-11})#self.calc_grad)#,options={'ftol':1e-10})#,options={'eps':1e-5,'gtol':1e-10 OR 1e-8,'maxiter':100}) #,'maxcor':50  'factr':1e8,
        return  #,bounds=[[-1.1,2.1],[None,None],[None,None]]
    
    #def grad(self,triple,which=[],fract = 1e-6):
    
    def get_rot_via_svd(self,initial_guess):
        """Obtains the rotation matrix via Singular Value Decomposition (SVD).
        !!!N.B.: NOT SURE IF STABLE!!!
        Parameter
        : initial_guess(triple): orientation triple as initial guess

        Return
        : rotmat(numpy array): shape=() orientation matrix"""
        v1,v2 = copy.deepcopy(self.v1),copy.deepcopy(self.v2)
        self.rotate(v2,initial_guess)
        
        H=numpy.zeros((3,3),dtype='float64')
        for i in range(self.n):
            H += numpy.outer(v2[self.which[i]],v1[i])
        self.pprint(H, 'H')
        [U, S, Vt] = numpy.linalg.svd(H)
        self.pprint(U, 'U')
        self.pprint(S, 'S')
        self.pprint(Vt, 'Vt')
        rotmat = numpy.dot(Vt.T,U.T).T  # results in correct solution for dot(v2,rotmat) == v1 
        self.pprint(rotmat, 'rotmat')
        return rotmat
    
    
#def find_rotation(points_distorted, points_perfect):
    #"""
    #This finds the rotation matrix that aligns the (distorted) set of points "points_distorted" with respect to the
    #(perfect) set of points "points_perfect" in a least-square sense.
    #:param points_distorted: List of points describing a given (distorted) polyhedron for which the rotation that
                             #aligns these points in a least-square sense to the set of perfect points "points_perfect"
    #:param points_perfect: List of "perfect" points describing a given model polyhedron.
    #:return: The rotation matrix
    #"""
    #isexact = True
    #for ip, pp in enumerate(points_distorted):
        #if not np.allclose(pp, points_perfect[ip]):
            #isexact = False
            #break
    #if isexact:
        #rot = np.eye(3)
        #return rot
    #H = np.zeros([3, 3], np.float)
    #for ip, pp in enumerate(points_distorted):
        #H += vectorsToMatrix(pp, points_perfect[ip])   #same as numpy.outer
    #[U, S, Vt] = svd(H)
    #rot = matrixMultiplication(transpose(Vt), transpose(U))
    #return rot


#def find_scaling_factor(points_distorted, points_perfect, rot):
    #"""
    #This finds the scaling factor between the (distorted) set of points "points_distorted" and the
    #(perfect) set of points "points_perfect" in a least-square sense.
    #:param points_distorted: List of points describing a given (distorted) polyhedron for which the scaling factor has
                             #to be obtained.
    #:param points_perfect: List of "perfect" points describing a given model polyhedron.
    #:param rot: The rotation matrix
    #:return: The scaling factor between the two structures and the rotated set of (distorted) points.
    #"""
    #rotated_coords = rotateCoords(points_distorted, rot)
    #num = np.sum([np.dot(rc, points_perfect[ii]) for ii, rc in
                  #enumerate(rotated_coords)])
    #denom = np.sum([np.dot(rc, rc) for rc in rotated_coords])
    #return num / denom, rotated_coords

    def optimize_given_which(self,which):
        self.which = numpy.array(which,dtype='int')
        if self.mpi_rank == 0:
            rot = numpy.random.uniform(0,1,3)
        else: 
            rot = None
        rot = self.mpi_comm.bcast(rot)
        self.optimize_rot(initial_guess = rot)
        while numpy.isnan(self.opt.fun) == True:
            if self.mpi_rank == 0:
                rot = numpy.random.uniform(0,1,3)
            else: 
                rot = None
            rot = self.mpi_comm.bcast(rot)
            self.optimize_rot(initial_guess = rot)
        o = self.bound(self.opt.x)
        p = self.calc_penalty(o)
        return p, o

    def get_all_orientations(self,ntrials = 3):
        """get exhaustively all the possible orientations. It works for very small coordination
        numbers and based on permutations of the assignment.
        TO CLARIFY [RA]

        Return
        : pens[-1](float): the lowest penalty
        : orientations(list of triples): shape=(N!,3)
        : which(list of lists of integers): (N!,N)
        """
        import itertools
        perms = [list(i) for i in itertools.permutations(range(self.n))]
        orientations, which, pens = [], [], []
        for p in perms:
            penalty = 10000.0
            for i in range(ntrials):
                pass
                self.which = numpy.array(p,dtype='int')
                if self.mpi_rank == 0:
                    rot = numpy.random.uniform(0,1,3)
                else: 
                    rot = None
                rot = self.mpi_comm.bcast(rot)
                self.optimize_rot(initial_guess = rot)
                if self.opt.fun < penalty:
                    best_which = p
                    best_orient= self.bound(self.opt.x)
                    penalty = self.calc_penalty(best_orient)
            which.append(best_which)
            pens.append(penalty)
            orientations.append(best_orient)
        return pens, orientations, which
    
    def screen_orientations(self,mintrials, ntrials, thresh=1.0e-3, nsteps=50, small=1.0e-3, no_rot=False):
        """TBI
        
        Parameters:
            - mintrials(int): number of trials executed, no matter if converged of not
            - ntrials(int): number of trials which are executed if zero is not reached within thresh
            - thresh(float): the threshold... [specify, RA]
            - small(float): TBA
            - no_rot (bool): defaults to False, if True no rotation is performed and only which is detected for initial position 
        
        Returns
        : lpen(float): just the lowest penalty
        : orientations(list of triples): the orientation vectors
        : which(list of lists of integers): the list of connectors associated per each orientation vector
            v_1[i] = v_2[w[i]], where v_1 and v_2 are the same vectors in the respective vector sets
            w maps the indices from vector set 1 to vector set 2.
        """
        # this part still is a mess ... [JK] will clean up soon
        lowest_pen = 1.0e12
        self.assign_indices()
        if no_rot:
            # do no rotation at all and just find which in order to connect
            which = self.which
            orient = numpy.zeros([3])
            lpen = frotator.calc_penalty(self.n,self.v1n.T,self.v2n.T,which,orient)
            return lpen, orient, which
        self.stats=[]
        i = 0
        orientations=[]
        penalties=[]
        done=False
        v2n = copy.copy(self.v2n)
        starttime = time.time()
        if self.use_sobol == True:
            rng = numpy.zeros((max([mintrials,ntrials])+5,3),dtype='float')
            for ii in range(max([mintrials,ntrials])+5):
                [rnum,seed] = sobol.i4_sobol(3,ii)
                rng[ii,:] = rnum
        while not done:
            if self.use_sobol:
                rot = self.bound(rng[i+4,:] + 0.01)
            else:
                if self.mpi_rank == 0:
                    rot = numpy.random.uniform(0,1,3)
                else: 
                    rot = None
                rot = self.mpi_comm.bcast(rot)
            self.v2n = copy.copy(v2n)
            self.rotate(self.v2n,rot)
            self.assign_indices()
            self.v2n = copy.copy(v2n)
            v2nn= copy.copy(self.v2n)
            self.optimize_rot(initial_guess=rot)
            self.rotate(v2nn,self.opt.x)
            fval = self.calc_penalty(self.bound(self.opt.x))
            i+=1
            if (i >= ntrials) or (fval <= thresh):
                ### this stuff should not be here at all [JK]
                #btw it should not stop when it finds zero... it could find different zeros!
                #i.e. you could miss the other orientations still self.opt.fun <= thresh
                #choose mintrials instead
                if i >= mintrials:
                    done=True
                    self.v2n = copy.copy(v2n)
            if fval < (lowest_pen-small):
                orientations=[]
                which = []
                lowest_pen = fval
            if abs(fval-lowest_pen)<small:
                is_new = True
                o = self.bound(self.opt.x)
                for oo in orientations:
                        if equal_orientations(o,oo):
                            is_new = False
                if is_new:
                    orientations.append(o)
                    which.append(self.which)
            if numpy.isnan(fval) == True:
                # okay that should not have happened at all, but sometimes the optimizer
                # really does weired things In this case rerun with slightly altered settings!
                logger.warning('nan encountered in function value, rerunning ... ')
                #import pdb; pdb.set_trace()
                self.use_sobol=False
                return self.screen_orientations(mintrials+1, ntrials+1, thresh=thresh, nsteps=nsteps, small=small)
        self.timing = time.time() - starttime
        starttime = time.time()
        self.timing2 = time.time() - starttime
        lpen = frotator.calc_penalty(self.n,self.v1n.T,self.v2n.T,which[0],orientations[0])

        return lpen, orientations, which
    
    def screen_step(self):
        pass
    
    def opt_sd(self):
        pass
        
    def linesearch(self):
        pass
    
    def report_timer(self):
        self.timer.write()
