# -*- coding: utf-8 -*-
from numpy import *
from . import vector
import numpy
import itertools
import time

try: 
    from . import frotator
except ImportError:
    from . import frotator3 as frotator

def equal_orientations(o1,o2,small=1.0e-2):
    dev = abs(o1-o2)
    return all(less(dev,small))

class sbu_rotator:

    def __init__(self, sbu):
        self.cxyz = sbu.connector_xyz
        self.norm_cxyz = vector.normalize(self.cxyz)
        self.counter = 0
        self.nc = len(self.cxyz)
        self.nxyz = sbu.nvertex_xyz
        self.norm_nxyz = vector.normalize(self.nxyz)
        self.l2_nxyz = vector.scal_prod(self.nxyz,self.nxyz)
        #print('c',self.cxyz)
        #print('n',self.nxyz)
        self.nvert= sbu.nvertex
        self.subsets = None
        if sbu.specific_conn:
            self.subsets = len(sbu.specific_conn)
            self.conn_sets = sbu.connectors_type
            self.vert_sets = [sbu.specific_conn.index(i) for i in self.nvert]
        self.o   = zeros([3],"d")
        self.pen = self.calc_penalty(self.o)
        #
        return

    ########## specific penalty functions ###################
    
    # Comment on this function: it will be called ONCE to get the 
    # proper connection between the closest connector and vertex
    # the result (which) will be kept
    # so only here we need to take care of special connectors (or subsets)

    def calc_penalty(self, triple):
        normcon = self.norm_cxyz.copy()
        #print("DEBUG normcon before rotate")
        #print(normcon)
        #print(triple)
        normcon = vector.rotate_by_triple(normcon, triple)
        if self.subsets:
            pen = 0.0
            which = self.nc*[0]
            for i in range(self.subsets):
                conn_ind = []
                for j,c in enumerate(self.conn_sets):
                    if c==i:
                        conn_ind.append(j)
                ncon_set = take(normcon, conn_ind, axis=0)
                vert_ind = []
                for j,v in enumerate(self.vert_sets):
                    if v==i:
                        vert_ind.append(j)
                nvert_set = take(self.norm_nxyz, vert_ind, axis=0)
                if len(conn_ind) != len(vert_ind):
                    raise ValueError("Number of connectors and vertices do not fit for special connectors")
                setsize = len(conn_ind)
                pen_set, which_set = self.calc_penalty_subset(ncon_set, nvert_set, setsize)
                #print("solved subset %d" % i)
                #print("penalty %12.6f" % pen_set)
                #print(which_set)
                pen += pen_set
                # now resolve back which connector needs to be connected to which vertex 
                # this is your riddle for today :-)
                for k in range(len(conn_ind)):
                    which[conn_ind[k]] = vert_ind[which_set[k]]
        else:
            pen, which = self.calc_penalty_subset(normcon, self.norm_nxyz, self.nc)
        return pen, which
        
    def calc_penalty_subset(self, normcon, normvert, setsize): 
        pen = 0.0
        cosa_mat = 1.0-sum(normcon[:,newaxis,:]*normvert[newaxis,:,:], axis=2)
        #cosa_mat = zeros((len(normcon),len(normvert)))
        #for xx in range(len(normcon)):
            #for yy in range(len(normvert)):
                #test[xx,yy] = dot(normcon[xx],normvert[yy])
                #cosa_mat[xx,yy] = 1.0-dot(normcon[xx],normvert[yy])
        #print('cosa_mat', cosa_mat)
        smallest = argmin(cosa_mat, axis=1)         # changed by JK from axis=1 to axis=0 to prevent crashes!
        error1 = False
        if sorted(smallest) != range(setsize):
            #print("need to fix up -> using permutations")
            smallest = smallest.tolist()
            #print("before :")
            #print(smallest)
            duplicate_j = []
            missing_j   = []
            for i in range(setsize):
                c = smallest.count(i)
                if c == 2 : duplicate_j.append(i)
                if c == 0 : missing_j.append(i)
                if c >= 3:   # .ge. instead of .gt. required here? crashes sometimes
                    error1=True
                    #test = zeros((len(normcon),len(normvert)))
                    #for xx in range(len(normcon)):
                        #for yy in range(len(normvert)):
                            #test[xx,yy] = dot(normcon[xx],normvert[yy])
                            ##print(xx,yy, dot(normcon[xx],normvert[yy]))
                    #print('---------------------------------Value Error debug print(-------'))
                    #print(argmin(test, axis=1),'argmin(test, axis=1)')
                    #print(argmin(test, axis=0),'argmin(test, axis=0)')
                    #print('normcon',normcon)
                    #print('normvert', normvert)
                    #print('setsize', setsize)
                    #print('cosa_mat', cosa_mat)
                    #print('smallest', smallest)
                    #print(' trying other axis!', smallest)
                    #raise ValueError("j appears three times .. this is bad!")
                    return self.exhaustive_i(cosa_mat)
                    #a=1.0 / numpy.sqrt(2)
                    #test2=numpy.array([[-a,-a,0.0],[a,-a,0],[-a,a,0],[a,a,0]])
                    #test1=numpy.array([[0,0,-1.0],[-0.89442719 , 0.0     ,    -0.4472136 ],[ 0.0    ,      0.0         , 1.0        ],[ 0.89442719 ,0.0,0.4472136 ]])
            if False:
                smallest = argmin(cosa_mat, axis=1)
                if sorted(smallest) != range(setsize):
                    #print("need to fix up -> using permutations")
                    smallest = smallest.tolist()
                    #print("before :")
                    #print(smallest)
                    duplicate_j = []
                    missing_j   = []
                    for i in range(setsize):
                        c = smallest.count(i)
                        if c == 2 : duplicate_j.append(i)
                        if c == 0 : missing_j.append(i)
                        if c >= 3 :   # .ge. instead of .gt. required here? crashes sometimes
                            test = zeros((len(normcon),len(normvert)))
                            for xx in range(len(normcon)):
                                for yy in range(len(normvert)):
                                    test[xx,yy] = 1.0-dot(normcon[xx],normvert[yy])
                                    #print(xx,yy, dot(normcon[xx],normvert[yy]))
                            print('---------------------------------Value Error debug -------')
                            print(argmin(test, axis=1),'argmin(test, axis=1)')
                            print(argmin(test, axis=0),'argmin(test, axis=0)')
                            print(argmin(cosa_mat, axis=1),'argmin(cosa, axis=1)')
                            print(argmin(cosa_mat, axis=0),'argmin(cosa, axis=0)')
                            print('normcon',normcon)
                            print('normvert', normvert)
                            print('setsize', setsize)
                            print('cosa_mat', cosa_mat)
                            print('smallest', smallest)
                            raise ValueError("j appears three times .. this is bad!")
                    
            # now resolve duplicates by doing all permutations of the subset
            # first identify all the critical i
            crit_i = []
            for dj in duplicate_j:
                first_i=smallest.index(dj)
                crit_i.append(first_i)
                crit_i.append(smallest[first_i+1:].index(dj)+first_i+1)
            crit_j = duplicate_j + missing_j
            size_of_problem = len(crit_j)
            if size_of_problem > 8:
                print(size_of_problem)
                raise ValueError("We stop because permutation problem above 6!!")
            pen_min = size_of_problem*2.0
            for p in itertools.permutations(crit_j):
                pen = 0.0
                #print('criti', crit_i)
                for i, j in enumerate(p): pen += cosa_mat[crit_i[i],j]
                if pen < pen_min:
                    pen_min = pen
                    p_keep = p
            # now we have to fill in
            for ii, j in enumerate(p_keep):
                smallest[crit_i[ii]] = j
            #print("after :")
           # print(smallest)
        pen = 0.0
        for i, j in enumerate(smallest): pen += cosa_mat[i,j]
        return pen, smallest

    def exhaustive_i(self,acos_mat):
        #acos_mat = self.acos_mat#1.0-numpy.sum(self.v1n[:,numpy.newaxis,:]*self.v2n[numpy.newaxis,:,:], axis=2)
        pmin = 1e10
        print('exhaustive search starting, problem size: %i' % self.nc)
        for p in itertools.permutations(range(self.nc)):
            pen=0.0
            for i, j in enumerate(p): pen += acos_mat[i,j]
            #pen = self.calc_penalty_fixed([1.0,1.0,1.0],which=p)
            if pen < pmin: pmin,p_keep =pen,p
            #print(p, pen)
        #print('best:', p_keep,pmin)
        which = numpy.array(p_keep,dtype='int')
        return pmin, which

    def calc_penalty_fixed(self, triple, which):
        self.counter += 1
        normcon = self.norm_cxyz.copy()
        normcon = vector.rotate_by_triple(normcon, triple)
        cosa = 1.0-vector.scal_prod(normcon, self.norm_nxyz[which])
        pen = sum(cosa)       
        return pen
    
    def calc_penalty_f(self,triple,which):
        #triple=self.bound(triple)
        which = numpy.array(which)
        return frotator.calc_penalty(self.nc,self.norm_nxyz.T,self.norm_cxyz.T,which.T,triple.T)

    
    ########################
    def bound(self,triple):
        triple=numpy.array(triple,dtype='float64')
        floor = numpy.floor(triple[0]) % 2.0
        triple[0:3] %= 1.0
        if floor >= 0.5: triple[0] = 1.0 - triple[0]
        return triple
    
    def calc_fgrad(self,triple,which,delta=1e-4):
        triple=self.bound(triple)
        which = numpy.array(which)
        p,g = frotator.grad(self.nc,self.norm_nxyz.T,self.norm_cxyz.T,which.T,triple.T,delta) 
        return p,g

    def calc_penalty_and_grad(self, triple, which, delt=0.0001):
        #self.counter += 1
        grad = zeros([3],"d")
        pen = self.calc_penalty_fixed(triple,which)
        o = triple.copy()
        #print("DEBUG: this is penalty grd")
        # the first coordinate of the orientational triple needs to be constrained between
        #    zero and one => close to the boundaries use single sided differences
        #                    at the boundary the gradient is set to zero
        if o[0] < delt:
            if o[0] > 0.0:
                # do singel sided FD grad at low bound
                o[0] += delt
                pp = self.calc_penalty_fixed(o, which)
                o[0] -= delt
                grad[0] = (pp-pen)/delt
        elif o[0] > (1.0-delt):
            if o[0] < 1.0:
                # do single sided FD grad at upper bound
                o[0] -= delt
                pm = self.calc_penalty_fixed(o, which)
                o[0] += delt
                grad[0] = (pen-pm)/delt
        else:
            # do the regular double sided FD grad
            o[0] -= delt
            pm = self.calc_penalty_fixed(o, which)
            o[0] += 2*delt
            pp = self.calc_penalty_fixed(o, which)
            o[0] -= delt
            grad[0] = (pp-pm)/(2*delt)
        for i in range(1,3):
            o[i] -= delt
            pm = self.calc_penalty_fixed(o, which)
            o[i] += 2*delt
            pp = self.calc_penalty_fixed(o, which)
            o[i] -= delt
            grad[i] = (pp-pm)/(2*delt)
            #print(i, pm, pp, grad[i])
        #print('g',grad)
        return pen, grad

    def line_search(self, triple, pen_init, which, direction, step):
        pen_low = pen_init
        # do a stupid line search: try step init .. if energy rises decrease
        step_init = copy(step)
        stop = False
        nred = 0
        ninc = 0
        while not stop:
            o = triple - direction*step
            # check boundary conditions
            if (o[0]<0.0) or (o[0]>1.0):
                o[0] = clip(o[0],0.0,1.0)
            o[1:3] %= 1.0
            pen_new = self.calc_penalty_fixed(o, which)
            #print("line search: pen: %12.6f step %10.5f o: %s" % (pen_new, step, str(o)))
            if pen_new < pen_init:
                stop = True
                pen_low = pen_new
            else:
                step *= 0.5
                nred += 1
                if nred > 100: stop = True
        # we found a lower penalty . now test if we can go further
        if nred == 0:
            stop = False
            while not stop:
                step *= 1.1
                ninc += 1
                o = triple - direction*step
                # check boundary conditions
                if (o[0]<0.0) or (o[0]>1.0):
                    o[0] = clip(o[0],0.0,1.0)
                o[1:3] %= 1.0
                pen_new = self.calc_penalty_fixed(o, which)
                #print("   LS inc pen: %12.6f step %10.5f o: %s" % (pen_new, step, str(o)))
                if pen_new < pen_low:
                    # found a lower penalty .. continue
                    pen_low = pen_new
                else:
                    # penalty is rising ... go back
                    step /= 1.1
                    ninc -= 1
                    stop = True
        o = triple - direction*step
        # check boundary conditions
        if (o[0]<0.0) or (o[0]>1.0):
            o[0] = clip(o[0],0.0,1.0)
        o[1:3] %= 1.0
        #print("   LS END pen %12.6f -> %12.6f (%d, %d) step %10.5f" % \
        #        (pen_init, pen_low, nred, ninc, step))
        return pen_low, step, o
                    
    def opt_sd(self, nsteps, sdstep=0.005, thresh=0.001, start_random=False):
        if start_random: self.o = random.random(3)
        stop = False
        step = 0
        pen, which = self.calc_penalty(self.o)
        #print("DEBUG: initial penalty %10.5f (assignement %s)" % (pen, str(which)))
        while not stop:
            p,g = self.calc_penalty_and_grad(self.o, which)
            #print('----')
            #print('%12.6f %12.6f %12.6f %12.6f' % (p, g[0], g[1], g[2]))
            #p,g = self.calc_fgrad(self.o,which)
            #print('%12.6f %12.6f %12.6f %12.6f' % (p, g[0], g[1], g[2]))
            #print('----')
            rmsg = sqrt(sum(g*g))
            gnorm = vector.norm(g)
            #print("iter %d  pen %10.5f rmsg %10.5f grad %s  (%s)" % \
            ##        (step,p,rmsg, array2string(g,precision=3, suppress_small=1),\
            #        array2string(self.o, precision=3, suppress_small=1)))
            #self.o -= sdstep*g
            #if (self.o[0]<0.0) or (self.o[0]>1.0):
            #    self.o[0] = clip(self.o[0],0.0,1.0)
            #    sdstep *= 0.5
            #self.o[1:3] %= 1.0
            if gnorm > 0.0:
                pline, sdstep, self.o = self.line_search(self.o, p, which, g/gnorm, sdstep)
                step += 1
            if step>nsteps: stop = True
            if rmsg<thresh: stop = True
            #if step>2:
            #    if rmsg == last_rmsg: stop=True
            #last_rmsg = rmsg
        pen, which_nv = self.calc_penalty(self.o)
        #print("iter %d  pen %10.5f rmsgrad %10.5f  (%s)" % (step,pen,rmsg, str(self.o)))
        self.pen = pen
        self.which_nv = which_nv
        return self.o, self.pen, self.which_nv
        

    def opt_MC(self, ntrials):
        lowest_pen = 1.0e12
        best_orientation = zeros([3],"d")
        for t in range(ntrials):
            orientation = random.random(3)
            pen, which_nv = self.calc_penalty(orientation)
            if pen<lowest_pen:
                lowest_pen = pen
                final_which_nv = which_nv
                best_orientation = orientation
        self.o   = best_orientation
        self.pen = lowest_pen
        self.which_nv = final_which_nv
        return best_orientation, lowest_pen, final_which_nv

    def screen_orientations(self, mintrials, ntrials, thresh, nsteps=500, small=1.0e-3):
        lowest_pen = 1.0e12
        stop = False
        step = 0
        starttime = time.time()
        fx = []
        while not stop:
            o, p, w = self.opt_sd(nsteps, start_random=True)
            fx.append(p)
            if p<(lowest_pen-small):
                #print("found lower ... discarding")
                orientations=[]
                which = []
                step = 0
                lowest_pen = p
            if abs(p-lowest_pen)<small:
                is_new = True
                #print(p, array2string(o,precision=3, suppress_small=True))
                for oo in orientations:
                    if equal_orientations(o,oo):
                        #print("equal orienations")
                        #print(o, oo)
                        is_new = False
                if is_new:
                    #print("added orientation")
                    orientations.append(o)
                    which.append(w)
            step +=1
            if step>ntrials: stop= True
            if (step>mintrials) and (lowest_pen<thresh): stop=True
        timing = time.time() - starttime
        #print('---------------------', self.counter, timing)
        self.fx = fx
        self.stats = [self.counter,self.fx,timing]
        #print("Found %d out of %d orientations with lowest penalty of %10.5f (time %8.4f s)" % \
        #            (len(orientations),ntrials,lowest_pen, timing))
        return lowest_pen, orientations,which
                
    def scan(self,initial,dim,steps):
        o = array(initial,"d")
        p, which = self.calc_penalty(o)
        for i in range(steps+1):
            val = float(i)/steps
            o[dim]= val
            p = self.calc_penalty_fixed(o,which)
            print(val, p)
        return

