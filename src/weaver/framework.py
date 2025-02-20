# -*- coding: utf-8 -*-
"""
WEAVER

    reimplemenation of the weaver code using concepts of original weaver

    major changes:

    - on ly one class (instead of hierarchy of nets) which keeps both the
      net and the real system 
    
    - mechanism of finding the proper arrangement of building blocks is
      completely changed: instead of trying to analyse the geom the BB
      is rotated iteratively into position
      
    Revision by Julian Keupp using Fortran rotator ...
    
    - some things are lost from previous versions and need to be recovered
    
    - all is so much faster now!
      
"""

from numpy import *
from scipy.linalg import norm
from string import *
import os
import sys
import copy
import numpy as np
from collections import defaultdict
from random import choice
import itertools

import molsys
import molsys.util.images as images
import molsys.util.elems as elems
from molsys.util.images import idx2arr, arr2idx

from .rotator_old import sbu_rotator
from . import rotator
from . import vector
from molsys.util import rotations

import logging
logger = logging.getLogger("molsys.framework")

import molsys.util.parutil as parutil
from molsys.molsys_mpi import mpiobject


class framework(mpiobject):
    def __init__(self, name="default_name", use_symmetry=False,mpi_comm=None, out=None,mode='atypes'):
        super(framework,self).__init__(mpi_comm, out)
        logger.info('framework instance is being created ... ')
        self.name = name
        self.net = molsys.mol()
        self.bbs = {}
        self.bblist = None
        self.pdlp = None
        self.use_symmetry=use_symmetry
        self.mode = mode
        return
        
    @classmethod
    def fromTopo(cls, t, name="fromTopo"):
        """set self.net from topo instance"""
        f = cls(name=name)
        logger.info("reading topo instance as net")
        f.net = copy.deepcopy(t)
        return f

    @classmethod
    def asBlueprint(cls, fname, supercell=[1,1,1], add_pconn=True,
        name=None):
        """set self.net from read blueprint"""
        if name is None: name = fname.split(".")[0]
        f = cls(name=name)
        f.net.read(fname)
        f.net.make_supercell(supercell)
        if add_pconn: f.net.add_pconn()
        ###TBA: read_topo for wrap_in_box???
        return f

    ### network part ###################################################      

    def read_topo(self, fname, ftype = None, report=False):
        """read topo-type mfpx file (cf. mfpx file specifications)
        TBI: link to mfpx files
        fname (str):
        ftype (str):
        report (bool): """
        self.net.read(fname,ftype=ftype)
        #self.net.wrap_in_box()
        if report: self.net.report_conn()
        return
        
    def scale_net(self, scale):
        """The default way of storing topologies is to have vertex 
        distances equal to 1 (keyword: maximum-symmetry embedding or
        barycentric embedding).
        In order to manually scale the nets, to fit the lenght of the
        BBs this function is used.
        scale (float or iterable of floats with len 3): 
        if float: isotropic scaling
        if iterable: anisotropic scaling
            (of each cell vector with the respective factor)
        TBA: clearer description"""
        if self.net.periodic:
            self.net.scale_cell(scale)
        if scale is None: scale = [1,1,1]
        if hasattr(scale, '__iter__'):
            scalefmt = ''.join([' %8.4f' for s in scale])
            scaleprt = tuple(scale)
        else:
            scalefmt = ' %8.4f'
            scaleprt = (scale,)
        logger.info('net scaled by factor(s) ' + scalefmt % scaleprt)
        if self.net.periodic:
            logger.info('new scaled cell is \n%s' % (np.array2string(self.net.get_cell(),precision=2),) )
        return
    
    def autoscale_net(self,fiddle_factor=1.5):
        """automatically scales the blueprint
        
        When a raw blueprint is used whose edge distances do not correspond to the distances between
        the building blocks, this function can be used to automatically compute a resonable scaling factor.
        Isotropic version. does not work well when anisotropic scaling is required.
            fiddle_factor (float, optional): Defaults to 1.5. [The fiddling factor is the average length of the bonds that
            are to be formed in the construction process
        """
        # compute the average edge distance
        logger.info('autoscaling net ...')
        dcount, d = 0,0.0
        for i,cc in enumerate(self.net.conn):
            for j,c in enumerate(cc):
                d += self.net.get_distvec(i,c)[0]
                dcount += 1
        dnetavg = d / float(dcount)
        print('dnetavg',dnetavg) 
        # compute the average center-connector distance
        connector_distances = 0.0
        for i,bbkey in enumerate(self.bbs.keys()):
            bb = self.bbs[bbkey]
            connector_distances += np.mean(np.linalg.norm(bb.bb.connector_xyz, axis=1))
        connector_distances /= float(i+1) # i+1 = number of building blocks
        scaling_factor = (2.0*connector_distances+fiddle_factor) / dnetavg
        self.pprint(('scaling_factor: %f' % scaling_factor))
        logger.info('scaling_factor: %f' % scaling_factor)
        self.scale_net(scaling_factor)
        
    def autoscale_net_anisotropic(self, connections):
        """Autoscale the net to fit the framework blocks
        *** DEVELOPMENT IN PROGRESS ***"""
        oindex = self.framework.orientation
        rdists = []
        bdists = []
        for itbc in connections:
            va, vb, ca, cb, wa, wb = itbc
            bba, bbb = self.bblist[va], self.bblist[vb]
            oa, ob = oindex[va], oindex[vb]
            ### NET FACTOR ###
            dist = self.net.get_distvec(va, vb)
            if 13 not in dist[2]:
                continue
            rdist = dist[1]
            rdists.append(rdist)
            ### BLOCK FACTOR ###
            acoc = bba.bb.get_coc()
            aconn = bba.bb.connector_xyz[:,0,:]
            ooa = bba.orientations[oa]
            ae = bba.elems[bba.bb.connector[wa][0]] ### multiple connections broken
            acov = elems.cov_radii[ae]
            ar = np.mean(aconn[wa] - acoc, axis=0) #if multiple connectors
            ar = rotations.rotate_by_triple(ar,ooa)
            ar *= acov/norm(ar) + 1
            ar = rdist*np.dot(ar,rdist)/np.dot(rdist,rdist)
            bcoc = bbb.bb.get_coc()
            bconn = bbb.bb.connector_xyz[:,0,:]
            oob = bbb.orientations[ob]
            be = bbb.elems[bbb.bb.connector[wb][0]] 
            bcov = elems.cov_radii[be]
            br = np.mean(bconn[wb] - bcoc, axis=0)
            br = rotations.rotate_by_triple(br,oob)
            br *= bcov/norm(br) + 1
            br = rdist*np.dot(br,rdist)/np.dot(rdist,rdist)
            bdist = ar - br
            logger.debug("%.2f+%.2f=%.2f vs. %.2f" % (norm(ar), norm(br), norm(ar)+norm(br), norm(bdist)))
            bdists.append(bdist) ###BROKEN FOR MULTIPLE CONNECTORS
        rdists = np.array(rdists)
        rsum = np.sum(abs(rdists),axis=0)
        bdists = np.array(bdists)
        bsum = np.sum(abs(bdists),axis=0)
        scale = np.max(abs(bdists), axis=0)
        logger.debug("Automatic net scaling factor is: %.3f %.3f %.3f" % tuple(scale))
        self.scale_net(scale)
        return

    def interpenetrate_net(self, frac_vect=[[0.5, 0.5, 0.5]]):
        """ generate an interpenetrated system
            a list(!) of translation vectors (N,3) must be provided
            defining the times of interpentration by the number (N) of vectors
        """
        for tv in frac_vect:
            new_net = copy.deepcopy(self.net)
            shift = sum(self.net.cell*(np.array(tv)[np.newaxis,:]), axis=0) #basically a dot product
            logger.info("\nInterpenetrating network with a copy of the net shifted by %s" % shift)
            self.net.add_mol(new_net, translate=shift)
        self.net.wrap_in_box()
        return
        
    ### BB catalogue ##################################################    
        
    def assign_bb(self, vertex, fname, ftype='mfpx',mol=None, specific_conn=None, linker=False, zflip=False, nrot=1, use_sobol=True, no_rot=False,target_norients=None, rmsdthresh = None, vertex_type = 'symmetry'):
        """Assign bb blocks based on vertex types as defined in the topo-type mfpx file
        Parameters:
            - vertex(string or iterable of strings): 
            - fname(str): bb-type mfpx file name: it must contain the information on the connecting atoms (LINK TBA)
            - specific_conn (list of couples): if specific_conn is not None, then it is a list of vertex types of the length of the connector types.
                   The first connector binds only to the first vertex type etc.
            -linker(bool): if True: just one vector is aligned TBA: it takes further discussion
            - zflip(bool): if True: a linker is flipped on z-axis (the axis which connectors belong to)
                this is for asymmetric linker units
            - nrot(int): number of rotations around the z-axis for a linker bb
            - use_sobol(bool): by default the same pseudorandom numbers are used for the rotation detection. Turn off to get random orientations.
            - no_rot(bool): do not rotate at all
            - target_norients: if not None, the number of orientations to be considered in a screening
            - rmsdthresh: if not None, use this as the bb rmsd thresh in screening for inequivalent structures
            - vertex_type: defines which vertex label is used for the weaving process, 'elem': the element symbol, 'atype': the atomtype column
                this is usually the symmetry label, at least if the topo is RCSR derived, or 'systre', which is the systre key labels. 
                In case of 'systre' the topo file has to be of 'topo_format new' type, where the systrekey is actually given along with the 
                systre vertex - vertex mappings as the additional column in the new format.
        """
        vertex = vertex.lower()
        if mol is None:
            self.bbs[vertex] = molsys.mol()
            self.bbs[vertex].addon('bb')
            self.bbs[vertex].read(fname,ftype=ftype)
        else:
            self.bbs[vertex] = mol
        self.setup_bb(self.bbs[vertex],name=fname.split('.')[0],nrot = nrot,zflip = zflip,linker = linker,
                specific_conn=specific_conn, label = vertex, no_rot=no_rot,use_sobol = use_sobol, 
                target_norients = target_norients,rmsdthresh = rmsdthresh)
        return

    def setup_bb(self,bbmol,**kwargs):
        bbmol.bb.__dict__.update(kwargs)
        return
    
    def assign_bb_auto(self, fname,ftype='mfpx', specific_conn=None, linker=False, zflip=False, nrot=1):
        """if specific_conn is given, then it is a list of vertex types of the length
           of the connector types. The first connector binds only to the first vertex
           type etc """
        m = molsys.mol()
        m.addon('bb')
        m.read(fname,ftype=ftype)
        self.setup_bb(m,name=fname.split('.')[0],nrot = nrot,zflip = zflip,linker = linker,
                   specific_conn=specific_conn, label = 'none')
        nconn  = len(m.bb.connector)
        vtypes = list(set(self.net.atypes))
        cn     = [len(self.net.conn[self.net.atypes.index(i)]) for i in  vtypes]
        self.pprint(vtypes,cn)
        for vt,c in zip(vtypes,cn):
            if c == nconn:
                self.assign_bb(vt,fname,ftype=ftype,specific_conn=specific_conn,linker=linker,zflip=zflip, nrot=nrot)
        return

    def assign_bbs(self, couples, **kwargs):
        """Method for multiple bbs, not really needed after.
        :Example:
            f.assign_bbs([
                ('0', 'btc.mfpx'),
                ('1', 'CuPW.mfpx'),
            ])
        """
        assert hasattr(couples, "__iter__"), "Give ((v1, f1),(v2,f2),...,(vn,fn)) iterable\
            \n    where vi, fi are vertex and fname, respectively"
        for vertex, fname in couples:
            self.assign_bb(vertex, fname, **kwargs)

    def add_linker_vertex(self, label, between=None):
        """ adds a new vertex as a 2coord linker bb either between all vertices or just between
        the specified vertex types (like a bond)
        between (None or couple of strings): if None: a vertex is added between any bounded couple
        of vertices; if couple of strings: a vertex is added between any bounded couple of vertices
        with the respective vertex types"""
        add_atoms = []
        for i in range(self.net.natoms):
            conni = self.net.conn[i]
            for ci in range(len(conni)):
                j = conni[ci]
                if j>=i:
                    add = True
                    if between:
                        add = False
                        alist = sorted([self.net.atypes[i], self.net.atypes[j]])
                        if sorted(between) == alist:
                            add = True
                else:
                    add = False
                if add:
                    xyz_j = self.net.get_neighb_coords(i, ci)
                    logger.debug("neighb dist", vector.norm(self.net.xyz[i]-xyz_j)\
                           , vector.norm(self.net.xyz[i]-self.net.xyz[j]))
                    xyz = (self.net.xyz[i]+xyz_j)/2.0
                    add_atoms.append((xyz,i,j))
        # now add all these atoms to the system
        logger.info("adding %d linker BBs" % len(add_atoms))
        ### note (JK): for 1x1x1 pcu this crashes due to there is too mcuh atoms to be inserted, needs to be fixed!
        ### quick hack: at this point use:
        ###     add_atoms = [add_atoms[0]]+[add_atoms[2]]+[add_atoms[4]] 
        ### to remove them
        for a in add_atoms:
            xyz, i, j = a
            logger.debug("inserting between %d and %d at %s" % (i, j, str(xyz)))
            #self.net.insert_atom('na',label.lower(),xyz,i,j)
            self.net.insert_atom('na',label.lower(),i,j,xyz=xyz)
        return
                
    def add_linker_vertex_(self, label, between=None):
        """ PCONN-AWARE adds a new vertex as a 2coord linker bb either between all vertices or just between
        the specified vertex types (like a bond)
        between (None or couple of strings): if None: a vertex is added between any bounded couple
        of vertices; if couple of strings: a vertex is added between any bounded couple of vertices
        with the respective vertex types"""
        ### note (RA): tentative fix for 1x1x1 pcu
        add_atoms = []
        raise NotImplementedError
        if between is None:
            for bond in self.net.etab:
                ei, ej, p = bond
                xyz_j = self.net.get_neighb_coords_(ei,ej,idx2arr[p])
                logger.debug("neighb dist",
                    vector.norm(self.net.xyz[ei]-xyz_j),
                    vector.norm(self.net.xyz[ei]-self.net.xyz[ej])
                )
                xyz = .5*(self.net.xyz[ei]+xyz_j)
                add_atoms.append((xyz, bond))
        else:
            raise NotImplementedError("TBI!!!")
        # now add all these atoms to the system
        logger.info("adding %d linker BBs" % len(add_atoms))
        for a in add_atoms:
            raise NotImplementedError
            xyz, bond = a
            logger.debug("inserting between %d and %d in img %d at %s" % (bond[0], bond[1], bond[2], str(xyz)))
            self.net.insert_atom_('na',label.lower(),bond,xyz=xyz)
        return
                
    def weight_vertex(self, label, weights, shuffle=True):
        """ Substitutes vertices with derived labels, in place of the
        parent vertex with label, e.g. '1' -> '1a', '1b'.
        Useful for the insertion of defects by design.
        :Parameters:
        - label (str)               - the label of the net
        - weights (list of reals)   - concentration of siblings
        - shuffle (bool)            - enables position shuffling"""
        l = [i for i, el in enumerate(self.net.atypes) if el == label]
        if shuffle: self.FisherYatesKnuth_shuffle(l)
        N = len(l)
        w = weights
        W = sum(w) #weights sum
        f = [wi*N / float(W) for wi in w] #frequencies
        p = [int(fi)/1 for fi in f] #occurences
        r = [fi%1 for fi in f] #remainder
        cr = np.cumsum(r) #cumulative remainders
        for i in range(N-sum(p)):
            if self.mpi_rank == 0:
                rnd = random.random()
            else:
                rnd = 0
            rnd = self.mpi_comm.bcast(rnd)
            p[ self.choose(rnd*cr[-1], cr) ]+=1
        cp = np.cumsum(p)
        asc = list(string.ascii_lowercase)
        for i in range(len(w)):
            if i != 0:
                ll = l[cp[i-1]:cp[i]]
            else:
                ll = l[:cp[i]]
            for j in ll:
                self.net.atypes[j] = label + asc[i]

    ### utils #########################################################

    def choose(self, target, thresholds, n=0):
        ''' Choice based on thresholds. Similar to numpy.piecewise.
        Return index 0 if:
            target < thresholds[0]
        else return n if:
            thresholds[n-1] <= target < thresholds[n]
        It best fits random choices.
        N.B.: target < thresholds[-1] must hold!
        :Parameters
        - target (real)             - the given number against n is chosen
        - thresholds (list of real) - strictly increasing thresholds
        - n (int)                   - non-negative index, for recursion.'''
        if target < thresholds[n]:
            return int(n)
        else:
            return choose(target, thresholds, n+1)

    def FisherYatesKnuth_shuffle(self, items):
        ''' Shuffles an item according to Fisher-Yates algorithm (Knuth implementation).
        All the n! permutations are accessible.
        For further information:
        http://eli.thegreenplace.net/2010/05/28/the-intuition-behind-fisher-yates-shuffling/ '''
        i = len(items)
        if self.mpi_size > 1: self.pprint('NOT WORKING IN PARALLEL')
        while i > 0:
            j = random.randrange(i)
            logger.debug(j, i-1)
            items[j], items[i-1] = items[i-1], items[j]
            i = i -1
        return


    ### true BBs ######################################################
    
    def generate_bblist(self, use_elems=False):
        """Copies n times the building blocks and stores it in the bblist.
        and gets the vertex types and vertex positions of connected
        vertices per vertex.
        """
        logger.info("generating list of building blocks")
        self.bblist = []
        self.norientations = []
        natoms = 0
        if use_elems is True: # for backward compatibility
            self.mode = 'elems'
        if self.mode == 'elems':
            vertex_labels = self.net.elems
        elif self.mode == 'atypes':
            vertex_labels = self.net.atypes
        elif self.mode == 'systre':
            vertex_labels = [str(x+1) for x in self.net.topo.skey_mapping]
        for i in range(self.net.natoms):
            v = vertex_labels[i]
            bb = copy.deepcopy(self.bbs[v])
            #bb.hide_dummy_atoms()
            # test if the connectivity of the vertex fits with the bb
            if len(bb.bb.connector) == 1:
                bb.bb.single = True
            elif len(self.net.conn[i]) != len(bb.bb.connector):
                raise IOError("BB %d has the wrong connectivity %d for vertex %s with coordination number %d"\
            % (i, len(bb.bb.connector), v, len(self.net.conn[i])))
            else: 
                bb.bb.single = False
            # now attach some extra data to this object for later use in orienting it
            nvertex     = []
            nvertex_xyz = []
            for ci in range(len(self.net.conn[i])):
                n = self.net.conn[i][ci]
                # symbol of neighbor vertex (for special connections)
                #nvertex.append(self.net.elems[n])
                nvertex.append(vertex_labels[n])
                # position of neighb vertex
                nvertex_xyz.append(self.net.get_neighb_coords(i,ci).tolist())
            # make an array and shift to have the bbs vertex in the origin
            nvertex_xyz = np.array(nvertex_xyz,"d")-self.net.xyz[i]
            bb.bb.nvertex = nvertex
            bb.bb.nvertex_xyz = nvertex_xyz
            # append the atom offset per each bb for atom indexing
            bb.bb.atom_offset = natoms
            natoms += bb.natoms
            bb.bb.index = i
            self.bblist.append(bb)
            self.norientations.append(0)
        return
    
    def get_permuted_orientations(self,base_which,bbidx):
        if not self.bblist: self.generate_bblist()
        for bbindex, bb in enumerate(self.bblist):
            if bbindex != bbidx: continue
            s = bb
            all_o, all_w = [],[]
            for i in range(len(base_which)):
                if i==0 :
                    current_which = base_which
                else:
                    current_which = current_which[1:]+[current_which[0]]
                self.pprint('current_which:', current_which)
                rot = rotator.rotator(s.nvertex_xyz,s.bb.connector_xyz,bbinstance=s,use_sobol = s.bb.use_sobol)
                p, o = rot.optimize_given_which(current_which)
                all_o.append(o)
                all_w.append(copy.copy(current_which))
            s.rot = rot
            s.orientations = all_o
            s.which        = all_w
            s.penalty      = p
            self.norientations[bbindex] = len(all_o)
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return
    
    def get_all_orientations(self):
        """Exhaustive method to get all the possible orientations.
        It permutes range(natoms) and use it as assignment.
        You can use it just for very small coordination numbers.
        """
        if not self.bblist: self.generate_bblist()
        logger.info("permuting assignment to get all possible orientations") 
        for sindex, s in enumerate(self.bblist):
            self.pprint('sindex:', sindex, s.name)
            rot = rotator.rotator(s.nvertex_xyz,s.bb.connector_xyz,bbinstance=s,use_sobol = s.bb.use_sobol)
            p, all_o, all_w = rot.get_all_orientations()
            s.rot = rot
            s.orientations = all_o
            s.which        = all_w
            s.penalty      = p
            self.norientations[sindex] = len(all_o)
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return
    
    def get_target_orientations(self,rmsdthresh=0.1,interactive=False):
        """
        """
        if not self.bblist: self.generate_bblist()            

        for sindex, s in enumerate(self.bblist):
            self.pprint('sindex:', sindex, s.name)
            rot = rotator.rotator(s.bb.nvertex_xyz,s.bb.connector_xyz,bbinstance=s,use_sobol = s.bb.use_sobol)
            p, all_o, all_w = rot.get_all_orientations()
            pargsorted = np.argsort(p)
            p, all_o, all_w = [p[x] for x in pargsorted], [all_o[x] for x in pargsorted], [all_w[x] for x in pargsorted]
            s.rot = rot

            bb_rmsdthresh = s.bb.rmsdthresh if s.bb.rmsdthresh is not None else rmsdthresh
            asymm_p,asymm_o,asymm_w = self.detect_atomistic_equivalence(p,all_o,all_w,s,bb_rmsdthresh,report=False)
            #import pdb; pdb.set_trace()
            self.pprint('unique orientations:')
            for i in range(len(asymm_p)):
                self.pprint('%3d %8.6f %s %s' % (i,asymm_p[i],str(asymm_o[i]),str(asymm_w[i])))
            if interactive is True:
                from builtins import input
                # let the user decide per vertex which ones to choose.
                self.open_vmd(sindex,asymm_p,asymm_o,asymm_w)
                choice = eval(input('choose orientations by frame index (comma separated) (blank for first n!)'))
                if choice  == '':
                    p_choice = asymm_p[0:s.bb.target_norients]
                    o_choice = asymm_o[0:s.bb.target_norients]
                    w_choice = asymm_w[0:s.bb.target_norients]
                else:
                    choice = [int(x) for x in choice.split(',')]
                    p_choice = [asymm_p[x] for x in choice]
                    o_choice = [asymm_o[x] for x in choice]
                    w_choice = [asymm_w[x] for x in choice]
                pass
            else:
                if s.bb.target_norients is not None:
                    p_choice = asymm_p[0:s.bb.target_norients]
                    o_choice = asymm_o[0:s.bb.target_norients]
                    w_choice = asymm_w[0:s.bb.target_norients]
            s.bb.orientations = o_choice
            s.bb.which        = w_choice
            s.bb.penalty      = p_choice
            self.norientations[sindex] = len(o_choice)
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return

    def open_vmd(self,bbidx,p,o,w):
        bb = self.bblist[bbidx]
        from molsys.fileIO import lammpstrj
        n = len(o)
        vmdscript = ''
        structfile = open('orients_%d.lammpstrj' % (bbidx,),'w')
        vmdscript += '''
mol delrep 0 0
mol color Name
mol representation DynamicBonds 1.600000 0.100000 30.000000
mol selection all and not name Y V B F I P X
mol material Opaque
mol addrep 0
mol color Name
mol representation VDW 0.3000000 30.000000
mol selection all and name Y V B F I P X
mol material Opaque
mol addrep 0
'''

        
        
        for i in range(n):
            m = molsys.mol()
            m.add_mol(bb,rotate=o[i])
            m.center_com()
            m.translate(self.net.xyz[bbidx])
            m.set_cell(self.net.get_cell()*10,cell_only=True)
            center = m.add_atom('X','x1',self.net.xyz[bbidx])
            #center = m.add_atom('X','x1',np.array([0.0,0.0,0.0]))
            #import pdb; pdb.set_trace()
            for j,c in enumerate(self.net.conn[bbidx]):
                pconn = self.net.pconn[bbidx][j]
                xyz = self.net.get_image(self.net.xyz[c],pconn)
                v = xyz - self.net.xyz[bbidx]
                v *= (np.mean(bb.conn_dist) +1.0) / np.linalg.norm(v)
                v = self.net.xyz[bbidx] + v
                newtype = ['Y','V','B','F','I','P'][j]
                newatom = m.add_atom(newtype,'xx',v)
                vmdscript += '''
mol color Name
mol representation DynamicBonds 20.000000 0.200000 30.000000
mol selection index %d %d
mol material Opaque
mol addrep 0
''' % (center,newatom)
                connector_xyz = self.net.xyz[bbidx] + rotations.rotate_by_triple(bb.bb.connector_xyz[w[i][j]],o[i])
                newatom2 = m.add_atom(newtype,'xx',connector_xyz)
                vmdscript += '''
mol color Name
mol representation DynamicBonds 20.000000 0.200000 30.000000
mol selection name %s
mol material Opaque
mol addrep 0
''' % (newtype)
                #import pdb; pdb.set_trace()
                m.add_bond(center,newatom)
            lammpstrj.write(m,structfile)
        structfile.close()
        f = open('vmdinput_%d.tcl' % (bbidx,),'w')
        f.write(vmdscript); f.close()
        os.system('vmd -lammpstrj '+ 'orients_%d.lammpstrj -e vmdinput_%d.tcl > /dev/null 2>&1' % (bbidx,bbidx))
        return

    def scan_orientations(self, mintrials, ntrials=20, thresh=0.01,rmsdthresh=0.2,small=0.001):
        """Stochastic method to get a reasonable sample of possible orientations.
        optimization of vector superposition
        per each trial a random orientation, rotate the second vector set, which vector belongs to which vector
        then linear sum assignment problem
        and then with this fixed assignment you optimize the rotation and in the end you get the penalty p.
        and now the rotator stores the p for any trial and in the end olny keeps those which have the smallest p, but since it is numerical optimization the p can of course vary by some points (.1, .05, etc.)
        in order to dedice where to cut we define the thresh.
        Usually when you have ui.e. btc you can get three different penalties which are close to zero and the superposition will tell you at the end that they are all equal

        sorts the penalties from smallest to greatest
        picks the orientations with penalty close to the smallest according to thresh(old)

        per each of the picked lowest-penalty orientations, the atomistic structure is rotated keeping
        the same center of rotation. Those rotated structures are superposed each other to find the structurally
        distinct ones according to the RMSD algorithm. If they are within the rmsdthresh  each other, they are
        considered equal and just one rotated structure is kept.
        if it is only one, it will be the only one in the output
        (if it is two, then we need some method to tell which one, e.g. a screening)
        [NEEDS POLISHING, RA]

        mintrials(int): number of minimum trials to be executed
        ntrials(int):
        thresh(float): threshold to decide as to whether an AAD value is considered equal to another
        rmsdthresh(float): threshold to decide was th whether an rmsd value is considered equal to one another
        small(float):"""
        if not self.bblist: self.generate_bblist()
        logger.info("scanning orientations using %d trials" % mintrials)
        logger.info("using fortran rotator")
        #self.bblist = parutil.scatter(self.bblist)
        #self.norientations = parutil.scatter(self.norientations)
        for bindex, bb in enumerate(self.bblist):
            self.scan_bb(bindex, bb, mintrials, ntrials=ntrials, thresh=thresh, rmsdthresh=rmsdthresh, small=small)
        #self.bblist = parutil.finish(self.bblist)
        #self.norientations = parutil.finish(self.norientations)
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return

    def scan_bb(self, bindex, bb, mintrials, ntrials, thresh, rmsdthresh, small):
        ###N.B: bindex PER RANK, bb.index GLOBALLY###
        if bb is None:
            return
        if bb.natoms == 1:
            bb.bb.orientations = [np.array([0.0,0.0,0.0])]
            bb.bb.which        = [list(range(len(bb.bb.connector_xyz)))]
            bb.bb.penalty      = 0.0
            self.norientations[bindex] = 1
            return
        rot = self.bb_rotator(bb)
        #if bb.no_rot == True:
        if bb.bb.no_rot == True:
            #import pdb; pdb.set_trace()
            p = rot.assign_indices() # updates rot.which
            o = np.zeros([3])
            w = rot.which
            all_o, all_w = [o],[w]
            bb.bb.orientations = [o]
            bb.bb.penalty = p
            bb.bb.which = [w]
            self.norientations[bindex] = 1
            if bb.bb.zflip: self.norientations[bindex] *= 2
            #import pdb; pdb.set_trace()
            #return
        elif bb.bb.single == True:
            p, all_o, all_w = rot.screen_orientations(1, 1, 1.0)
            #return
        elif bb.bb.linker:
            p, all_o, all_w = rot.screen_orientations(2,2,small=1.0e-12)
            logger.debug('linker', all_o,all_w)
        else:
            p, all_o, all_w = rot.screen_orientations(mintrials, ntrials, thresh, small=small)
        bb.bb.rot = rot
        p, keep_o, keep_w = self.detect_atomistic_equivalence(p, all_o, all_w, bb, rmsdthresh)
        if bb.bb.linker:
            keep_o = [keep_o[0]]
            keep_w = [keep_w[0]]
        if bb.bb.no_rot:
            keep_o = [(1.0, 0.0, 0.0)]
            keep_w = [keep_w[0]]
        bb.bb.orientations = keep_o
        bb.bb.which        = keep_w
        bb.bb.penalty      = p
        if bb.bb.linker:
            self.norientations[bindex] = 1
            if bb.bb.zflip: self.norientations[bindex] *= 2
        elif bb.bb.no_rot == True:
            pass
        else:
            self.norientations[bindex] = len(keep_o)
        self.pprint(("BB %3d ( %-10s) # orientations: %3d penalty %10.5f" % (bb.bb.index, bb.bb.name, self.norientations[bindex], p)))
        return

    def detect_atomistic_equivalence(self,p, all_o, all_w, bb, rmsdthresh,report=False):
        keep_o = [all_o[0]]
        keep_w = [all_w[0]]
        keep_p = [p[0]] if type(p)  ==  type([1]) else [p] 
        for i in range(1,len(all_o)):
            o = all_o[i]
            is_new = True
            v1 = copy.copy(bb.xyz)
            rot2 = rotator.rotator(v1,v1)
            rot2.rotate(rot2.v1,o)
            v2 = copy.copy(bb.xyz)  #####################work here ##################
            for j in range(len(keep_o)):
                oo = keep_o[j]
                # now we need to test if the bb is equal in orientation o and oo
                rot2.v2 = copy.copy(v2)
                rot2.rotate(rot2.v2,oo)
                equal, rmsd = rot2.is_superpose(rot2.v1,rot2.v2, thresh=rmsdthresh,use_fortran=True)
                if report is True: self.pprint('%d %d %.4f, %.2f' % (i,j,rmsd, rmsdthresh))
                if rmsd <= rmsdthresh:
                    equal = True
                    is_new = False
                    logger.debug("all_o: %d with keep_o: %d: rmsd=%10.5f" % (i,j,rmsd))
                    break
            if is_new:
                keep_o.append(all_o[i])
                keep_w.append(all_w[i])
                try:
                    keep_p.append(p[i])
                except:
                    keep_p.append(p)
        if type(p)  ==  type([1]):
            return keep_p, keep_o, keep_w
        return p, keep_o, keep_w

    def detect_symmetry(self):
        """Detect symmetry of the topology in order to use it for faster orientational screening
        [description]
        """
        self.use_symmetry=True
        self.net.addon('spg')
        spg = self.net.spg 
        spg.generate_spgcell()
        spg.generate_symmetry_dataset()
        self.spg = spg

    def scan_orientations_symmetry(self, mintrials, ntrials=20, thresh=0.01,rmsdthresh=0.2,small=0.001):
        if not self.use_symmetry:   self.detect_symmetry()
        if not self.bblist:         self.generate_bblist()
        logger.info("scanning orientations using %d trials" % mintrials)
        spg = self.net.spg
        for i,e in enumerate(spg.base_indices):
            bindex = e
            bb = self.bblist[e]
            self.scan_bb(bindex, bb, mintrials, ntrials=ntrials, thresh=thresh, rmsdthresh=rmsdthresh, small=small)
            for j,d in enumerate(spg.derived_indices[i]):
                if d == e: continue
                self.copy_rotation_from_symmetry(e,d)
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return
    def copy_rotation_from_symmetry(self,root_bbidx,bbidx):
        root_bb = self.bblist[root_bbidx]
        bb = self.bblist[bbidx]
        rot = self.bb_rotator(bb)
        bb.rot = rot
        bb.orientations = root_bb.orientations
        bb.penalty = root_bb.penalty
        bb.which = []
        ### which has to be set up properly.
        for i,o in enumerate(bb.orientations):
            rot2 = copy.deepcopy(rot)
            rot2.rotate(rot2.v1n, o)
            rot2.v2n = np.dot(self.net.spg.transformations[bbidx][0],rot2.v1n.T).T ### CHECK THIS [RA]
            rot2.assign_indices()
            bb.which.append(rot2.which)
        self.norientations[bbidx] = len(bb.orientations)
        return

    def mask_orientations(self,orientations):
        """Get orientations mask for the number of orientations list 
        
        [description]
        """
        for i in range(len(self.norientations)):
            if self.norientations[i] == 1:
                orientations[i] = 0 
        return orientations

    def sym_orientations_try(self, use_naive=True):
        """Symmetry method [RA]"""
        if not self.bblist: self.generate_bblist()
        if use_naive:
            self.sym_orientations_naive()
        else:
            self.sym_orientations_smart()
        raise NotImplementedError("This method must be implemented by RA!!!")
        return

    def sym_orientations_naive(self):
        raise NotImplementedError("This method must be implemented by RA!!!")
        #for bindex, bb in enumerate(self.bblist):
        #    import pdb; pdb.set_trace()
        return

    def sym_orientations_smart(self):
        raise NotImplementedError("This method must be implemented by RA!!!")
        return

    def sym_orientations(self):
        "Simmetries blocks in pockets to get all the possible orientations"
        if not self.bblist: self.generate_bblist()
        self.pprint("GENERATE ORIENTATIONS BY SYMMETRY")
        try:
            import spglib
        except ImportError:
            raise ImportError("spglib not available, install via 'sudo pip install spglib'")
        for bindex, bb in enumerate(self.bblist):
            p = molsys.mol.fromArray(bb.nvertex_xyz)
            self.scan_unequivalents(p,bb)

    def rowsort(self,a):
        """Sorts by rows keeping rows (i.e. without messing up with coordinates)
        :Parameter:
            -a(np.array o np.array-able iterable): 3,N matrix to be sorted
        :Return:
            -asort(same as a): sorted matrix"""
        try:
            view = a.view('f8,f8,f8')
        except AttributeError:
            view = np.array(a).view('f8,f8,f8')
        asort = np.sort(view, kind="mergesort", order=['f0','f1','f2'], axis=0).view(np.float64)
        return asort

    def scan_unequivalents(self, molp, molb):
        """p: pocket; b:block"""
        cell = 1.2*abs(molp.xyz).max(axis=0)+0.2
        import pdb; pdb.set_trace()
        Tp = molp.xyz.mean(axis=0)
        molp.xyz -= Tp
        molp.set_cellparams(list(cell) + [90., 90., 90.])
        molp.addon('spg')
        molp.spg.generate_spgcell()
        lrot, ltra, leqa = molp.spg.get_symmetry()
        syms = list(zip(lrot, ltra)) ###no equivalent positions
        xyzb = copy.deepcopy(molb.xyz)
        Tb = molb.xyz.mean(axis=0)
        molb.xyz -= Tb
        import pdb; pdb.set_trace()
        uneqs = [self.rowsort(xyzb)]
        for sym in syms:
            new_xyzb = (np.dot(xyzb,sym[0]) + sym[1])
            new_xyzb = self.rowsort(new_xyzb)
            check = np.where((new_xyzb == uneqs).all(axis=1), True, False)
            if not check.all(axis=1).any():
                uneqs.append(new_xyzb)
        return uneqs

    def clone_orientations(self):
        pass

    def orientations_to_string(self,version="1.1"):
        """
        Method to write to a string.
        """
        buffer = ""
        buffer += "%5d  orientations version %s\n" % (self.net.natoms,version) 
        for si, s in enumerate(self.bblist):
            no = self.norientations[si]
            if no==0: no=1
            if s.bb.linker : no = 1
            nc = len(s.bb.connector)
            if version == '1.1': buffer+="%-5d %-5s %-20s %2d %2d %10.5f\n" % (si, s.bb.label, s.bb.name, nc, no, s.bb.penalty)
            if version == '2.0': buffer+="%-5d %-5s %-20s %2d %2d %10.5f\n" % (si, s.bb.label, s.bb.name, nc, no, np.mean(s.bb.penalty))
            for oi, o in enumerate(s.bb.orientations):
                w = s.bb.which[oi]
                ostring = (3*"%10.5f ") % tuple(o)
                ostring += "   |   "
                ostring += (nc*"%2d ")%tuple(w)
                if version == '2.0': ostring += ' %.4f'  % (s.bb.penalty[oi],)
                ostring += "\n"
                buffer+=ostring
        return buffer

    def write_orientations(self, fname,version="1.1"):
        """Writes the orientations file (cf. generic orientation files where everything is explained in such a
        way it is easily understandable).
        Instead of recomputing the orientations you can read the files later."""
        logger.info(("WRITING ORIENTATIONS TO FILE %s") % fname)
        if self.mpi_rank == 0:
            with open(fname, "w") as f:
                f.write(self.orientations_to_string(version=version))
        return
        
    def read_orientations(self, fname):
        """Reads orientations from orientations file without recomputing them."""
        if not self.bblist: self.generate_bblist()
        logger.info(("READING ORIENTATIONS FROM FILE %s" % fname))
        of = open(fname, "r")
        l = of.readline().split()
        nbb = int(l[0])
        version = l[3]
        if nbb != self.net.natoms:
            raise IOError("Wrong number of BBs")
        if version == "1.1":
            offset = 1
        elif version == '2.0':
            offset = 1
        else:
            offset = 0
        self.norientations = []
        for si, s in enumerate(self.bblist):
            nc = len(s.bb.connector)
            l = of.readline().split()
            if l[1+offset] != s.name:
                raise IOError("wrong BB #%d : %s" % (si, l[1]))
            if int(l[2+offset]) != nc :
                raise IOError("inconsistent number of connectors for BB %s" % s.name)
            s.bb.orientations = []
            s.bb.which = []
            no = int(l[3+offset])
            s.bb.penalty = float(l[4+offset])
            for i in range(no):
                l = of.readline().split()
                s.bb.orientations.append(array([float(i) for i in l[0:3]]))
                if version != '2.0':
                    s.bb.which.append([int(i) for i in l[4:]])
                else:
                    s.bb.which.append([int(i) for i in l[4:-1]]) # in 2.0 we have the AAD value for each orientation

            self.norientations.append(no)
            if s.bb.linker:
                #self.norientations[si] = s.nrot
                self.norientations[si] = 1
                if s.bb.zflip : self.norientations[si] *= 2
            logger.info("BB %3d (%-20s) # orientations: %3d penalty %10.6f" % (si, s.bb.name, no, s.bb.penalty))
        of.close()
        self.nframeworks = self.get_number_isoreticular_frameworks()
        return
            
    def get_number_isoreticular_frameworks(self, norients=None):
        """Get number of possible isoreticular isomers for given number of orientations per block
        :Parameter:
            -norients(list of int): the list of number of orientations per block
                (default: self.norientations)
        """
        if norients is None:
            norients = self.norientations
        nisomers = np.prod(norients)
        logger.info("number of isoreticular frameworks: "+str(nisomers))
        if nisomers > 1: logger.info("Isoreticular isomerism detected!\n    (further info: Bureekaew et al., CrystEngComm, 2015, 17.2, 344)")
        return nisomers

###  graph methods  ##################################################
    def detect_graphs(self, colpen_srules, colpen_orules, atypes='1', selection=[0], write_flag=True):
        """detect graphs according to molecular symmetry (see molsys.topo) [NEEDS STRONG CLARIFICATION [RA]]
        
        :Parameters:
        colpen_srules(dict): color penalty sum rules: keys are vertex type, values are penalty sums (cf. TBA)
        colpen_orules(dict): color penalty orientation rules: keys are vertex type, values are orientation sums (cf. TBA)
        atypes(string): vertex type upon which all the possible permutations are computed
        selection(array of integers? strings? to be check): vertex type according to the symmetry operations
        are founded
        write_flag(bool):"""
        m = self.net.dummy_col2vex(addon='spg') 
        self.net.set_colpen_rule(colpen_srules, colpen_orules)
        verts = self.net.filter_vertices(atypes=atypes)
        perms, L, M, N = self.net.compute_permutations(atypes)
        bcol = self.net.set_bcol_from_perms(perms,M)
        bcv = self.net.filter_bcv(verts, bcol, set_arg=True)
        self.net.set_chromosomes()
        self.net.set_bcolchromosomes()
        graphs = self.net.filter_graph(verts)
        """detect unequivalent graphs by symmetry"""
        ###INITIALISE LISTS AND COUNTERS
        lcv = []
        lf = []
        ld, lsg, lm = [],[],[]
        pcount, count, maxcount = 1, 1, len(graphs)
        try:
            for g in graphs:
                logger.info("Search framework n.%s" % (len(lcv),))
                logger.info("Starting partial cycle n.%s for a total of n.%s/%s" % (pcount, count, maxcount))
                self.net.init_color_edges(bcolors=g)
                cv = self.net.col2vex(sele=selection)
                cv.set_cell(self.net.cell)
                cv.addon('spg')
                match = None
                for i,cvi in enumerate(lcv):
                    try:
                        match = cv.spg.find_symmetry_from_colors(cvi.colors, m.spg.symperms)
                        break #No need to run over the whole lcv
                    except ValueError:
                        pass #No symmetry found, proceed to the next cvi
                count += 1
                if match is None:
                    ### NEW FRAMEWORK FOUND!
                    bothcv = self.net.col2vex()
                    bothcv.addon('spg')
                    bothcv.spg.generate_spgcell()
                    ld.append( len(bothcv.spg.get_symmetry()[0]) )
                    lsg.append(bothcv.spg.get_spacegroup())
                    lm.append(bothcv)
                    pcount = 1
                    lcv.append(cv)
                    lf.append(copy.deepcopy(self)) ###TO BE CHECKED!!!
                    logger.info("Match not found! => %s. NEW framework found!" % (len(lcv),))
                else:
                    ### FRAMEWORK RELATED BY SYMMETRY TO ANY OF OLD FRAMEWORKS
                    pcount += 1
                    logger.info("Match found: lcv[%s] with sym[%s]" % (i, match[0]))
        except KeyboardInterrupt: ###SO THAT IT CAN BE INTERRUPTED (N.B. there are flaws)
            pass
        ###CREATE DIRECTORY WHERE TO WRITE STRUCTURES
        if write_flag: ### TO BE IMPLEMENTED
            superstring = ''.join(map(str,self.net.supercell))
            try:
                os.makedirs('%s_%s' % (superstring, self.name))
            except OSError:
                pass
            ###WRITE SYMMETRY-UNEQUIVALENT STRUCTURES
            for i in range(len(lcv)):
                lcv[i].write("%s_%s/%s_edgebary.txyz" % (superstring, self.name, i))
                lcv[i].write("%s_%s/%s_edgebary.xyz"  % (superstring, self.name, i))
                #lf[i].net.add_vertex_on_color(0, "b", "2")
                #lf[i].net.add_vertex_on_color(1, "k", "4")
                lf[i].net.add_vertex_on_color(((0, "b", "2"),(1, "k", "4"))) ###CHECK
                lf[i].net.write("%s_%s/%s_color.txyz" % (superstring, self.name, i))
                lf[i].net.write("%s_%s/%s_color.xyz"  % (superstring, self.name, i))
        ###LOGGING INFORMATION
        logger.info('*'*80)
        logger.info('NG:', len(ld))
        logger.info('LD:', ld)
        logger.info('LSG:',)
        for sg in lsg:
            logger.info(sg[0],' ',)
        return

################################################################################

    ###  real framework ################################################
    
    def generate_framework(self, oindex=None, autoscale=False,atype=True,random_oindex=False):
        """Generates the atomistic structure of the framework given a valid set of orientations.
        I.E. either detect_orientations or read_orientations must be called BEFORE generate_framework.
        Steps:
        1)get framework connections as nested list of:
            pair of bound vertices, pair of connecting atoms, pair of which indices for orientations
        2)if requested, autoscale framework according to
            block size, orientations, radii of connecting atoms
        3)set framework positions of blocks as molecules
            according to vertex positions and computed orientations
        N.B.: connections goes before positions and autoscale must be in the middle
            to get connectors and "whiches" for scaling. Storing connections in (1) is then needed
            to connect blocks via atoms in (3).
        :Parameters:
        - oindex(list of integers): orientation chromosome
            i.e. indices of which orientation is used per vertex
        - autoscale(bool): if True, autoscale net according to bb size
        - atype(bool): if true, the final framework is atomtyped using the standard convention"""
        if oindex is None:
            if random_oindex is False:
                oindex = len(self.norientations)*[0]
            else:
                if self.mpi_rank == 0:
                    oindex = self.random_genome()
                else:
                    oindex = None
                oindex = self.mpi_comm.bcast(oindex)
            if self.nframeworks > 1:
                logger.warning("Number of possible frameworks is %s" % (self.nframeworks,))
                if random_oindex is False:
                    logger.warning("Default orientation index is the \"first\" of the orientations per each bb\n    WHATEVER IT MEANS")
                else:
                    logger.warning("random orientation index requested:")
        if self.nframeworks > 1:
            logger.info("generating framework with oindex")
        else:
            logger.info("generating the only possible framework")
        logger.info("indices of orientations:\n    [%s]" % (",".join(map(str,oindex))),)
        self.framework = molsys.mol()
        self.framework.orientation = oindex
        framework_connections = self.get_framework_connections()
        if autoscale: self.autoscale_net(framework_connections)
        if self.net.periodic:
            self.framework.set_cellparams(self.net.cellparams)
        self.set_framework_positions()
        self.set_framework_connections(framework_connections)
        self.framework.remove_dummies(keep_conn=True)
        if atype is True:
            at = molsys.util.atomtyper(self.framework); at()
        return self.framework

    def generate_all_frameworks(self,basename):
        ''' function to generate all possible frameworks
			works only for divalent ambiguieties yet'''
        import itertools
        genes = []
        nprod = self.norientations.count(2)
        for p in itertools.product([0,1],repeat=nprod):
       	    gene = []
            cno2 = 0
            for i,no in enumerate(self.norientations):
                if no == 1:
                    gene.append(0)
                else:
                    gene.append(p[cno2])
                    cno2 += 1
            genes.append(gene)
        # framework construction
        for i,gene in enumerate(genes):
            self.generate_framework(gene)
            self.write_framework(basename+'_'+str(i)+'.mfpx')
        return i

    def generate_all_frameworks(self,basename):
        ''' function to generate all possible frameworks
			works only for divalent ambiguieties yet'''
        import itertools
        gene = [0 for x in range(self.net.natoms)]
        genes = [gene]
        for i,n in enumerate(self.norientations):
            if n != 1:
                new_genes = [copy.deepcopy(genes) for x in range(n-1)]
                print(('new', new_genes))
                for k in range(n-1):
                    for j in range(len(new_genes[k])):
                        new_genes[k][j][i] = k+1
                        #print i,k,j,new_genes[k][j][i]
                    genes += copy.copy(new_genes[k])
        for i,gene in enumerate(sorted(genes)):
            print(gene)
            self.generate_framework(gene)
            self.write_framework(basename+'_'+str(i)+'.mfpx')
        return i

    def random_genome(self):
        gene = []
        for i,no in enumerate(self.norientations):
            if no == 1:
                gene.append(0)
            else:
                gene.append(np.random.randint(0,no))
        return gene  

    def generate_n_frameworks(self,n,basename):
        ''' function to generate n random frameworks '''
        #if n >= self.nframeworks:
        #    self.generate_all_frameworks(basename)
        #    return
        used_genes = []
        for i in range(n):
            gene = self.random_genome()
            while used_genes.count(gene) != 0:
                gene = self.random_genome()
            self.generate_framework(gene)
            self.write_framework(basename+'_'+str(i)+'.mfpx')
            used_genes.append(gene)
        return

    def get_framework_connections(self):
        """Get framework connections as nested list.
        Each list is a connection and includes all the information needed to
            make an atomistic (!) bond btw. blocks
        :Returns:
        - connections(list of lists of int/np.int): each list contains, in order
            - va, vb:
            - ca, cb:
            - wa, wb:
        i.e. all the required indices to connect block A and block B via
            a bond btw. connector A and connector B."""
        connections = []
        for va in range(self.net.natoms):
            for ivb, vb in enumerate(self.net.conn[va]):
                if vb < va: ### i.e. only if va < vb
                    continue
                # find index of va in connectivity of vb (if this fails connectivity is wrong!)
                if self.net.conn[vb].count(va) > 1:
                    connb = copy.copy(self.net.conn[vb])
                    right_pair = False
                    while not right_pair:
                        iva = connb.index(va)
                        right_pair = all(equal(self.net.pconn[vb][iva]*-1.0,self.net.pconn[va][ivb]))
                        if not right_pair:
                            # remove this possibility
                            connb[iva] = -1
                    #logger.warning('%d-%d bond found in position n.%s' % (va,vb,iva)) ###PLEASE EXPLAIN [RA]
                else:
                    iva = self.net.conn[vb].index(va)
                logger.debug("add bond between vertex %d and %d" % (va, vb))
                ca, wa = self.get_connection(va,ivb)
                cb, wb = self.get_connection(vb,iva)
                #self.pprint("%s-%s -> %s-%s" % (vb,iva,cb,wb))
                connections.append([va,vb,ca,cb,wa,wb]) ###STORE THEM, bb[va], bb[vb] can be derived
        return connections

    def set_framework_connections(self, connections):
        """Set framework connections via list of connections
        A connection is represented by six (or more) indices"""
        for iconn in connections:
            va, vb, ca, cb, wa, wb = iconn
            bba, bbb = self.bblist[va], self.bblist[vb]
            self.connect(bba,bbb,ca,cb,wa,wb)
        return

    def get_connection(self,v,iv_other):
        """return corresponding connecting atom
        :Parameters:
        - v(int):
        - iv_other(int):
        :Variables:
        - oiv(int):
        :Returns:
        - c(int):
        - w(int):"""
        bb = self.bblist[v] 
        oiv = self.framework.orientation[v]
        w = self.get_which(v, iv_other, bb, oiv)
        c = self.get_connector(w, bb)
        return c, w
    
    def get_which(self, v, iv_other, bb, oiv):
        """ what is which? 
        :Parameters:
        - v(int):
        - bb(bb type mol object):
        - iv_other(int):
        - oiv(int):
        :Variables:
        - wl(list of int):
        :Returns:
        - w1(int):
        :Notes:
        if bb.linker: oiv refers to nrot and, if set, zflip
            if bb.zflip: whichlist must be reverted after oiv/bb.nrot possible rotations (EXPLAIN WHY) [RA]"""
        if bb.bb.linker:
            wl = list(bb.bb.which[0])
            if bb.bb.zflip:
                if oiv/bb.bb.nrot:
                    #print('warning, get_which changed for debug purposes')
                    wl.reverse()
        elif bb.bb.single:
            wl = [0]
        elif bb.bb.no_rot:
            wl = bb.bb.which[0]
            if bb.bb.zflip:
                if oiv/bb.bb.nrot:
                    #print('warning, get_which changed for debug purposes')
                    wl.reverse()
        else:
            wl = list(bb.bb.which[oiv])
        w = wl[iv_other]
        return w
    
    def get_connector(self, w, bb):
        """get connecting atom
        :Parameters:"
        - w(int)
        - bb(mol.bb)
        :Returns:
        - c(int): connecting atom with proper atom offset due to block insertion as molecules (see later)"""
        if bb.bb.single:
            c = bb.bb.connector[0]+bb.bb.atom_offset
        else:
            try:
                c = bb.bb.connector[w]+bb.bb.atom_offset
            except:
                import pdb; pdb.set_trace()
        return c


    def set_framework_positions(self):
        bbcounter = defaultdict(int)
        for i, bb in enumerate(self.bblist):
            bbcounter[bb.name] += 1
            if i%500==0: logger.debug('-----',i,'-----')
            logger.debug("bbcounter is %s" % (bbcounter,))
            bb = self.bblist[i]
            xyz = self.net.xyz[i]
            oi = self.framework.orientation[i]
            #bb.set_fragtypes([bb.name]*bb.get_natoms())
            #bb.set_fragnumbers([1]*bb.get_natoms())
            self.add_framework_molecule(xyz, oi, bb)
        # if self.framework.periodic: self.framework.wrap_in_box()
        return

    def add_framework_molecule(self, xyz, oi, bb):
        """Add single block as molecule in the framework
        Connectivity btw. blocks is NOT included"""

        if self.use_symmetry == True:
            rotmat = self.spg.transformations[bb.index][0]
        else:
            rotmat = None
        if bb.bb.linker:
            # if it is a linker we can have only one orientation
            # the orientation encoding is slightly different
            o = bb.bb.orientations[0]
            max_oi = bb.bb.nrot
            if bb.bb.zflip: max_oi *= 2
            if oi >= max_oi:
                raise ValueError("orientation index for linker is too large")
            zflip = 0
            rot   = 0
            if bb.bb.zflip:
                zflip = oi/bb.bb.nrot
                rot   = oi%bb.bb.nrot
            else:
                rot = oi
            scale = 1.0
            if zflip: scale = [1.0, 1.0, -1.0]
            roteuler = None
            if bb.bb.nrot > 1:
                roteuler = [0.0, 0.0, rot*pi*2.0/bb.bb.nrot]
            self.framework.add_mol(bb, translate=xyz, rotate=o, scale=scale, roteuler=roteuler,rotmat=rotmat)
        elif bb.bb.no_rot == True:
            o = bb.bb.orientations[0]
            if bb.bb.zflip:
                zflip = oi/bb.bb.nrot
                rot   = oi%bb.bb.nrot
            scale = 1.0
            if bb.bb.zflip: scale = [1.0, 1.0, -1.0]
            roteuler = None
            self.framework.add_mol(bb, translate=xyz, rotate=o, scale=scale, roteuler=roteuler,rotmat=rotmat)
        else:
            o = bb.bb.orientations[oi]
            self.framework.add_mol(bb, translate=xyz, rotate=o,rotmat=rotmat)
        return

    def bb_rotator(self,s):
        return rotator.rotator(s.bb.nvertex_xyz,s.bb.connector_xyz,bbinstance=s,use_sobol = s.bb.use_sobol)
    
    def connect(self,bba,bbb,ca,cb,wa,wb):
        logger.debug('ca,cb,wa,wb',ca,cb,wa,wb,bba.bb.connector[wa],bbb.bb.connector[wb])
        iwa,iwb = bba.bb.connector[wa],bbb.bb.connector[wb]
        #connectionsa,connectionsb = [iwa],[iwb]
        connectionsa,connectionsb = bba.bb.connector_atoms[wa],bbb.bb.connector_atoms[wb]
        if bba.bb.connector_dummies.count(iwa) != 0:
            logger.debug(ca, iwa)
            connectionsa = bba.bb.connector_atoms[bba.bb.connector_dummies.index(iwa)]
            if bbb.bb.connector_dummies.count(iwb) != 0:
                connectionsb = bbb.bb.connector_atoms[bbb.bb.connector_dummies.index(iwb)]
        idxa = [bba.bb.atom_offset + i for i in np.array(connectionsa)]
        idxb = [bbb.bb.atom_offset + i for i in np.array(connectionsb)]
        lenca = len(connectionsa)
        lencb = len(connectionsb)
        logger.debug(connectionsa, connectionsb, idxa, idxb)
        ### choose case ###
        #import pdb; pdb.set_trace()
        
        if lenca == 1 and lencb == 1:
            # 1-1 case
            self.framework.add_bonds(idxa[0],idxb[0])
            return
        elif lenca == 1 or lenca == 1:
            # 1-N case
            self.framework.add_bonds(idxa,idxb)
            return
        elif lenca == lencb:
            if lenca == 2:
                # 2-2 case, four times faster than standard
                #self.framework.add_naive_hungarian_bonds(idxa,idxb)
                #self.framework.add_standard_hungarian_bonds(idxa,idxb)
                self.framework.add_shortest_bonds(idxa,idxb)
                return
            else:
                # N-N case
                self.framework.add_standard_hungarian_bonds(idxa,idxb)
                return
        else:
            # N-M case
            self.framework.add_advanced_hungarian_bonds(idxa,idxb)
            return
        
    def write_framework(self, fname, **kwargs):
        self.framework.write(fname, **kwargs)
        return
        
    ###  pydlpoly methods ################################################
    
    def pdlp_setup(self, sysname, bcond=3):
        logger.info("Starting up pydlpoly with %s.key and %s.control as input" % sysname)
        # first generate an arbitrary framework for startup
        oi = len(self.norientations)*[0]
        self.generate_framework(oi)
        self.write_framework(sysname+".xyz")
        self.pdlp = pydlpoly.pydlpoly(sysname)
        self.pdlp.setup(bcond=bcond)
        return
        
    def pdlp_relax(self, oindex, suffix, xyzmin=0.1, cellmin=.5):
        """ optimizes the structure defined by oindex to the tolerances given
        it writes the final config with an appended suffix and returns the total energy """
        logger.info("Optimizing the structure")
        self.generate_framework(oindex)
        xyzfile = self.pdlp.name+".xyz"
        self.write_framework("../"+xyzfile)
        self.pdlp.reFIELD(xyzfile)
        self.pdlp.MIN_lbfgs(xyzmin)
        self.pdlp.LATMIN_sd(cellmin, xyzmin)
        self.pdlp.write_tinker_xyz(name+"_"+suffix+".xyz")
        e = self.pdlp.get_energy_contribs()
        return e.sum()
        
        
    ##  methods for net database ##########################################
        
    def scan_single_bb(self,model,vertex_name,mintrials=200, ntrials=300, thresh=0.01,outpen=False):
        bb = model.bb
        logger.debug("SCANNING ORIENTATION OF %s USING %d TRIALS" % (model.name,ntrials))
        ind = self.net.atypes.index(vertex_name); i = ind
        nvertex_xyz = np.array([self.net.get_neighb_coords(i,ci) for ci in range(len(self.net.conn[i]))],'d') - self.net.xyz[i]
        rot = rotator.rotator(nvertex_xyz,model.bb.connector_xyz)
        try:
            p, all_o, all_w = rot.screen_orientations(mintrials, ntrials, thresh)
        except ValueError:
            return ['ERROR!!!',-1,-1,-1]
            logger.warning('ERROR IN ROTATING')
        return p 
    
