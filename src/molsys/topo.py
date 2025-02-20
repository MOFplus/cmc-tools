# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import string as st
import numpy as np
import types
import copy
import string
import logging
from collections import Counter

from . import util
from . import mol
from molsys.util import cell_manipulation
from .util import unit_cell
from .util import elems as elements
from .util import rotations
from .util import images
import random
import itertools
from . import mol as mol
from scipy.optimize import linear_sum_assignment as hungarian

try:
    from ase import Atoms
    from pyspglib import spglib
except ImportError:
    spg = False
else:
    spg = True


import logging
logger    = logging.getLogger("molsys")
#logger.setLevel(logging.DEBUG)
#fhandler  = logging.FileHandler("molsys.log")
#fhandler.setLevel(logging.DEBUG)
#shandler  = logging.StreamHandler()
#shandler.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
#fhandler.setFormatter(formatter)
#shandler.setFormatter(formatter)
#logger.addHandler(fhandler)
#logger.addHandler(shandler)

np.set_printoptions(threshold=20000)

deg2rad = np.pi/180.0
SMALL_DIST = 1.0e-3


class topo(mol):
    def __init__(self):
        mol.__init__(self)
        self.periodic= True
        self.use_pconn = True   # flag to use pconn: keeps the image number along with the bond
        self.pconn=[]
        self.ptab= []
        # extra default for pyspglib
        if spg:
            self.symprec = SMALL_DIST        # precision in symmetry detection .. pyspglib default of 1.0e-5 seems to be way too small for large systems (in Angstrom)
            self.nonhydrogen = False  # use only non-hydrogen atoms if True in symmetry detection or any operation
        return

    @classmethod
    def fromMol(cls, m):
        """set self from mol instance"""
        t = cls()
        ### MINIMUM ATTRIBUTE SETUP ###
        ###TBI: all relevant non-method attributes retrieved ###
        t.set_natoms(m.natoms)
        t.set_xyz(m.xyz)
        t.set_conn(m.conn)
        t.set_elems(m.elems)
        t.set_atypes(m.atypes)
        t.set_cell(m.cell)
        t.add_pconn()
        return t

    ###### helper functions #######################

#RS
# looks like this is not used anymore ...

#    def get_distvec2(self, i, j,exclude_self=True):
#        """ vector from i to j
#        This is a tricky bit, because it is needed also for distance detection in the blueprint
#        where there can be small cell params wrt to the vertex distances.
#        In other words: i can be bonded to j multiple times (each in a different image)
#        and i and j could be the same!! """
#        leni = True
#        lenj = True
#        try:
#            l=len(i)
#            if l > 1:
#                ri = np.array(i)
#            else:
#                leni = False
#                ri = self.xyz[i]
#        except:
#            ri = self.xyz[i]
#        try:
#            l=len(j)
#            if l > 1:
#                rj = np.array(j)
#            else:
#                rj = self.xyz[j]
#        except:
#            rj = self.xyz[j]
#            lenj = False
#        if 1:
#            all_rj = rj + self.images_cellvec
#            all_r = all_rj - ri
#            all_d = np.sqrt(np.add.reduce(all_r*all_r,1))
#            d_sort = np.argsort(all_d)
#            if exclude_self and (np.linalg.norm(ri-rj) <= 0.001):
#                d_sort = d_sort[1:]
#            closest = d_sort[0]
#            closest=[closest]
#            if (abs(all_d[closest[0]]-all_d[d_sort[1]]) < SMALL_DIST):
#                for k in d_sort[1:]:
#                    if (abs(all_d[d_sort[0]]-all_d[k]) < SMALL_DIST):
#                        closest.append(k)
#            d = all_d[closest[0]]
#            r = all_r[closest[0]]
#        return d, r, closest

    # thes following functions rely on an exisiting connectivity conn (and pconn)

#    def get_neighb_coords(self, i, ci):
#        """ returns coordinates of atom bonded to i which is ci'th in bond list """
#        j = self.conn[i][ci]
#        rj = self.xyz[j].copy()
#        if self.periodic:
#            if self.use_pconn:
#                img = self.pconn[i][ci]
#                rj += np.dot(img, self.cell)
#            else:
#                all_rj = rj + self.images_cellvec
#                all_r = all_rj - self.xyz[i]
#                all_d = np.sqrt(np.add.reduce(all_r*all_r,1))
#                closest = np.argsort(all_d)[0]
#                return all_rj[closest]
#        return rj
#
#    def get_neighb_dist(self, i, ci):
#        """ returns coordinates of atom bonded to i which is ci'th in bond list """
#        ri = self.xyz[i]
#        j = self.conn[i][ci]
#        rj = self.xyz[j].copy()
#        if self.periodic:
#            if self.use_pconn:
#                img = self.pconn[i][ci]
#                rj += np.dot(img, self.cell)
#            else:
#                all_rj = rj + self.images_cellvec
#                all_r = all_rj - self.xyz[i]
#                all_d = np.sqrt(np.add.reduce(all_r*all_r,1))
#                closest = np.argsort(all_d)[0]
#                return all_rj[closest]
#        dr = ri-rj
#        d = np.sqrt(np.sum(dr*dr))
#        return d

    ######## manipulations in particular for blueprints

#    def make_supercell(self, supercell):
#        self.supercell = tuple(supercell)
#        logger.info('Generating %i x %i x %i supercell' % self.supercell)
#        img = [np.array(i) for i in images.tolist()]
#        ntot = np.prod(supercell)
#        nat = copy.deepcopy(self.natoms)
#        nx,ny,nz = self.supercell[0],self.supercell[1],self.supercell[2]
#        pconn = [copy.deepcopy(self.pconn) for i in range(ntot)]
#        conn =  [copy.deepcopy(self.conn) for i in range(ntot)]
#        xyz =   [copy.deepcopy(self.xyz) for i in range(ntot)]
#        elems = copy.deepcopy(self.elems)
#        left,right,front,back,bot,top =  [],[],[],[],[],[]
#        neighs = [[] for i in range(6)]
#        iii = []
#        for iz in range(nz):
#            for iy in range(ny):
#                for ix in range(nx):
#                    ixyz = ix+nx*iy+nx*ny*iz
#                    iii.append(ixyz)
#                    if ix == 0   : left.append(ixyz)
#                    if ix == nx-1: right.append(ixyz)
#                    if iy == 0   : bot.append(ixyz)
#                    if iy == ny-1: top.append(ixyz)
#                    if iz == 0   : front.append(ixyz)
#                    if iz == nz-1: back.append(ixyz)
#        for iz in range(nz):
#            for iy in range(ny):
#                for ix in range(nx):
#                    ixyz = ix+nx*iy+nx*ny*iz
#                    dispvect = np.sum(self.cell*np.array([ix,iy,iz])[:,np.newaxis],axis=0)
#                    xyz[ixyz] += dispvect
#
#                    i = copy.copy(ixyz)
#                    for cc in range(len(conn[i])):
#                        for c in range(len(conn[i][cc])):
#                            if (img[13] == pconn[i][cc][c]).all():
#                                #conn[i][cc][c] += ixyz*nat
#                                conn[i][cc][c] = int( conn[i][cc][c] + ixyz*nat )
#                                pconn[i][cc][c] = np.array([0,0,0])
#                            else:
#                                px,py,pz     = pconn[i][cc][c][0],pconn[i][cc][c][1],pconn[i][cc][c][2]
#                                #print(px,py,pz)
#                                iix,iiy,iiz  = (ix+px)%nx, (iy+py)%ny, (iz+pz)%nz
#                                iixyz= iix+nx*iiy+nx*ny*iiz
#                                conn[i][cc][c] = int( conn[i][cc][c] + iixyz*nat )
#                                pconn[i][cc][c] = np.array([0,0,0])
#                                if ((px == -1) and (left.count(ixyz)  != 0)): pconn[i][cc][c][0] = -1
#                                if ((px ==  1) and (right.count(ixyz) != 0)): pconn[i][cc][c][0] =  1
#                                if ((py == -1) and (bot.count(ixyz)   != 0)): pconn[i][cc][c][1] = -1
#                                if ((py ==  1) and (top.count(ixyz)   != 0)): pconn[i][cc][c][1] =  1
#                                if ((pz == -1) and (front.count(ixyz) != 0)): pconn[i][cc][c][2] = -1
#                                if ((pz ==  1) and (back.count(ixyz)  != 0)): pconn[i][cc][c][2] =  1
#                                #print(px,py,pz)
#        self.conn, self.pconn, self.xyz = [],[],[]
#        for cc in conn:
#            for c in cc:
#                self.conn.append(c)
#        for pp in pconn:
#            for p in pp:
#                self.pconn.append(p)
#        self.natoms = nat*ntot
#        self.xyz = np.array(xyz).reshape(nat*ntot,3)
#        self.cellparams[0:3] *= np.array(self.supercell)
#        self.cell *= np.array(self.supercell)[:,np.newaxis]
#        self.inv_cell = np.linalg.inv(self.cell)
#        self.elems *= ntot
#        self.atypes*=ntot
#        self.images_cellvec = np.dot(images, self.cell)
#        return xyz,conn,pconn

    ######### connectivity things #################################

    def detect_conn(self, fixed_cutoff=None, pconn=False, exclude_pairs=None, cov_rad_buffer=0.1):
        self.conn = []
        if pconn:
            self.use_pconn=True
        else:
            self.use_pconn=False
        for i in range(self.natoms):
            self.conn.append([])
            if self.use_pconn: self.pconn.append([])
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                d,r,imgi=self.get_distvec(i,j)
                bond = False
                if fixed_cutoff:
                    if d<fixed_cutoff: bond = True
                else:
                    covradi = elements.cov_radii[self.elems[i]]
                    covradj = elements.cov_radii[self.elems[j]]
                    if d<(covradi+covradj+cov_rad_buffer) : bond = True
                # exclude pairs testing
                if exclude_pairs and bond:
                    el_p1,el_p2 = (self.elems[i], self.elems[j]),(self.elems[j], self.elems[i])
                    for expair in exclude_pairs:
                        if (expair == el_p1) or (expair == el_p2):
                            bond= False
                            break
                if bond:
                    if len(imgi)>1 and not self.use_pconn:
                        raise ValueError("Error in connectivity detection: use pconn!!!")
                    for ii in imgi:
                        self.conn[i].append(j)
                        self.conn[j].append(i)
                        if self.use_pconn:
                            image = images[ii]
                            self.pconn[i].append(image)
                            self.pconn[j].append(image*-1)
        return
    
#    def detect_conn_by_coord_WRONG(self, pconn=False, exclude_pairs=None):
#        assert len(self.ncoord) == self.natoms , "number of coordination per vertex must be set"
#        self.use_pconn = pconn
#        self.set_empty_conn()
#        ncoord = self.ncoord[:]
#        if self.use_pconn: self.set_empty_pconn()
#        for i in xrange(self.natoms):
#            dists = []
#            js = []
#            imgs = []
#            maxcoord = ncoord[i]
#            if maxcoord == 0:
#                continue
#            for j in xrange(i+1,self.natoms):
#                d,r,imgi=self.get_distvec2(i,j)
#                ### exclude w/ exclude pairs
#                excluded = False
#                if exclude_pairs:
#                    # delete excluded pairs
#                    el_p1,el_p2 = (self.elems[i], self.elems[j]),(self.elems[j], self.elems[i])
#                    for expair in exclude_pairs:
#                        if (expair == el_p1) or (expair == el_p2):
#                            excluded = True
#                            break
#                if not excluded:
#                    dists.append(d)
#                    js.append(j)
#                    imgs.append(imgi)
#            ### INIT np.array
#            dists = np.array(dists)
#            js = np.array(js)
#            imgs = np.array(imgs)
#            ### GET SORTING INDICES
#            sortind = np.argsort(dists)
#            ### ARRANGE BY SORTING INDICES
#            dists = dists[sortind]
#            js = js[sortind]
#            imgs = imgs[sortind]
#            ### GET MAX IND UNTILL BOND
#            imgslen = map(len,imgs)
#            imgscum = np.cumsum(imgslen)
#            try:
#                tillind = np.where(imgscum == maxcoord)[0][0] + 1
#            except IndexError:
#                print(self.name, self.ncoord[i], maxcoord)
#                print(self.ncoord)
#                print(ncoord)
#                print(imgslen, imgscum)
#                break
#            for k,j in enumerate(js[:tillind]):
#                if imgslen[k] > 1 and not self.use_pconn:
#                    raise ValueError("Error in connectivity detection: use pconn!!!")
#                for ii in imgs[k]:
#                    self.conn[i].append(j)
#                    self.conn[j].append(i)
#                    if self.use_pconn:
#                        image = images[ii]
#                        self.pconn[i].append(image)
#                        self.pconn[j].append(image*-1)
#                ncoord[j] -= imgslen[k]
#        return
    
    def detect_conn_by_coord(self, pconn=False, exclude_pairs=None):
        """TBI: EXCLUDE_PAIRS"""
        """Detects connectivity by coordination number of the vertices/atoms
        :Variables:
            - self.natoms = N (int): number of atoms
            - self.ncoord (list of int's): coordination number per vertex/atom
            - dists (NxN float np.triu array): distances per distinct couple of vertices
            - imgs (NxN list np.triu array): (periodic) images per bond
            - nimgs (NxN int np.triu array): number of images per bond
            - maximgs (N int array): max number of images per bond per atom
                N.B.: 1 where 0
            - rimgs (NxN int np.tril array): remainder nimgs - maximgs
            - lnimgs (N list of lists): default addend list for possible bonds (=0)
            - lrimgs (N list of lists): very-high addend list for impossible bonds (>>0, e.g. sys.maxint)
                N.B.: impossible bonds are self bonds, inverse-cross bonds, and bonds that are non-detected as periodic images (hence rimgs)
            - aimgs (NxM int array): augmented addend cost array for images
            - adists (NxM float array): augmented dists array
                N.B.: adists == 0 where aimgs returns "impossible" cost
                else: returns default cost
                N.B.B.: aimgs == very-high cost where adists CAN BE != 0
        """
        np.set_printoptions(precision=2,linewidth=240)
        assert len(self.ncoord) == self.natoms,\
            "number of coordination per vertex must be set"
        self.use_pconn = pconn
        self.set_empty_conn()
        if self.use_pconn: self.set_empty_pconn()
        dists = np.zeros([self.natoms]*2)
        imgs = np.empty([self.natoms]*2, dtype=np.object_)
        imgs.fill([])
        imgs = np.frompyfunc(list,1,1)(imgs)
        nimgs = np.zeros([self.natoms]*2, dtype=np.int)
        for i in xrange(self.natoms):
            for j in xrange(i+1, self.natoms):
                dists[i,j], rdummy, imgs[i,j] = self.get_distvec(i,j)
            nimgs[i] = map(len,imgs[:,i])
        nimgs = nimgs.T[:]
        ### ### ### ### ###
        from pyscipopt import Model, quicksum
        self.E = {}
        model = Model("getConnByCoord")
        for i,row in enumerate(imgs):
            for j,entries in enumerate(row): #last is empty
                for e in entries:
                    self.E[(i,j),e] = model.addVar(
                        vtype="B",
                        name="e(%s-%s.%s)" % (i,j,e)
                    )
        imgs_redundant = imgs + imgs.T
        imgs_mod = []
        for i, row in enumerate(imgs_redundant):
            rowmod = [[[(i,j),e] for e in entries] if entries != [] and i<j else [[(j,i),e] for e in entries] if i>j else [] for j,entries in enumerate(row)]
            rowmod = sum([e for e in rowmod if e !=[]],[])
            rowmod = [tuple(e) for e in rowmod]
            imgs_mod.append(rowmod)
            try:
                model.addCons(
                    quicksum( self.E[k] for k in rowmod ) == self.ncoord[i],
                    "RespectCoordination(c[%i]==%s)" % (i,self.ncoord[i])
                )
            except ValueError as e:
                print("Failure! "+str(e)+" "+self.name)
                return
        model.setObjective(
            quicksum( dists[k[0]]*e for k,e in self.E.items() ),
            sense="minimize"
        )
        model.hideOutput()
        model.optimize()
        if model.getStatus() == "optimal":
            print("Success!")
        else:
            print("Failure! Not optimized! "+self.name)
            return
        pbonds = []
        for k,v in self.E.items():
            if int(model.getVal(v)) == 1:
                pbonds.append(k)
        conn, pimages = zip(*pbonds)
        pconn = [images[i] for i in pimages]
        self.set_ctab( conn,  conn_flag=True)
        self.set_ptab(pconn, pconn_flag=True)
        #self.model = model
        return
    
    def remove_duplicates(self, thresh=SMALL_DIST):
        badlist = []
        for i in range(self.natoms):
            for j in range(i+1, self.natoms):
                d,r,imgi=self.get_distvec(i,j)
                if d < thresh:
                    badlist.append(j)
        new_xyz = []
        new_elems = []
        new_atypes = []
        for i in range(self.natoms):
            if not badlist.count(i):
                new_xyz.append(self.xyz[i].tolist())
                new_elems.append(self.elems[i])
                new_atypes.append(self.atypes[i])
        self.xyz = np.array(new_xyz, "d")
        self.elems = new_elems
        self.natoms = len(self.elems)
        self.atypes = new_atypes
        return

    #def add_pconn(self):
    #    """ with the method detect_conn the connectivity is detected from a distance search
    #        if a connectivity is read via a tinker file there is no pconn present.
    #        with this metod it is added for the use with weaver2 """
    #    self.use_pconn= True
    #    self.pimages = []
    #    self.pconn = []
    #    for i,c in enumerate(self.conn):
    #        atoms_pconn = []
    #        atoms_image = []
    #        for ji, j in enumerate(c):
    #            # If an atom or vertex is connected to another one multiple times (in an image), this
    #            # will be visible in the self.conn attribute, where the same neighbour will be listed
    #            # multiple times.
    #            # Sometimes, the distances are a bit different from each other, and in this case, we
    #            # have to increase the threshold, until the get_distvec function will find all imgis.
    #            n_conns = c.count(j)
    #            t = 0.01
    #            while True:
    #                d,r,imgi = self.get_distvec(i,j,thresh=t)
    #                t += 0.01
    #                if n_conns == len(imgi):
    #                    break
    #            if len(imgi) == 1:
    #                # only one neighbor .. all is fine
    #                atoms_pconn.append(images[imgi[0]])
    #                atoms_image.append(imgi[0])
    #            else:
    #                # we need to assign an image to each connection
    #                # if an atom is connected to another atom twice this means it must be another
    #                # image
    #                for ii in imgi:
    #                    # test if this image is not used for this atom .. then we can use it
    #                    if atoms_image.count(ii)==0:
    #                        atoms_image.append(ii)
    #                        atoms_pconn.append(images[ii])
    #                    else:
    #                        # ok, we have this image already
    #                        use_it = True
    #                        #print(c, "=>", j)
    #                        #print(atoms_image)
    #                        for k, iii in enumerate(atoms_image):
    #                            #print('k',k)
    #                            if (iii == ii) and (c[k] == j): use_it=False
    #                        if use_it:
    #                            atoms_image.append(ii)
    #                            atoms_pconn.append(images[ii])
    #        self.pimages.append(atoms_image)
    #        self.pconn.append(atoms_pconn)
    #    return

        # 'na',lower(label),xyz,i,j)
    def insert_atom(self, lab, aty, xyz, i, j):
        xyz.shape=(1,3)
        self.xyz = np.concatenate((self.xyz, xyz))
        self.elems.append(lab)
        self.atypes.append(aty)
        ci = self.conn[i]
        cj = self.conn[j]
        #self.natoms += 1
        #print(i, ci)
        #print(j, cj)
        if ((i <= -1) or (j <= -1)):
            self.conn.append([])
            #self.pconn.append([])
            return
        if self.use_pconn:
            pci = self.pconn[i].pop(ci.index(j))
            pcj = self.pconn[j].pop(cj.index(i))
        ci.remove(j)
        cj.remove(i)
        ci.append(self.natoms)
        cj.append(self.natoms)
        if self.use_pconn:
            self.pconn[i].append(np.zeros([3]))
            self.pconn[j].append(pcj)
        self.conn.append([i,j])
        if self.use_pconn:
            self.pconn.append([np.zeros([3]),pci])
        self.natoms += 1
        #print("end of insert .. conn:")
        #print(self.conn)
        return


    #RS !!! HACK !!! this is not pretty but becasue of the pconn here in topo we need another add_atom
    #maybe we can just call the add_atom of the mol parent class and just add the pconn stuff here.
    def add_atom(self, elem, atype, xyz):
        assert type(elem) == str
        assert type(atype)== str
        assert np.shape(xyz) == (3,)
        self.natoms += 1
        self.elems.append(elem)
        self.atypes.append(atype)
        xyz.shape = (1,3)
        if isinstance(self.xyz, np.ndarray):
            self.xyz = np.concatenate((self.xyz, xyz))
        else:
            self.xyz = xyz
        self.conn.append([])
        self.pconn.append([])
        return self.natoms -1




    def delete_atom(self,bad):
        ''' deletes an atom and its connections and fixes broken indices of all other atoms '''
        new_xyz = []
        new_elems = []
        new_atypes = []
        new_conn = []
        new_pconn = []
        for i in range(self.natoms):
            if i != bad:
                new_xyz.append(self.xyz[i].tolist())
                new_elems.append(self.elems[i])
                new_atypes.append(self.atypes[i])
                new_conn.append(self.conn[i])
                new_pconn.append(self.pconn[i])
                for j in range(len(new_conn[-1])):
                    if new_conn[-1].count(bad) != 0:
                        new_conn[-1].pop(new_conn[-1].index(bad))
        self.xyz = np.array(new_xyz, "d")
        self.elems = new_elems
        self.natoms = len(self.elems)
        self.atypes = new_atypes
        for i in range(len(new_conn)):
            #try:
                #len(new_conn[i])
            #except:
                #new_conn[i] = [new_conn[i]]
            for j in range(len(new_conn[i])):
                if new_conn[i][j] >= bad:
                    new_conn[i][j]=new_conn[i][j]-1
        self.conn = new_conn
        self.pconn = new_pconn
        return

    def add_conn(self, a1, a2):
        """ add a connection between a1 and a2 (in both directions)
        """
        if self.use_pconn:
            raise ValueError("Can not add bonds to systems with pconn - well, we can fix this ;) ")
        self.conn[a1].append(a2)
        self.conn[a2].append(a1)
        d,v,imgi = self.get_distvec(a1,a2)
        self.pconn[a1].append(images[imgi])
        d,v,imgi = self.get_distvec(a2,a1)
        self.pconn[a2].append(images[imgi])
        logger.warning('pconn may not be properly updated!!!')
        return

    def delete_conn(self,el1,el2):
        ''' removes the connection between two atoms
        :Parameters:
            - el1,el2 : indices of the atoms whose connection is to be removed'''
        idx1,idx2 = self.conn[el1].index(el2),self.conn[el2].index(el1)
        self.conn[el1].remove(el2)
        self.conn[el2].remove(el1)
        self.pconn[el1].pop(idx1)
        self.pconn[el2].pop(idx2)
        logger.warning('pconn may not be properly updated!!!')
        return

    def set_elems_by_coord_number(self):
        elems = []
        for i in range(self.natoms):
            elems.append(elements.topotypes[len(self.conn[i])])
        self.set_elems(elems)
        return

############# Plotting

    def plot(self,scell=False,bonds=False,labels=False):
        from mpl_toolkits.mplot3d import axes3d, Axes3D 
        import matplotlib.pyplot as plt
        col = ['r','g','b','m','c','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k']+['k']*200
        fig = plt.figure(figsize=plt.figaspect(1.0)*1.5)
        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        atd = {}
        for i,aa in enumerate(list(set(self.atypes))):
            atd.update({aa:col[i]})
        print(atd)
        if bonds:
            for i in range(self.natoms):
                conn = self.conn[i]
                for j in range(len(conn)):
                    if self.pconn:
                        if np.sum(np.abs(self.pconn[i][j])) == 0:
                            ax.plot([self.xyz[i][0],self.xyz[conn[j]][0]],[self.xyz[i][1],self.xyz[conn[j]][1]],[self.xyz[i][2],self.xyz[conn[j]][2]],color='black')
                        else:
                            xyznew = self.get_image(self.xyz[conn[j]],self.pconn[i][j])
                            ax.scatter(xyznew[0],xyznew[1],xyznew[2],color='orange')
                            ax.plot([self.xyz[i][0],xyznew[0]],[self.xyz[i][1],xyznew[1]],[self.xyz[i][2],xyznew[2]],color='green')
                    else:
                        ax.plot([self.xyz[i][0],self.xyz[conn[j]][0]],[self.xyz[i][1],self.xyz[conn[j]][1]],[self.xyz[i][2],self.xyz[conn[j]][2]],color=atd[self.atypes[i]])

        if labels:
            for i in range(self.natoms):
                label = str(i)+'-'+str(self.atypes[i]) +'-'+str(len(self.conn[i]))
                ax.text(self.xyz[i][0], self.xyz[i][1], self.xyz[i][2]+0.005, label, color='k',fontsize=9)
        if scell:
            xyz3 = self.make_333(out=True)
            xyz3 =  np.array(xyz3)
            ax.scatter(xyz3[:,0],xyz3[:,1],xyz3[:,2],color='r',alpha=0.5)
        xyz=np.array(self.xyz)
        for i,xx in enumerate(xyz):

            ax.scatter(xx[0],xx[1],xx[2],color=atd[self.atypes[i]])
        minbound = np.min([np.min(xyz[:,0]),np.min(xyz[:,1]),np.min(xyz[:,2])])
        maxbound = np.max([np.max(xyz[:,0]),np.max(xyz[:,1]),np.max(xyz[:,2])])
        ax.auto_scale_xyz([0.0, maxbound], [0.0, maxbound], [0.0, maxbound])
        #ax.scatter(xyz1[:,0],xyz1[:,1],xyz1[:,2],color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

########## additional stuff for edge coloring ############################################

    def color_edges(self, proportions, maxiter=100, maxstep=100000, nprint=1000, penref=0.3, thresh=1.0e-3, MC=True):
        """
        wrapper to search for a zero penalty solution of edge coloring. An initial coloring is generated
        randomly and a flipMC run is started ... if no zero penalty is found after maxsteps it is repeated ...
        """
        self.thresh = thresh
        self.MC = MC
        converged = False
        niter = 0
        while not (converged and (niter<maxiter)):
            self.init_color_edges(proportions, MC=self.MC)
            result = self.run_flip(maxstep, nprint=nprint, penref=penref, thresh=thresh)
            if result[0] : converged=True
            niter += 1
        print("**********************************************************************")
        print("edge coloring is convereged !!")
        print("final penalty is %12.6f" % result[2])
        return


    def init_color_edges(self, proportions=[], colors=[], bcolors=[], thresh=1.0e-3, MC=False):
        """
        generate datastructures for edge coloring and set up random
        the proportions are a list or tuple of integer.
        the length determines the number of colors, the numbers are the multiples
        one integer should be always 1.
        so for example proportions=[2,1] generates a two color system (red, blue)
        with 2 red and 1 blue ... this means the total number of edges inthe periodic net must
        be a multiple of 3
        """
        self.thresh = thresh
        self.MC = MC
        assert (bool(proportions),bool(list(colors)),bool(list(bcolors))).count(True) == 1,\
            "either proportions or colors or bcolors must be non-empty"
        ### generate the list of bonds only "upwards" bonds (i1<i2) are stored
        blist = [ [i,j] for i,ci in enumerate(self.conn) for j in ci if i<j ]
        ### convert blist into numpy array
        self.blist = np.array(blist)
        self.nbonds = len(self.blist)
        if list(colors):
            self.prop, self.nprop = self.colors2proportions(colors, set_arg=True)
            self.bcolors = self.colors2bcolors(colors, set_arg=True)
        elif list(bcolors):
            self.colors = self.bcolors2colors(bcolors, set_arg=True)
            self.prop, self.nprop = self.colors2proportions(self.colors)
        else:
            ###get colors via proportions, set proportions by argument
            self.colors = self.proportions2colors(proportions, set_arg=True)
            self.bcolors = self.colors2bcolors(self.colors)
            if self.MC: random.shuffle(colors)
        self.colors = np.array(self.colors)
        self.ncolors = len(self.prop)
        ### set up penalty table (per vertex)
        ### defaults
        self.colpen_sum_fact    = 1.0
        self.colpen_orient_fact = 0.5
        ### self.colpen_sumrule = self.natoms*[4.0]
        self.penalty = np.zeros([self.natoms],dtype="float64")
        ### set up the scalmat array for all vertices
        self.setup_scalprod_mats()
        for i in range(self.natoms):
            self.penalty[i] = self.calc_colpen(i)
        self.totpen = self.penalty.sum()
        return

    def flip_color(self):
        """
        does one color flip and computes the new penalty
        the old settings are kept
        """
        # get bond A
        self.bondA = random.randint(0, self.nbonds-1)
        self.colA  = self.colors[self.bondA]
        vert = self.blist[self.bondA].tolist()
        peninit = [self.penalty[vert[0]], self.penalty[vert[1]]]
        # get bond B
        self.colB = self.colA
        while self.colA == self.colB:
            self.bondB = random.randint(0, self.nbonds-1)
            self.colB = self.colors[self.bondB]
        vert += self.blist[self.bondB].tolist()
        peninit += [self.penalty[vert[2]], self.penalty[vert[3]]]
        peninit = np.array(peninit)
        # now flip the colors
        self.colors[self.bondA] = self.colB
        self.colors[self.bondB] = self.colA
        # now correct entries in bcolors
        self.set_bcol(self.bondA)
        self.set_bcol(self.bondB)
        # compute penalty of he four affected vertices only
        pennew = []
        for v in vert:
            pennew.append(self.calc_colpen(v))
        self.pennew = np.array(pennew)
        self.changed_vert = vert
        # compute delta in penalty
        delta_pen = self.pennew.sum()-peninit.sum()
        # print("flipped colors of bonds %3d and %3d (vertices: %20s) -- delta penalty: %10.3f" % (self.bondA, self.bondB, str(vert), delta_pen))
        return delta_pen

    def unflip_colors(self):
        """
        call this directly after a flip to put everything back
        """
        self.colors[self.bondA] = self.colA
        self.colors[self.bondB] = self.colB
        self.set_bcol(self.bondA)
        self.set_bcol(self.bondB)
        return

    def accept_flip(self):
        """
        call this directly after flip to keep the flip
        """
        for i, v in enumerate(self.changed_vert):
            self.penalty[v] = self.pennew[i]
        self.totpen = self.penalty.sum()
        return

    def run_flip(self, maxstep, nprint=1000, penref=0.2, thresh=1.0e-3):
        """
        run a MC with color flips as moves for maxstep or until the total penalty
        is below thresh. the virtual "temperature" or reference energy for the acceptance
        (kT) is in the same unit as the penalty and is given as penref

        :Parameters:
            - maxstep : maximum number of MC steps
            - nprint  : number of steps after which a printout is made [100]
            - penref  : reference penalty for the MC aceptance criterion exp(-pen/penref) [0.2]
            - thresh  : threshold under which convergence is assumed (zero penalty is not always reached for orientation penalty) [1.0e-3]
        """
        self.thresh = thresh
        step = 0
        while (step < maxstep) and (self.totpen>thresh):
            dpen = self.flip_color()
            accept = False
            if dpen <= 0.0:
                accept = True
            else:
                prob = np.exp(-dpen/penref)
                #print("dpen %10.5f prob %10.5f" %(dpen, prob))
                if random.random() < prob:
                    accept=True
            if accept:
                self.accept_flip()
            else:
                self.unflip_colors()
            if (step%nprint) == 0:
                print("step: %7d ; penalty %10.5f" % (step, self.totpen))
            step += 1
        if step<maxstep:
            print("Converged after %7d steps with penalty %10.5f" % (step, self.totpen))
            print("last delta_pen was %10.5f" % dpen)
            converged = True
        else:
            print("Not converged!!!")
            converged = False
        return (converged, step, self.totpen)

### GRAPH TO MOL METHODS #######################################################
    def add_vertex_on_color(self, col, lelem=None, laty=None):
        if lelem is None and laty is None:
            errmsg = \
            "add_vertex_on_color expects a triplet or alist of nested triplets"
            assert hasattr(col,'__iter__'), errmsg
            for c in col:
                assert len(c) == 3, errmsg
                self.add_vertex_on_color(*c)
            return
        for i,b in enumerate(self.blist):
            if self.colors[i] == col:
                i,j = b
                ci = self.conn[i].index(j)
                xyz_j = self.get_neighb_coords(b[0], ci)
                xyz = (self.xyz[i]+xyz_j)/2.0
                self.insert_atom(lelem, laty, xyz, i, j)
        return

    def col2vex(self, sele=None, lelem=None, laty=None):
        """from colors to vertices, returns molsys.mol instance
        original vertices are kept the same
        edges are condensed in the baricenter"""
        if sele is None:
            ncol = self.ncolors
            col = self.colors.astype(np.int)
            nbonds = self.nbonds
            ba = self.blist[:,0]
            bb = self.blist[:,1]
        else:
            try:
                indexcol = np.where(sum([self.colors == i for i in sele]))[0] ###since bool arrays, here "sum" means "or"
            except ValueError:
                indexcol = []
            col = self.colors[indexcol]
            ncol = len(sele)
            nbonds = len(col)
            ba = self.blist[:,0][indexcol]
            bb = self.blist[:,1][indexcol]
        xyz_a = self.xyz[ba]
        xyz_c = []
        for i in range(nbonds):
            bci = self.conn[ba[i]].index(bb[i])
            xyz_ic = self.get_neighb_coords(ba[i], bci)
            xyz_c.append(xyz_ic)
        xyz_c = np.array(xyz_c) + xyz_a
        xyz_c *= .5
        m = mol.fromArray(xyz_c)
        ### DEFAULT ASSIGNMENT
        if lelem is None and laty is None:
            laty = range(ncol)
            lowercase = list('kbabcdefghijklmnopqrstuvwxyz')
            lelem = [lowercase[i] for i in laty]
            laty = [str(i) for i in laty]
        elif lelem is None or laty is None:
            raise TypeError("lelem and laty must be both either None or ndarrays")
        elif len(lelem) != ncol or len(laty) != ncol:
            raise ValueError("len of sele, lelem and laty must be the same!\nsele:\t%s\nlelem:\t%s\nlaty:\t%s" % (sele, lelem, laty))
        lelem = np.array(lelem)
        laty = np.array(laty)
        m.set_elems(lelem[col])
        m.set_atypes(laty[col])
        if hasattr(self,'cell'): m.set_cell(self.cell)
        if hasattr(self,'supercell'): m.supercell = self.supercell[:]
        m.colors = self.colors
        return m

    def dummy_col2vex(self, lelem=['c'], laty=['0'], addon=None):
        """returns uncolored graph as molsys.mol instance
        if addon='spg': generate symmetry perks
            
        :Variables:
        - etypes(set): set of unique vertex types (as letters)
        - dumpensum(dict): dummy penalty summations: 1 per each vertex type
        - dumpenori(dict): dummy penalty orientations: None per each vertex type"""
        ### APPLY CONDITIONS ***AFTER*** THE DUMMY COLORING! ###
        etypes = set(self.elems)
        lenetypes = len(etypes)
        lones = [1]*lenetypes
        lnones = [None]*lenetypes
        dumpensum = dict(zip(etypes, lones))
        dumpenori = dict(zip(etypes, lnones))
        self.set_colpen_sumrule(dumpensum)
        self.set_colpen_orientrule(dumpenori)
        self.init_color_edges([1])
        m = self.col2vex(lelem=lelem, laty=laty)
        if addon=='spg':
            m.addon('spg')
            m.spg.generate_spgcell()
            m.spg.generate_symmetries()
            m.spg.generate_symperms()
        return m

### COMPUTE PERMUTATIONS #######################################################
    def compute_permutations(self, atypes):
        v = self.atypes.index(atypes)
        M = len(self.conn[v])
        L = M - self.colpen_sumrule[v] ### HACK ###
        perms = itertools.permutations(range(M),L)
        perms = [list(i) for i in perms]
        for i in perms: i.sort()
        perms = [tuple(i) for i in perms]
        perms = set(perms)
        perms = [list(i) for i in perms]
        N = len(perms)
        self.perms = perms
        return perms, L, M, N

### SET CHROMOSOMES ############################################################
    def set_chromosomes(self):
        allele = [len(bcvi) for bcvi in self.bcv]
        permchrom = [range(ia) for ia in allele]
        print("NUMBER OF PERMUTATIONS:", np.prod(allele))
        chromosomes = list(itertools.product(*permchrom))
        self.chromosomes = chromosomes

    def set_bcolchromosomes(self):
        bcolchroms = [self.chrom2bcolchrom(chrom) for chrom in self.chromosomes]
        self.bcolchroms = bcolchroms

    def chrom2bcolchrom(self, chrom):
        return self.bcv[range(self.bcv.shape[0]),chrom]

### BCOLOR SETTING FUNCTIONS ###################################################
    def set_bcol(self, bond):
        """
        utility to set color values in bcolors for bond

        :Parameters:
            - bond : index of bond i self.blist (and colors)

        """
        i, j = self.blist[bond]
        c = self.colors[bond]
        #logger.debug("bond from %3d to %3d : color %3d" % (i, j, c) )
        #logger.debug("i_conn: %s" % repr(self.conn[i]) )
        #logger.debug("j_conn: %s" % repr(self.conn[j]) )
        ### set for i
        ### BUG FOR 2x2x2 pcu: a vertex connects twice to the same vertex!!!
        ### TBI: add pconn for toper
        j_ind = self.conn[i].index(j)
        i_ind = self.conn[j].index(i)
        #logger.debug("i_ind %6d" % i_ind)
        #logger.debug("j_ind %6d" % j_ind)
        self.bcolors[i][j_ind] = c
        self.bcolors[j][i_ind] = c
        return

    def set_jbcol_from_ibcol(self, vert, bcol, set_arg=False):
        """
        utility to set color value of j from color of i

        :Parameters:
            - vert (int): the vertex i from which the colors of the other j
                vertices are assigned
            - bcol (list of int's): list of colors for that vertex
            - set_arg (bool): if True, bcolors of vertex is assigned as bcol
        """
        iconn = self.conn[vert]
        for i,iatom in enumerate(iconn):
            j = self.conn[iatom].index(vert)
            self.bcolors[iatom][j] = int(bcol[i])
        if set_arg: self.bcolors[vert] = list(bcol)
        return

    def set_bcol_from_perms(self, perms, nconn):
        bcol = np.zeros((len(perms),nconn), dtype=np.int)
        for i,e in enumerate(perms):
            bcol[i,e] = 1
        bcol = [list(bc) for bc in bcol]
        return bcol

        ### N.B.: works only for 0/1 coloring
        ### TBI: any number of colors
        
### PENALTY FUNCTION HANDLERS ##################################################
    def calc_colpen(self, vert):
        # print("calculating penalty for vert %d (%s)  colors: %s" % (vert, self.elems[vert], str(self.bcolors[vert])))
        pen_sum = self.calc_colpen_sum(vert)
        if pen_sum == 0.0:
            # this vertex has the correct number of colors on the edges
            # now compute in addition the penalty on the orientation
            return self.calc_colpen_orient(vert)
        else:
            return pen_sum

    def calc_colpen_from_bcolor(self, vert, bcolor):
        # print("calculating penalty for vert %d (%s)  colors: %s" % (vert, self.elems[vert], str(bcolor)))
        pen_sum = self.calc_colpen_sum_from_bcolor(vert, bcolor)
        if pen_sum == 0.0:
            # this vertex has the correct number of colors on the edges
            # now compute in addition the penalty on the orientation
            return self.calc_colpen_orient_from_bcolor(vert, bcolor)
        else:
            return pen_sum

    def calc_colpen_sum_from_bcolor(self, vert, bcolor):
        """ compute the color penalty for vertex vert
            rules are in list self.colpen_sumrule
        """
        # get rule for first color (currently only two colors are supported for testing)
        nc0 = self.colpen_sumrule[vert]
        nc = bcolor.count(0)
        pen = abs(nc-nc0)*self.colpen_sum_fact
        return pen*pen

    def calc_colpen_sum(self, vert):
        """ compute the color penalty for vertex vert
            rules are in list self.colpen_sumrule
        """
        # get rule for first color (currently only two colors are supported for testing)
        nc0 = self.colpen_sumrule[vert]
        nc = self.bcolors[vert].count(0)
        pen = abs(nc-nc0)*self.colpen_sum_fact
        return pen*pen

    def set_colpen_rule(self,svert_dict, overt_dict):
        self.set_colpen_sumrule(svert_dict)
        self.set_colpen_orientrule(overt_dict)
        return

    def set_colpen_sumrule(self, vert_dict):
        """
        set the color penalty for the sum of colors
        :Paramteres:

            - vert_dict: dictionary of vertices with the number of expected edges for color 0
        """
        self.colpen_sumrule = np.zeros([self.natoms], dtype="int32")
        for i in range(self.natoms):
            self.colpen_sumrule[i] = vert_dict[self.elems[i]]
        return

    def setup_scalprod_mats(self):
        self.scalmat = []
        for i in range(self.natoms):
            v = []
            for ji in range(len(self.conn[i])):
                v.append(self.get_neighb_coords(i, ji)-self.xyz[i])
            v = np.array(v)
            vn = v / np.linalg.norm(v, axis=-1)[:, np.newaxis]
            mat = np.sum(vn[:,np.newaxis,:]*vn[np.newaxis,:,:], axis=2)
            self.scalmat.append(mat)
        #self.scalmat = np.array(self.scalmat)
        return

    def calc_colpen_orient(self, vert):
        ### BROKEN ???
        # this is a HACK ... works only for vertices with two colors
        # if self.colpen_orientrule is None ignore
        # if not use the number to be added to scal
        if self.colpen_orientrule[vert] != None:
            col0_edges = []
            for i,c in enumerate(self.bcolors[vert]):
                if c == 0: col0_edges.append(i)
            scal = self.scalmat[vert][col0_edges[0], col0_edges[1]]
            return self.colpen_orient_fact*abs(scal+self.colpen_orientrule[vert])
        else:
            return 0.0

    def calc_colpen_orient_from_bcolor(self, vert, bcolor):
        # this is a HACK ... works only for vertices with two colors
        # if self.colpen_orientrule is None ignore
        # if not use the number to be added to scal
        if self.colpen_orientrule[vert] != None:
            col0_edges = []
            for i,c in enumerate(bcolor):
                if c == 0: col0_edges.append(i)
            scal = self.scalmat[vert][col0_edges[0], col0_edges[1]]
            return self.colpen_orient_fact*abs(scal+self.colpen_orientrule[vert])
        else:
            return 0.0

    def set_colpen_orientrule(self, vert_dict):
        """
        set the color penalty for the orientation of colors
        currently this works only for color zero sum=2 (if more ... how to dadd up penalties?)

        :Paramteres:

            - vert_dict: dictionary of vertices with either None or what to add to skal

        example: for color 0 (sum=2) being 180deg set it to 1.0 (-1.0+1.0 = 0.0)
                                            90deg set it to 0.0 and the fact to -0.5
        """
        self.colpen_orientrule = []
        for i in range(self.natoms):
            self.colpen_orientrule.append(vert_dict[self.elems[i]])
        return

### FILTER FUNCTIONS ###########################################################
    def filter_vertices(self, atypes=None, elems=None):
        assert bool(atypes) ^ bool(elems), """either atom types or elements"""
        if atypes:
            arr = self.atypes
            string = atypes
        elif elems:
            arr = self.elems
            string = elems
        cond = np.array(arr) == string
        filtered = np.where(cond)[0]
        return list(filtered)

    def filter_bcv(self, vertices, bcol, thresh=1e-8, set_arg=False):
        """filter by color penalty threshold the allowed bond colorings
            per vertex
        """
        bcv = []
        for v in vertices:
            bcvi = []
            for bc in bcol:
                colpen = self.calc_colpen_from_bcolor(v,bc)
                #print(bc, colpen)
                if colpen < 1e-8:
                    bcvi.append(bc)
            bcv.append(bcvi)
        bcv = np.array(bcv) ###allowed coloring per vertex
        if set_arg: self.bcv = bcv
        return bcv

    def filter_graph(self, vertices, bcolchroms=None, thresh=1e-8):
        """filter by color penalty threshold the graphs given by bond
            coloring encoded by chromosomes
        """
        if bcolchroms is None: bcolchroms = self.bcolchroms
        graphs = []
        for ib, ibcolchrom in enumerate(bcolchroms):
            for iv,v in enumerate(vertices):
                self.set_jbcol_from_ibcol(v,ibcolchrom[iv],set_arg=True)
            totpen = sum([self.calc_colpen(i) for i in range(self.natoms)])
            #print(totpen)
            if totpen < 1e-8:
                if self.bcolors not in graphs:
                    graphs.append(copy.deepcopy(self.bcolors))
        return graphs


### AUXILIARY FUNCTIONS ########################################################
    def bcolors2colors(self, bcolors, set_arg=False):
        colors = []
        for i,iconn in enumerate(self.conn):
            for j,jatom in enumerate(iconn):
                if i < jatom:
                    jcol = bcolors[i][j]
                    colors.append(jcol)
        if set_arg: self.bcolors = bcolors
        return colors

    def colors2bcolors(self, colors, set_arg=False):
        bcolors = [[None]*len(conni) for conni in self.conn]
        for k,(a,b) in enumerate(self.blist):
            conni = self.conn[a].index(b)
            connj = self.conn[b].index(a)
            bcolors[a][conni] = colors[k]
            bcolors[b][connj] = colors[k]
        if set_arg: self.colors = colors
        return bcolors

    def colors2proportions(self, colors, set_arg=False):
        ###set colors by argument, get proportions via color
        assert self.nbonds == len(colors), "number of colors is different than number of bonds"
        prop, nprop = zip( *Counter(colors).most_common() ) ###already sorted
        ###TBI: working for nprop[-1]!=1
        if set_arg: self.colors = colors
        return prop, nprop

    def proportions2colors(self, prop, set_arg=False):
        nprop = np.array(prop).sum()
        assert self.nbonds%nprop==0,  "these proportions do not work"
        nc = self.nbonds/nprop
        colors = [c for c,p in enumerate(prop) for i in range(p*nc)]
        if set_arg: self.prop, self.nprop = prop, nprop
        return colors

    @classmethod
    def GCD(cls,*num):
        """compute greatest common divisor for a list.
        rationale: fractions.gcd works only for two numbers"""
        from fractions import gcd
        if hasattr(num[0],'__iter__'): return cls.GCD(*num[0])
        if len(num) > 2:
            return reduce(lambda x,y:gcd(x,y),num)
        else:
            return gcd(*num)
