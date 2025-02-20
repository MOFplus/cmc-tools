#!/usr/bin/env python
# -*- coding: utf-8 -*-
import string
from graph_tool import Graph
from graph_tool.topology import *
import numpy
import pdb
import molsys.mol as mol
import copy
from math import fmod
# from molsys.util import RCSR  # never used

from molsys.util import systrekey

class reader(object):

    def __init__(self):
        return

    def load_keys_from_file(self,fname):
        self.keys = {}
        f = open(fname, 'r')
        for line in f.readlines():
            sline = line.split()
            if len(sline)>0:
                if sline[0] == 'key':
                    #dim = int(sline[1])
                    key = " ".join(sline[1:])
                if sline[0] == 'id':
                    name = sline[1]
                    self.keys[name] = key
        return

    def dump_keys_to_pickle(self,pname):
        return

    def load_keys_from_pickle(self,pname):
        return

    def __call__(self,name):
        return self.keys[name]


class lqg(object):

    def __init__(self, dim = 3):
        self.dim = dim
        return

    def read_systre_key(self, skey):
        """generate data structures from systrekey
        
        Args:
            skey (string): systrekey 
        """
        # split skey into a list
        skey = skey.split()
        self.dim = int(skey[0])
        skey = skey[1:]
        # now skey contains only edges with (i j lx ly lz) i,j vertices, lxyz labels
        assert len(skey)%(self.dim+2) == 0
        dfac = self.dim+2
        self.nedges = int(len(skey)/dfac)
        self.nvertices = 1
        self.edges = []
        self.labels = []
        for i in range(self.nedges):
            edge = list(map(int, skey[i*dfac:i*dfac+2]))
            for j in edge:
                if j > self.nvertices: self.nvertices = j
            edge = list(numpy.array(edge)-1)
            label = list(map(int, skey[i*dfac+2:i*dfac+dfac]))
            self.edges.append(edge)
            self.labels.append(label)
        return


    def write_systre_pgr(self, id = "mfpb"):
        pgr = "PERIODIC_GRAPH\nID %s\nEDGES\n" % id
        for e,l in zip(self.edges,self.labels):
            entry = (" %s %s" + self.dim*" %1.0f" + "\n") % tuple(list(numpy.array(e)+1)+l)
            pgr += entry
        pgr += "END"
        return pgr

    def get_lqg_from_topo(self,topo):
        assert topo.is_topo
        # be careful not working for nets where an vertex is connected to itself
        self.dim = 3
        self.nvertices = topo.get_natoms()
        self.nedges = 0
        self.edges = []
        self.labels = []
        for i in range(self.nvertices):
            for j,v in enumerate(topo.conn[i]):
                if v > i:
                    self.nedges += 1
                    self.edges.append([i,v])
                    #pdb.set_trace()
                    self.labels.append(list(topo.pconn[i][j]))
        return

    def get_lqg_from_lists(self,edges,labels,nvertices,dim):
        assert len(edges) == len(labels)
        self.edges = edges
        self.labels = labels
        self.dim = dim
        self.nedges = len(edges)
        self.nvertices = nvertices
        return

    def get_systrekey(self):
        """method calls javascript systreKey (must be installed) and computes the systreKey
        """
        skey = systrekey.run_systrekey(self.edges, self.labels)
        return skey

#    def lqg_from_systrekey(self,skey):
#        skeys = skey.split()
#        n = int(skeys[0])
#        verts = [[int(x) for x in skeys[i:i+5]] for i in range(1,len(skeys),5)

    def build_lqg(self):
        self.nbasevec = self.nedges - self.nvertices + 1
        self.molg = Graph(directed=True)
        self.molg.ep.label  = self.molg.new_edge_property("vector<double>")
        self.molg.ep.number = self.molg.new_edge_property("int")
        for i in range(self.nvertices):
            iv = self.molg.add_vertex()
        for i,e in enumerate(self.edges):
            ie = self.molg.add_edge(self.molg.vertex(e[0]),self.molg.vertex(e[1]))
            self.molg.ep.label[ie] = self.labels[i]
            self.molg.ep.number[ie] = i
        return

    def get_cyclic_basis(self):
        nbasevec = self.nbasevec
        basis = numpy.zeros([nbasevec,self.nedges], dtype="int")
        self.molg.set_directed(False)
        tree = min_spanning_tree(self.molg)
        i = 0
        for e in self.molg.edges():
            if tree[e] == 0:
                self.molg.set_edge_filter(tree)
                vl, el = shortest_path(self.molg, self.molg.vertex(int(e.target())), self.molg.vertex(int(e.source())))
                self.molg.set_edge_filter(None)
                basis[i, self.molg.ep.number[e]] = 1
                neg = False
                for eb in el:
                    idx = self.molg.ep.number[eb]
                    ebt = self.get_edge_with_idx(idx)
                    if ebt.target() == e.target():
                        if neg != True:
                            basis[i, self.molg.ep.number[eb]] = -1
                            neg = True
                        else: 
                            basis[i, self.molg.ep.number[eb]] = 1
                            neg = False
                    elif ebt.source() == e.source():
                        if neg != True:
                            basis[i, self.molg.ep.number[eb]] = -1
                            neg = True
                        else: 
                            basis[i, self.molg.ep.number[eb]] = 1
                            neg = False
                    elif ebt.source() == e.target():
                        if neg != True:
                            basis[i, self.molg.ep.number[eb]] = 1
                            neg = False
                        else: 
                            basis[i, self.molg.ep.number[eb]] = -1
                            neg = True
                    elif ebt.target() == e.source():
                        if neg != True:
                            basis[i, self.molg.ep.number[eb]] = 1
                            neg = False
                        else: 
                            basis[i, self.molg.ep.number[eb]] = -1
                            neg = True
                    e = ebt
                i += 1
        self.cyclic_basis = basis
        self.molg.set_directed(True)
        return self.cyclic_basis

    def get_cocycle_basis(self):
        n = self.nedges - (self.nedges - self.nvertices +1)
        cocycles = numpy.zeros([n, self.nedges])
        self.molg.set_directed(False)
        i = 0
        for v in self.molg.vertices():
            el = v.out_edges()
            for eb in el:
                idx = self.molg.ep.number[eb]
                ebt = self.get_edge_with_idx(idx)
                if ebt.source() == v:
                    cocycles[i, idx] = 1
                else:
                    cocycles[i, idx] = -1
            i+=1
            if i == n: break
            self.cocycle_basis = cocycles
        return self.cocycle_basis

    def get_ncocycles(self,n):
        self.molg.set_directed(False)
        cocycles = numpy.zeros([n, self.nedges])
        i = 0
        for v in self.molg.vertices():
            el = v.out_edges()
            for eb in el:
                idx = self.molg.ep.number[eb]
                ebt = self.get_edge_with_idx(idx)
                if ebt.source() == v:
                    cocycles[i, idx] = 1
                else:
                    cocycles[i, idx] = -1
            i+=1
            if i == n: break
        return cocycles

    def get_B_matrix(self):
        n = self.nedges - (self.nedges - self.nvertices +1)
        if n > 0: 
             self.B = numpy.append(self.cyclic_basis, self.cocycle_basis, axis = 0)
        else:
            self.B = self.cyclic_basis
        return self.B

    def get_alpha(self):
        vimg = []
        labels = numpy.array(self.labels)
        for i in range(numpy.shape(self.cyclic_basis)[0]):
            img = numpy.sum(self.cyclic_basis[i]* labels.T,axis = 1)
            vimg.append(img)
        for i in range(self.nedges-self.nbasevec):
            if self.dim == 2:
                vimg.append([0,0])
            else:
                vimg.append([0,0,0])
        self.alpha = numpy.array(vimg)
        return self.alpha

    def get_image(self, vec):
        labels = numpy.array(self.labels)
        return  numpy.sum(vec*labels.T,axis = 1)

    def get_fracs(self):
        self.fracs = numpy.dot(numpy.linalg.inv(self.B),self.alpha)
        # compute the scale to make the shortest edge length 1.0
        self.scale = 1.0/numpy.min((numpy.sqrt((self.fracs*self.fracs).sum(axis=1))))
        return self.fracs

    def get_lattice_basis(self):
        idx = self.find_li_vectors(self.alpha)
        latbase = self.alpha[idx]
        Lr = self.cyclic_basis[idx]
        ### we need to orthonormalize the latbase ###
        L = numpy.zeros([self.dim,self.nedges])
        olatbase = numpy.eye(self.dim, self.dim)
        for i in range(self.dim):
            b = numpy.linalg.solve(latbase.T, olatbase[i,:])
            for j in range(self.dim):
                L[i,:]+= b[j]*Lr[j,:]
        self.lattice_basis = L
        return self.lattice_basis

    def get_kernel(self):
        k = numpy.zeros([self.nbasevec-self.dim+self.nvertices-1,self.nedges])
        idx = self.find_li_vectors(self.alpha)
        latbase = self.alpha[idx]
        counter = 0
        ### TODO: switch to other basis to make it more beautiful
        for i in range(self.nbasevec):
            if i not in idx:
                b = numpy.linalg.solve(latbase.T,self.alpha[i])
                bb = numpy.zeros(self.nedges)
                for j in range(self.dim):
                    bb += b[j]*self.cyclic_basis[idx[j]]
                k[counter] = self.cyclic_basis[i]-bb
                #print(self.get_image(k[counter]))
                counter += 1
        if self.nvertices > 1:
            k[self.nbasevec-self.dim:,:] = self.cocycle_basis[0:self.nvertices-1,:]
        self.kernel = k
        return self.kernel

    def get_cell(self):
        k = self.kernel
        L = self.lattice_basis
        S = numpy.dot(k,k.T)
        P = numpy.eye(self.nedges,self.nedges) - numpy.dot(k.T, 
                numpy.dot(numpy.linalg.inv(S), k))
        self.cell = numpy.dot(L, numpy.dot(P,L.T))
        return self.cell

    def place_vertices(self, first = numpy.array([0.0,0.0,0.0])):
        frac_xyz = numpy.zeros([self.nvertices,3])
        frac_xyz[0,:] = first
        done = [0]
        counter = 0
        while len(done) != self.nvertices:
            for i,e in enumerate(self.edges):
                if self.labels[i] == [0,0,0]:
                    if ((e[0] in done) and (e[1] not in done)):
                        #print(e, self.fracs[i,:])
                        frac_xyz[e[1],:] = (frac_xyz[e[0],:] + self.fracs[i,:])
                        done.append(e[1])
                    elif ((e[1] in done) and (e[0] not in done)):
                        nc = (frac_xyz[e[1],:] - self.fracs[i,:])
                        frac_xyz[e[0],:] = nc
                        done.append(e[0])
            counter += 1
            if counter > 10: break
            #frac_xyz = frac_xyz%1
        print(len(done))
        if len(done) != self.nvertices: 
            print('proceed')
            for i,e in enumerate(self.edges):
                if ((e[0] in done) and (e[1] not in done)):
                    print(e)
                    frac_xyz[e[1],:] = frac_xyz[e[0],:] + self.fracs[i,:]
                    done.append(e[1])
                elif ((e[1] in done) and (e[0] not in done)):
                    print(e, self.labels[i], self.fracs[i,:])
                    #### problem!!!!!
                    frac_xyz[e[0],:] = frac_xyz[e[1],:] - self.fracs[i,:]
                    done.append(e[0])
        ### perhaps a flooring has to be performe
        self.frac_xyz = frac_xyz
        return self.frac_xyz

    def to_mol(self):
        elems_map = {2:'x',3:'n',4:'s',5:'p',6:'o'}
        t = mol()
        t.natoms = self.nvertices
        t.set_cell(self.cell)
        t.set_xyz_from_frac(self.frac_xyz)
        t.set_atypes(self.nvertices*['1'])
        t.set_empty_conn()
        t.set_empty_pconn()
        for i,e in enumerate(self.edges):
            t.conn[e[0]].append(e[1])
            t.conn[e[1]].append(e[0])
            t.pconn[e[0]].append(numpy.array(self.labels[i]))
            t.pconn[e[1]].append(-1*numpy.array(self.labels[i]))
        #t.wrap_in_box()
        #t.set_elems_by_coord_number()
        t.elems = []
        for i in range(self.nvertices):
            t.elems.append(elems_map[len(t.conn[i])]) 
        return t

    def get_edge_with_idx(self, idx):
        for i in self.molg.edges():
            if self.molg.edge_index[i] == idx: return i
            #if self.molg.ep.number[i] == idx: return i

    def find_li_vectors(self,R):
        rank = numpy.linalg.matrix_rank(R)
        idx = []
        ### get first non zero vector of R
        fn = numpy.nonzero(R)[0][0]
        idx.append(fn)
        for i in range(fn+1,R.shape[0]):
            indep = True
            for j in idx:
                if i != j:
                    inner_product = numpy.dot( R[i,:], R[j,:] ) #compute the scalar product
                    norm_i = numpy.linalg.norm(R[i,:]) #compute norms
                    norm_j = numpy.linalg.norm(R[j,:])
                    if abs(inner_product - norm_j * norm_i) < 1e-4:
                        # vector i is linear dependent, iterate i
                        indep = False
                        break
            if indep == True:
                idx.append(i)
                if numpy.linalg.matrix_rank(R[idx]) != len(idx):
                    idx.pop()
                if len(idx)==rank: break
        return idx

    def vertex_positions(self, edges, used, pos={}):
        if self.dim == 2: return 'Not yet implemented'
        if len(pos.keys()) == self.nvertices: return pos
        self.molg.set_directed(True)
        for i, ed in enumerate(edges):
            e = ed
            if i == 0: break
        if int(str(e.source())) not in pos.keys() and int(str(e.target())) not in pos.keys():
            pass
        elif int(str(e.source())) not in pos.keys() or int(str(e.target())) not in pos.keys():
            from_v = int(str(e.source())) if int(str(e.source())) in pos.keys() else int(str(e.target())) 
            to_v = int(str(e.target())) if int(str(e.target())) not in pos.keys() else int(str(e.source())) 
            
            coeff = 0
            for i,ed in enumerate(self.molg.vertex(from_v).out_edges()):
                if e == ed: 
                    coeff = 1
                    break
            if coeff == 0: coeff = -1
            
            index = self.molg.ep.number[e]

            to_pos = coeff*numpy.array(self.fracs)[index] + pos[from_v]
            newedges = []

            to_pos = numpy.array([i%1 for i in to_pos])
            pos[to_v]=to_pos
            used.append(e)
            self.molg.set_directed(False)
            ee = self.molg.vertex(to_v).out_edges()
            newedges = [i for i in ee if i not in used and i not in edges]
            print(newedges)
            edges = newedges + edges[1:]
        else:
            used.append(e)
            edges = edges[1:]
        return self.vertex_positions(edges, used, pos)

    def __call__(self):
        self.build_lqg()
        self.get_cyclic_basis()
        self.get_cocycle_basis()
        self.get_B_matrix()
        self.get_alpha()
        self.get_lattice_basis()
        self.get_kernel()
        self.get_cell()
        self.get_fracs()
        self.place_vertices()
        # use the sclae computed in get_fracs() to scale
        # print (self.scale) 
        # self.cell *= self.scale
        # print (self.cell)
        return


class Systre(object):

    def __init__(self):
        self._mol = topo()
        return

    def get_cgd_info(self,reader, netname):
        info = reader._nets[netname]['cgd']
        self._mol.set_cellparams(numpy.array(map(float,info['CELL'])))
        self._mol.natoms = len(info['nodes'])
        self.nodes = numpy.array(info['nodes'])
        self._mol.set_xyz_from_frac(self.nodes)
        self._mol.set_elems(self._mol.natoms*["c"])
        self.edges = []
        for e in info['edges']:
            ed = []
            ed.append(e[0:3])
            ed.append(e[3:6])
            self.edges.append(numpy.array(ed))

    def compute_conn(self):
        conn = []
        for i in range(self._mol.natoms):
            conn.append([])
        for i,e in enumerate(self.edges):
            n1 = e[0]
            n1idx = self.find_vertex(n1)
            n2 = e[1]
            n2idx = self.find_vertex(n2)
            conn[n1idx].append(n2idx)
        self._mol.conn = conn
        self._mol.add_pconn()
            
    def find_vertex(self, pos):
        pos = numpy.array([i%1 for i in pos])
        dist = numpy.linalg.norm(self.nodes - pos,axis = 1)
        less = numpy.less(dist,0.0001)
        return numpy.where(less)[0][0]










