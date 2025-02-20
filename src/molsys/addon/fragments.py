# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:30:26 2016

@author: rochus
"""

import logging

logger = logging.getLogger("molsys.fragments")
import numpy as np

from itertools import combinations

import molsys
from molsys.util.timer import timer, Timer


class fragments:
    """
    fragments is an addon class to support advanced fragment
    handling
    """


    def __init__(self, mol):
        self.timer = Timer("fragments addon")
        self._mol = mol
        self.setup = False
        self.check()
        if self.setup:
            self.make_frag_conn()
        # a dictionary to hold derived fragment graphs
        self.fgraphs = {}
        return
    
    def report(self):
        self.timer.report()
        return

    @timer("check")
    def check(self):
        """
        check if all atoms are in a fragment and all is consistent
        """
        self.fragnames = []        # this a list of all existing fragment names
        self.frag_atoms = []       # a list of the fragments with their atoms
        self.nfrags =  max(self._mol.fragnumbers)+1
        self.fraglist  = [None]*(self.nfrags) # additional list to hold the fragments with their name
        self.frag_conn = []        # fragment connectivity (indices of fragments)
        self.frag_conn_atoms = []  # atoms making the fragemnt connectivity (tuples of atom indices: first in frgmnt, second in other fragmnt)
        for i in range(self.nfrags):
            self.frag_atoms.append([])
        if self.nfrags == 0:
            self.setup = False
            return
        self.setup = True
        for i in range(self._mol.natoms):
            ft = self._mol.fragtypes[i]
            fn = self._mol.fragnumbers[i]
            if ft == "0":
                logger.error("atom %d (%s) is not in fragment" % (i, self._mol.atypes[i]))
                self.setup=False
            else:
                if self.fraglist[fn] is None:
                    # set name of fragment
                    self.fraglist[fn] = ft
                else:
                    # check if this is the same name
                    assert self.fraglist[fn] == ft, \
                         "The fragmentname %s of atom %d does not match with the prior definition %s" % (ft, i, self.fraglist[fn])
                if ft not in self.fragnames:
                    self.fragnames.append(ft)
                self.frag_atoms[self._mol.fragnumbers[i]].append(i)
        # in the end make sure that all fragments have been named
        if None in self.fraglist:
            raise ValueError("A fragment name is missing")
        return

    def get_occurence_of_frag(self,name):
        '''
            returns the fragment count of the fragment with the given name
        '''
        return self.fraglist.count(name)

    def get_fragnames(self):
        return self.fragnames

    @timer("make_frag_conn")
    def make_frag_conn(self):
        """
        generate a fragment connectivity
        """
        assert self.setup
        # prepare the atom list for the fragments
        for i in range(self.nfrags):
            self.frag_conn.append([])
            self.frag_conn_atoms.append([])
        for i,f in enumerate(self.fraglist):
            # determine all external bonds of this fragment
            for ia in self.frag_atoms[i]:
                for ja in self._mol.conn[ia]:
                    j = self._mol.fragnumbers[ja]
                    if i != j:
                        # this is an external bond
                        self.frag_conn[i].append(j)
                        self.frag_conn_atoms[i].append((ia,ja))
                        logger.debug("fragment %d (%s) bonds to fragment %d (%s) %s - %s" %\
                                      (i, f, j, self.fraglist[j], self._mol.atypes[ia], self._mol.atypes[ja]))
        return

    # TODO(RS): put the "validator in here" ... no extra class. source is like in fragmentizer either "file" or "mofp"
    #           remove the reset_atypes stuff. we do not need this anymore
    @timer("analyze_frag_conn")
    def analyze_frag_conn(self, validator, reset_atypes=False):
        """
        detect all types of inter-fragment bonds out of frag_conn and frag_conn_atoms

        :Parameter:

            - validator: a validator obejct that validates a frag connection
            - reset_atypes (boolean): if True the fragment name (or the merged fragment name) will be appended to the atype

        """
        self.rev_fraglist = [None]*(self.nfrags)
        self.frag_bond_types= {}
        errors = 0
        # iterate over all frags and check their frag connections (if conencted frag is higher index)
        for i in range(self.nfrags):
            for nj, j in enumerate(self.frag_conn[i]):
                if j>i:
                    atom_pair = self.frag_conn_atoms[i][nj]
                    fbond = [self._mol.atypes[atom_pair[0]]+"_"+self.fraglist[i], self._mol.atypes[atom_pair[1]]+"_"+self.fraglist[j]]
                    fbond.sort()
                    fbond = ":".join(fbond)
                    if not fbond in self.frag_bond_types.keys():
                        self.frag_bond_types[fbond] = ""
                    # now check if this bond is allowed and if we need to change the fragtype in rev_fraglist
                    response = validator(fbond)
                    if response == False:
                        errors += 1
                        logger.error("No fragment connection for %s" % fbond)
                    else:
                        if response != "":
                            self.rev_fraglist[i] = response
                            self.rev_fraglist[j] = response
        if errors == 0:
            if reset_atypes:
                for i in range(self._mol.natoms):
                    f = self._mol.fragnumbers[i]
                    if self.rev_fraglist[f] is None:
                        ft = self.fraglist[f]
                    else:
                        ft = self.rev_fraglist[f]
                    self._mol.atypes[i] += "_"+ft
        return

    @timer("make_frag_conn")
    def make_frag_graph(self, add_atom_map=False, phenyl_like=["naph"]):
        """
        generate a graph of the frag_conn in analogy to the graph addon on the molecular level
        using the graph addons util_graph method

        RS: in order not to burden general runs with atom_maps it is not added by default
            ==> use the add_frag_graph() method to make a graph with atom map and store it in the dictionary
        """
        self._mol.addon("graph")
        # create here a second list of vertex types, with the aryl substituted species
        # for example a naph fragment is substituted by a ph
        vtypes2 = []
        for t in self.fraglist:
            if t in phenyl_like:
                vtypes2.append("ph")
            else:
                vtypes2.append(t)
        if add_atom_map:
            atom_map = self.frag_atoms
        else:
            atom_map = None
        self.frag_graph = self._mol.graph.util_graph(self.fraglist, self.frag_conn, vtypes2=vtypes2, atom_map=atom_map)
        # DEBUG here just for debug reasons
        #self._mol.graph.plot_graph("frag_conn", g=self.frag_graph)
        return self.frag_graph

    def add_frag_graph(self, fgname="base"):
        self.fgraphs[fgname] = self.make_frag_graph(add_atom_map=True)
        return
        
    def plot_frag_graph(self, fname, **kwargs):
        self._mol.graph.plot_graph(fname, g=self.frag_graph, **kwargs)
        return


    # depreceated .. in grag_graph we make a second vtypes2 with the "phenyl_like" species
    def upgrade(self, se, rep, rep_n = None):
        """
        upgrades the vertex labels in a frag graph
        :Parameters:
            - se  (str): vertex label to be replaced
            - rep (str): new vertex label
            - rep_n (int): optional. the amount of vertices to be replaced
        """
        assert type(se) == type(rep) == str
        assert hasattr(self, "frag_graph")
        nreplaced = 0
        for v in self.frag_graph.vertices():
            if self.frag_graph.vp.type[v] == se: 
                self.frag_graph.vp.type[v] = rep
                nreplaced += 1
                if rep_n is not None:
                    if nreplaced == rep_n:
                        return
        return


    def frags2atoms(self, frags):
        # HINT (RS) ... use atom_map and get it directly
        # TODO improve speed
        #assert type(frags) == list
        idx = []
        for i in range(self._mol.natoms):
            if self._mol.fragnumbers[i] in frags:
                idx.append(i)
        return idx


    ######## fgraphs methods  ##################################
    # 
    # the following methods operate on fragment graphs in the fgraphs dictionary
    #
    # the intrinsically rely on the availability of graphtools and the graph addon
    # 

    def merge_frags(self, source, target, mergelist, newfrag):
        """merge frags in merge list into one

        Args:
            source (string): name of fgraph to start with
            target (string): name of graph to generate (should not exist)
            mergelist (list of strings): names of frags to be merged
            newfrag (string): name of merged fragment
        """
        assert source in self.fgraphs
        assert target not in self.fgraphs
        # get copy of source graph
        g = self.fgraphs[source].copy()
        # first iterate over fragments and find which have to be merged
        merged_frag = []
        new_vertices = []
        skip_frag = []
        for i in range(g.num_vertices()):
            if i not in skip_frag:
                if g.vp.type[i] in mergelist:
                    # print ("frag %d (%s) in mergelist .. start searching" % (i, g.vp.type[i]))
                    new_frag = [i]
                    check_frag = [i]
                    skip_frag.append(i)
                    external_frag = []
                    stop = False
                    # start searching neighbors if they are also in the mergelist
                    while not stop:
                        # print ("checking fragments %s" % str(check_frag))
                        next_check_frag = []
                        for j in check_frag:
                            neig = g.get_all_neighbors(j)
                            for k in neig:
                                if g.vp.type[k] in mergelist:
                                    if (k > i) and (k not in new_frag):
                                        # print ("frag %d (%s) to append" % (k, g.vp.type[k]))
                                        new_frag.append(k)
                                        next_check_frag.append(k)
                                        skip_frag.append(k)
                                else:
                                    external_frag.append(k)
                        # done with iterating over check_frag .. next round
                        if len(next_check_frag) == 0:
                            stop = True
                        else:
                            check_frag = next_check_frag
                    # all to be merged frags (vertices) are collected in new_frag
                    # now we can make a new vertex with that info and store it for removing the obsolete vertices in the end
                    nv = g.add_vertex()
                    new_vertices.append(nv)
                    g.vp.type[nv] = newfrag # new vertex type
                    # generate a combined atom_map
                    namap = []
                    for f in new_frag:
                        namap += np.array(g.vp.atom_map[f]).tolist()
                    g.vp.atom_map[nv] = namap
                    # make edges to external frags
                    for e in external_frag:
                        g.add_edge(nv, g.vertex(e))
                    # remember frags/vertices to be deleted
                    merged_frag += new_frag
        # done with adding new merged vertices ..now remove all obsolete ones in one shot
        g.remove_vertex(merged_frag) 
        g.shrink_to_fit()
        # store in the dictionary and return the new graph
        self.fgraphs[target] = g
        return g
 
    def remove_dangling_frags(self, source, target, ignore = []):
        """remove all frags which dangling arms (kcore = 1)

        Args:
            source (string): name of fgraph to start with
            target (string): name of graph to generate (should not exist)

        """
        assert source in self.fgraphs
        assert target not in self.fgraphs
        ng = self.fgraphs[source].copy()
        kcore = self._mol.graph.get_kcore(ng)
        kfilt = ng.new_vertex_property("bool")
        kfilt.a = np.not_equal(kcore, 1)
        if ignore != []:
            for i in range(ng.num_vertices()):
                if not kfilt[i]:
                    if ng.vp.type[i] in ignore:
                        kfilt[i] == True
        ng.set_vertex_filter(kfilt)
        ng.purge_vertices()
        self.fgraphs[target] = ng
        return ng 

    def remove_bridge_frags(self, source, target, ignore=[]):
        """remove all frags with CN=2 just bridging
        Args:
            source (string): name of fgraph to start with
            target (string): name of graph to generate (should not exist)

        """
        ng = self.fgraphs[source].copy()
        skip_vertices = [] # ignore those already checked
        remove_vert =   [] # remove these in the end
        for v in ng.vertices():
            if v not in skip_vertices:
                if (v.out_degree() == 2) and (ng.vp.type[v] not in ignore) :
                    # print ("remove vert %d with type %s" % (v, ng.vp.type[v]))
                    remove_vert.append(v)
                    neighb = list(v.all_neighbors())
                    # now check on both ends if we need to extend to further cn=2 vertices
                    for d in range(2):
                        stop = False
                        while not stop:
                            nv = neighb[d]
                            if (nv.out_degree() == 2) and (ng.vp.type[nv] not in ignore):
                                # yes, this is another vertex to be deleted
                                if nv > v: skip_vertices.append(nv)
                                remove_vert.append(nv)
                                # which is the next next vertex?
                                for nnv in nv.all_neighbors():
                                    if nnv != v: neighb[d] = nnv
                            else:
                                stop = True
                    # connect neighb
                    #print ("connecting %s" % str(neighb))
                    ng.add_edge(neighb[0], neighb[1])
        # finally remove all vertices that have to go (edges go along with them)
        #print ("remove")
        #print (remove_vert)
        ng.remove_vertex(remove_vert)
        ng.shrink_to_fit()
        self.fgraphs[target] = ng
        return ng

    def remove_frag_type(self, source, target, rtype):
        """remove all frags with type rtype (and reconnect)
        Args:
            source (string): name of fgraph to start with
            target (string): name of graph to generate (should not exist)
            rtype  (string): name of type to remove

        """
        ng = self.fgraphs[source].copy()
        remove_vert =   [] # remove these in the end
        for v in ng.vertices():
            if ng.vp.type[v] == rtype:
                remove_vert.append(v)
                neighb = list(v.all_neighbors()) # we need to connect all neighbors with each other
                #print ("connecting %s" % str(neighb))
                for e in combinations(neighb, 2):
                    ng.add_edge(*e)
        ng.remove_vertex(remove_vert)
        ng.shrink_to_fit()
        self.fgraphs[target] = ng
        return ng

    def remove_cn_frags(self, source, target, cn):
        """remove all frags which have a given coordiantion number (degree)

        Args:
            source (string): name of fgraph to start with
            target (string): name of graph to generate (should not exist)

        """
        assert source in self.fgraphs
        assert target not in self.fgraphs
        ng = self.fgraphs[source].copy()
        remove_vert = [v for v in ng.vertices() if v.out_degree() == cn]
        ng.remove_vertex(remove_vert)
        ng.shrink_to_fit()
        self.fgraphs[target] = ng
        return ng 

    def report_graph(self, name):
        g = self.fgraphs[name]
        for v in g.vertices():
            print ("vertex %d (%s) with CN %d" % (v, g.vp.type[v], v.out_degree()))

    def find_subgraph(self, name, subg):
        subs = self._mol.graph.find_subgraph(self.fgraphs[name], subg)
        return subs

    def fgraph_to_mol(self, name, elemmap, subg=None):
        # if subg is given it is assumed to be a list of vertices, which are taken instead of "all" = fg.vertices()
        xyz = self.fgraph_get_coms(name)
        fg = self.fgraphs[name]
        if subg != None:
            filt = fg.new_vertex_property("bool")
            filt.set_value(False)
            for i in subg:
                filt[fg.vertex(i)] = True
            fg.set_vertex_filter(filt)
        atypes = [fg.vp.type[v] for v in fg.vertices()]
        natoms = len(atypes)
        elems = [elemmap[at] for at in atypes]
        if subg != None:
            xyz = xyz[subg]
        m = molsys.mol.from_array(xyz)
        m.set_elems(elems)
        m.set_atypes(atypes)
        # now generate connectivity
        conn = [[] for i in range(natoms)]
        for e in fg.edges():
            i, j = int(e.source()), int(e.target())
            if subg != None:
                i = subg.index(i)
                j = subg.index(j)
            conn[i].append(j)
            conn[j].append(i)
        m.set_conn(conn)
        if subg != None:
            fg.clear_filters()
        return m

    def fgraph_get_coms(self, name):
        """get an array with COMs of the fraggraph fragments

        This function works currently not in PBC

        Args:
            name (string): name of the fgraph to use
        """
        # assert self._mol.bcond == 0
        fg = self.fgraphs[name]
        # compute the xyz positions of the vertices from the fragments atom_maps
        mxyz = self._mol.get_xyz()
        self._mol.set_real_mass()
        mmass = np.array(self._mol.get_mass()) # make a numpy array from the list to allow index lookup
        xyz = np.zeros([fg.num_vertices(),3], dtype="float64")
        for v in fg.vertices():
            am = list(fg.vp.atom_map[v])
            fxyz = mxyz[am]
            fmass = mmass[am]
            com = (fxyz*fmass[:,np.newaxis]).sum(axis=0)/fmass.sum()
            xyz[int(v)] = com
        return xyz

