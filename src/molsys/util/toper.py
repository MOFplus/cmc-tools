#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import molsys
from graph_tool import Graph, GraphView
from graph_tool.topology import *
import numpy as np
import copy
from mofplus import user_api
from string import ascii_lowercase
from collections import Counter, defaultdict

from molsys.util.color import make_mol, vcolor2elem
from molsys.util.sysmisc import _makedirs, _checkrundir
from molsys.util.misc import argsorted, triplenats_on_sphere

import logging
logger = logging.getLogger("molsys.toper")

class conngraph:
    """ This is the "conngraph" class
    It is the basis of the molgraph and topograph classes, which use graph theory for the deconstruction of
    MOF structures, or the analysis of topologies, respectively. It provides the important ground functions
    used by both classes, like make_graph, flood_fill and center.
    """

    def __init__(self, mol):
        """
        :Parameters:
        - mol: molsys.mol or molsys.topo object
        """
        self.mol = mol
        if self.mol.periodic and not self.mol.use_pconn:
            self.mol.add_pconn()
        self.make_graph()

    def __contains__(self, other):
        return subgraph_isomorphism(
            other.molg, self.molg,
            vertex_label=(other.molg.vp.elem, self.molg.vp.elem),
            subgraph=True, max_n=1
        )

    def split(self, other=None, duplicates=True, **kwargs):
        """
        :Argument:
        - other(conngraph=None): other conngraph (as subgraph) used as graph
            separator. If other is None: a standard component split is performed

        :Return:
        - cgs(list of conngraphs): splitted conngraphs
        """
        ### split graph ###
        gws = self.split_graph(other=other, **kwargs)
        ### remove duplicates ###
        # N.B.: graphs must be pruned according to graph view
        if not duplicates:
            ugws = []
            for gw in gws:
                p_gw = Graph(gw, prune=True)
                found = False
                for ugw in ugws:
                    p_ugw = Graph(ugw, prune=True)
                    iso = subgraph_isomorphism(p_gw, p_ugw,
                        vertex_label=(p_gw.vp.elem, p_ugw.vp.elem), subgraph=False, max_n=1)
                    if iso:
                        found = True
                        break
                if not found:
                    ugws.append(gw)
            gws = ugws
        ### extract molecules ###
        ms = []
        for gw in gws:
            m = self.extract_mol_by_subgraph(gw)
            ms.append(m)
        ### split to new conngraphs ###
        cgs = []
        for m in ms:
            cg = self.__class__(m)
            cgs.append(cg)
        return cgs

    def split_graph(self, other=None, **kwargs):
        """
        :Caveat:
        - only regioisomerism detected
        """
        vfilt = self.molg.new_vertex_property('bool')
        vfilt.a = True
        if other is None:
            umolg = self.molg
        else:
            vfilt, vubs = self.label_subgraphs(other)
            umolg = GraphView(self.molg, vfilt=vfilt)
        # filtered graph is splitted into components
        components, histograms = label_components(umolg)
        # -1 as separator and 0 as first component (only when other is not None)
        components.a = components.a + vfilt.a -1
        gws = []
        for c in np.unique(components.a):
            if c == -1: # if filtered (only when other is not None)
                continue
            gw = GraphView(umolg, vfilt=components.a == c)
            gws.append(gw)
        return gws

    def label_subgraphs(self, cg=None, **kwargs):
        """
        Label graph with subgraph/s
        N.B.: if cg is iterable, then order of index is order of priority
        Example:
            if paddlewheel comes before the carboxylate, then
            paddlewheels and carboxylates out of the paddlewheels are found.
            if carboxylate comes before the paddlewheel, then
            carboxylates (also in paddlewheels) are found and not paddlewheels.

        :Return:
            vfilt (vprop): 1 where (collective) match, 0 else
            vsubs (vprop): index where match for ordered subgraph, -1 else
        """
        if cg is None:
            cg = self
        vsubs = self.molg.new_vertex_property('int64_t')
        vsubs.a = -1 # default: no match found, not visited by subgraphs
        vfilt = self.molg.new_vertex_property('bool')
        vfilt.a = 1 # to prevent further visiting of the subgraph isomorphism
        # multiple case #
        if hasattr(cg, "__iter__"):
            for i,icg in enumerate(cg):
                ivfilt, ivsubs = self.label_subgraphs(icg)
                vsubs.a[ivfilt.a==0] = i + ivsubs.a[ivfilt.a==0] # found
            vfilt.a[vsubs.a!=-1] = 0
            return vfilt, vsubs
        # single case #
        cs = 0
        while True: # infinite while until no subgraph is isomorphic
            # cg.molg serves as separator as in str.split
            umolg = GraphView(self.molg, vfilt=vfilt)
            subs = subgraph_isomorphism(cg.molg, umolg,
                vertex_label=(cg.molg.vp.elem, self.molg.vp.elem), subgraph=True, max_n=1)
            if subs:
                # a match is found and filtered
                sub = subs[0]
                vfilt.a[sub.a] = 0
                vsubs.a[sub.a] = cs
                cs += 1
            else:
                # last match it filtered
                break
        return vfilt, vsubs

    def label_subgraph_neighbours(self, vsubs):
        nsubs = self.molg.new_vertex_property('int64_t')
        nsubs.a = -2 # -2 -> not connector; -1 -> connector of unknown subgraph
        for v in self.molg.vertices():
            vsub = vsubs[v]
            ns = v.all_neighbours()
            for n in ns:
                nsub = vsubs[n]
                if nsub != vsub: # then v is a connector!
                    nsubs[v] = nsub
        return nsubs

    def label_subgraph_edges(self, vsubs):
        efilt = self.molg.new_edge_property("bool")
        efilt.a = 1
        esubs = self.molg.new_edge_property("vector<int64_t>")
        ekinds = set([])
        for e in self.molg.edges():
            vsource = vsubs[e.source()]
            vtarget = vsubs[e.target()]
            if vsource > vtarget:
                vtarget, vsource = vsource, vtarget
            ekind = (vsource, vtarget)
            if vsource != vtarget:
                ekinds |= set((ekind,))
                efilt[e] = 0
            esubs[e] = ekind
        return efilt, esubs

    def label_subgraph_components(self, efilt=None):
        molgw = GraphView(self.molg, efilt=efilt)
        components, histogram = label_components(molgw)
        return components, histogram

    def label_subgraph_elementsequences(self, nsubs, components):
        vfilt = self.molg.new_vertex_property("bool")
        vfilt.a = 1
        elems = [self.molg.vp.elem[v] for v in self.molg.vertices()]
        elems = np.array(elems)
        elemseqs = defaultdict(list)
        for i,c in enumerate(nsubs.a):
            if c == -2: #else: it is a connector
                continue
            # apply only on subgraph
            comp = components.a[i]
            vfilt.a[components.a == comp] = 1
            vfilt.a[components.a != comp] = 0
            molgw = GraphView(self.molg, vfilt=vfilt)
            # distances from connector to the subgraph
            dists = shortest_distance(molgw, i)
            dists.a[dists.a==0] = -1 # filtered out
            dists.a[i] = 0 # source
            cond = dists.a > -1 # where considered
            idists = dists.a[cond]
            ielems = elems[cond]
            # element sequence
            distelems = zip(idists, ielems)
            counter = Counter(distelems)
            udistelems = sorted(counter)
            udistseq = [ui[0] for ui in udistelems]
            distcounter = Counter(udistseq)
            # string sequence
            lfmt = ["" for di in distcounter]
            for ue in udistelems:
                lfmt[ue[0]] += "%s%s" % (ue[1], counter[ue])
            elemseq = "_".join(lfmt)
            elemseqs[elemseq].append(i)
        elemseqs = dict(elemseqs)
        return elemseqs

    def label_subgraph_connectors(self, vsubs, **kwargs):
        # label inter-subgraph edges
        efilt, _ = self.label_subgraph_edges(vsubs)
        # cut inter-subgraph edges
        components, _ = self.label_subgraph_components(efilt)
        # neighbour subgraphs
        nsubs = self.label_subgraph_neighbours(vsubs)
        # make string sequence of elements per each connector
        elemseqs = self.label_subgraph_elementsequences(nsubs, components)
        return nsubs, components, elemseqs

    def label_subgraph_blocks(self, components, elemseqs):
        ### TO-DO: Distinguish by connectors (regiochemistry, topological)
        ### TO-DO: Distinguish by coordinates (stereochemistry, geometric)
        molg = self.molg.copy() # to change molg.vp.elem
        for eseq in elemseqs:
            idxs = elemseqs[eseq]
            for i in idxs:
                molg.vp.elem[i] = eseq
        cvfilt = self.molg.new_vertex_property("bool")
        cvfilt.a = 1
        svfilt = self.molg.new_vertex_property("int64_t")
        svfilt.a = -1
        sblocks = []
        for uc in np.unique(components.a):
            cvfilt.a[components.a == uc] = 1
            cvfilt.a[components.a != uc] = 0
            molgw = GraphView(molg, vfilt=cvfilt)
            match = [] # first block not found by construction, so add to kind of blocks
            for ksbb in sblocks:
                p_ksbb = Graph(ksbb, prune=True)
                p_molgw = Graph(molgw, prune=True)
                match = subgraph_isomorphism(p_ksbb, p_molgw,
                    vertex_label=(p_ksbb.vp.elem, p_molgw.vp.elem), subgraph=False, max_n=1)
                if match:
                    break
            if not match:
                svfilt.a[components.a == uc] = len(sblocks) # starts from zero
                sblocks.append(molgw.copy()) # copy is needed otherwise view is the same
        return sblocks, svfilt

    def extract_mol_by_subgraph(self, cg, **kwargs):
        """
        Extract mol object out of connectivity graph

        :Parameters:
        cg(conngraph): connectivity graph or graph view

        :Return:
        m(mol): molecular object
        """
        idxs = [int(v) for v in cg.vertices()]
        m = self.mol.new_mol_by_index(idxs)
        return m

    def make_block_by_subgraph(self, subgraph, elemseqs):
        bb = self.extract_mol_by_subgraph(subgraph)
        bb.add_pconn()
        bb.addon("bb")
        bb.is_bb = True
        # connector type
        elems = [subgraph.vp.elem[v] for v in subgraph.vertices()]
        bbc = []
        bbca = []
        bbct = []
        for i,e in enumerate(elems):
            if e not in elemseqs:
                continue
            bbc.append(i)
            bbca.append([i])
            bbct.append(e)
        ct, ca = bb.bb.sort_connectors_type(bbct)
        bb.connectors_type = ct
        bb.connectors = [bbc[i] for i in ca]
        bb.connector_atoms = [bbca[i] for i in ca]
        bb.center_point = 'coc'
        bb = bb.unwrap_box()
        return bb

    def extract_blocks(self, cgs, folder="bbs", **kwargs):
        _, vsubs = self.label_subgraphs(cgs)
        nsubs, components, elemseqs = self.label_subgraph_connectors(vsubs)
        subblocks, subvfilt = self.label_subgraph_blocks(components, elemseqs)
        bbs = []
        _makedirs(folder)
        for i, subb in enumerate(subblocks):
            bb = self.make_block_by_subgraph(subb, elemseqs)
            bb.write(("%%s/%%%dd.mfpx" % len(str(len(subblocks)))) % (folder, i))
            bbs.append(bb)
        return bbs

    def make_graph(self, forbidden = []):
        """ Create a Graph object from a molsys.mol object. """
        self.molg = Graph(directed=False)
        ig = 0
        # setup vertices
        self.molg.vp.fix = self.molg.new_vertex_property("int64_t")
        self.molg.vp.midx = self.molg.new_vertex_property("int64_t")
        self.molg.vp.elem = self.molg.new_vertex_property("string")
        self.molg.vp.atype = self.molg.new_vertex_property("string")
        self.molg.vp.coord = self.molg.new_vertex_property("vector<double>")
        self.molg.vp.filled = self.molg.new_vertex_property("bool") # boolean for flood fill
        for i in range(self.mol.natoms):
            ig = self.molg.add_vertex()
            self.molg.vp.coord[ig] = self.mol.xyz[i,:]
            self.molg.vp.elem[ig] = self.mol.elems[i]
            self.molg.vp.atype[ig] = self.mol.atypes[i]
            self.molg.vp.midx[ig] = i
            if int(ig) in forbidden:
                self.molg.vp.fix[ig] = 1
            else:
                self.molg.vp.fix[ig] = 0
        # setup edges
        self.molg.ep.act = self.molg.new_edge_property("bool")
        self.molg.ep.Nk = self.molg.new_edge_property("int64_t")
        for i in range(self.mol.natoms):
            for j in self.mol.conn[i]:
                if j > i:
                    e = self.molg.add_edge(self.molg.vertex(i),self.molg.vertex(j))
                    self.molg.ep.act[e] = True
        # create Backup of the original graph for comparison
        self.keep = Graph(self.molg, directed=False)
        return

    def make_graph_from_morphism(self, morph):
        """make graph from morphism applied to current graph
        iold,jold TO BE CHECKED"""
        mol = copy.deepcopy(self.mol)
        morpha = morph.get_array()
        mol.xyz = mol.xyz[morpha]
        mol.elems = np.array(mol.elems)[morpha].tolist()
        mol.atypes = np.array(mol.atypes)[morpha].tolist()
        conn = copy.deepcopy(self.mol.conn)
        pconn = copy.deepcopy(self.mol.conn)
        mol.set_empty_conn()
        mol.set_empty_pconn()
        for iold,jold in self.mol.ctab:
            # apply morphism to indices
            i = morpha[iold]
            j = morpha[jold]
            # get ijth position of j in ith conn
            ij = conn[i].index(j)
            # pop index corresponding to j from ith conn (and check)
            assert conn[i].pop(ij) == j
            # append index to new ith conn
            mol.conn[i].append(j)
            # pop image corresponding to j from ith pconn
            jp = pconn[i].pop(ij)
            # append image to new ith pconn
            mol.pconn[i].append(jp)
            # same story here with switched indices i,j<-j,i
            ji = conn[j].index(i)
            assert conn[j].pop(ji) == i
            mol.conn[j].append(i)
            ip = pconn[j].pop(ji)
            mol.pconn[j].append(ip)
        mol.set_ctab_from_conn(pconn_flag=True)
        self.mol.set_etab_from_tabs()
        cg = self.__class__(mol) # new conngraph
        return cg
    
    def cut_to_2core(self):
        """
        Cuts graph to its 2-core
        """
        k = kcore_decomposition(self.molg).get_array()
        idxlist = np.argwhere(k==1).tolist()
        idx = []
        for i in idxlist:
            if self.molg.vp.fix[i[0]] == 0:
                idx.append(i[0])
        for v in reversed(sorted(idx)):
            self.molg.remove_vertex(v)
        return idxlist

    def remove_2conns(self):
        """
        From self.molg, removes all vertices with 2 connecting edges.
        
        Returns: boolean if the method could find any 2-connected vertices.
        """
        found_2conns = False
        for v in self.molg.vertices():
            if self.molg.vp.fix[v] == 0:
                neighbours = []
                for i in v.out_neighbours(): neighbours.append(i)
                if len(neighbours) == 2:
                    found_2conns = True
                    if not self.molg.edge(neighbours[0], neighbours[1]):
                        self.molg.add_edge(neighbours[0], neighbours[1])
                    self.molg.remove_vertex(v)
                    self.remove_2conns()
                    break
        return found_2conns
    
    def handle_islands(self, thresh_small=0.2, silent=False):
        """
        Handles parts of the structure which are not connected to the rest ("islands").
        
        :Parameters:
        - thresh_small: If the island is smaller than the size of the largest island multiplied by
          thresh_small, it will be deleted.
        - silent: if True, a warning will be printed out when there are multiple large islands.
        
        :Returns:
        - n_removed_atoms: Number of removed atoms
        - multiple_large_islands: boolean if there are multiple large islands
        """
        multiple_large_islands = False
        islands = self.get_islands()
        if len(islands) < 2:
            # we only have 1 island, so nothing has to be done
            return 0, False
        else:
            # ok we have multiple islands, time to find out what we have here.
            # first we sort the islands by size
            island_sizes = list(map(len, islands))
            sort_indices = np.array(island_sizes).argsort()
            biggest_size = island_sizes[sort_indices[-1]]
            remove_list = []
            # then we compare the sizes of the smaller islands with the size of the largest island
            for index, i in enumerate(island_sizes):
                # don't compare the biggest island with itself though...
                if index != sort_indices[-1]:
                    ratio = float(i)/float(biggest_size)
                    if ratio < thresh_small:
                        for v in reversed(sorted(islands[index])):
                            remove_list.append(v)
                    else:
                        multiple_large_islands = True
                        if not silent: 
                            print("Warning: Two large parts of the structure are not connected to each other. Check the structure.")
            # DELETE THE SMALL ISLANDS LIKE GLOBAL WARMING !!!!
            self.molg.remove_vertex(remove_list)
            n_removed_atoms = len(remove_list)
            return n_removed_atoms, multiple_large_islands
    
    def get_islands(self):
        """ Finds all islands. """
        self.molg.vp.filled.set_value(False)
        remain_list = list(range(self.molg.num_vertices()))
        islands = []
        while len(remain_list) > 0:
            l = self.flood_fill(self.molg, self.molg.vertex(remain_list[0]), [])
            #print(len(l) # debug message - prints sizes of all islands...)
            islands.append(l)
            for i in l:
                remain_list.remove(int(i))
        return islands
        
    def flood_fill(self, graph, vertex, return_list=[]):
        """
        Uses flood fill to find all vertices that are connected to the starting vertex.
        Caution: You might want to reset the vertex property "filled" before calling flood_fill!
        
        :Parameters:
        - vertex: starting vertex
        - return_list: list of vertices which have already been iterated (when calling the function, you might have to force this to be [])
        - graph: The graph in which flood fill should be performed
        
        :Returns:
        - list of all vertices that could be reached by flood fill.
        """
        # perform flood fill
        graph.vp.filled[vertex] = True
        return_list.append(vertex)
        for n in vertex.all_neighbours():
            if graph.vp.filled[n] == False:
                self.flood_fill(graph, n, return_list)
        return return_list

    def get_automorphisms(self, color=None, **kwargs):
        """Obtain all automorphisms of conngraph graph
        An automorphism is an isomorphism of the graph with itself.
        WARNING: the procedure may be slow, especially for uncolored nets.

        >>> autos_default = tt.tg.get_automorphisms()
        >>> autos_nocolor = tt.tg.get_automorphisms(color=False)
        >>> autos_justone = tt.tg.get_automorphisms(color=False, max_n=1)
        """
        molg = self.molg
        if color is None:
            color = hasattr(molg.ep, "color")
        if color:
            return subgraph_isomorphism(molg, molg,
                vertex_label=(molg.vp.elem, molg.vp.elem),
                edge_label=(molg.ep.color, molg.ep.color),
                subgraph=False, **kwargs)
        else:
            return subgraph_isomorphism(molg, molg,
                vertex_label=(molg.vp.elem, molg.vp.elem),
                subgraph=False, **kwargs)

    def is_isomorphic(self, other, color=None, **kwargs):
        """Return True if the instance graph is isomorphic to a target graph.
        Check the existence of at least an isomorphism between the
        instance conngraph graph and another conngraph graph.
        
        >>> tt.tg.is_isomorphic(tt.tg) # trivial
        True
        
        >>> tt.tg.is_isomorphic(tt.mg) # trivial (most of the time)
        False
        """
        molg1, molg2 = self.molg, other.molg
        if color is None:
            color = hasattr(molg1.ep, "color") & hasattr(molg2.ep, "color")
        if color:
            try:
                isom = subgraph_isomorphism(molg1, molg2,
                    vertex_label=(molg1.vp.elem, molg2.vp.elem),
                    edge_label=(molg1.ep.color, molg2.ep.color),
                    subgraph=False, max_n=1, **kwargs)
                return bool(isom) # is there any isomorphism? True/False
            except KeyError as e:
                if color is not None: # gentle: not isomorphic w/o colors
                    return False
                else: # something went wrong and unexpected
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)
        else:
            # is there at least one (=any) isomorphism? True/False
            #   isom is a list of zero or one element, and its boolean
            #   value is True whether it is not empty
            isom = subgraph_isomorphism(molg1, molg2,
                vertex_label=(molg1.vp.elem, molg2.vp.elem),
                subgraph=False, max_n=1, **kwargs)
            # N.B.: the following method
            #   `isomorphism(molg1, molg2, **kwargs)`
            # is NOT enough for taking vertex/edge labelling into account
            return bool(isom)

    def print_isomorphism(self, iso, vertex=None, edge=None):
        """Print isomorphism as map of indices"""
        ### TBI: separate RETURN from PRINT! (that's needed for ACAB)
        if vertex is None and edge is None:
            vertex = True
            edge = True
        molg = self.molg
        if vertex:
            vs_ = [iso[v] for v in molg.vertices()]
            vfd = len(str(len(vs_)-1))
            vs_f = ["%%%dd" % vfd % v_ for v_ in vs_]
            print("V:", ' '.join(vs_f))
        if edge:
            # TBI?: there should be a faster, more graph_tool-like way
            es_ = []
            for e in molg.edges():
                s,t = e.source(), e.target()
                s_, t_ = iso[s], iso[t]
                i_ = molg.edge_index[s_,t_]
                es_.append(i_)
            efd = len(str(len(es_)-1))
            es_f = ["%%%dd" % efd % e_ for e_ in es_]
            print("E:", ' '.join(es_f))
        return
    
class molgraph(conngraph):
    """ This class handles the deconstruction of a MOF structure into a graph. """
    
    def determine_Nk(self):
        """ Assigns size of minimal ring to every edge """
        i = 0
        for e in self.molg.edges():
            i += 1
            self.molg.ep.act[e] = False
            self.molg.set_edge_filter(self.molg.ep.act)
            dist = shortest_distance(self.molg, source=e.source(), target= e.target())
            self.molg.ep.act[e] = True
            if dist < 2147483647:
                self.molg.ep.Nk[e] = dist+1
            else:
                self.molg.ep.Nk[e] = 0
            #print(self.molg.ep.Nk[e])
        return

    def find_cluster_threshold(self):
        """
        Finds thresholds.
        Needs Nk values of the edges -> determine_Nk() has to be called before calling this method.
        """
        self.threshes = []
        Nk = [i for i in self.molg.ep.Nk.get_array() if i != 0]
        Nk.sort()
        for i in range(len(Nk)-1):
            if Nk[i+1]-Nk[i]>2:
                self.threshes.append(Nk[i+1])
        return self.threshes

    def get_clusters(self, remove_side_chains=False):
        """
        Get the clusters of the MOF.
        
        The clusters (saved in self.clusters) are lists of atoms, which belong in one group or form 
        one 'building block'. They are firstly defined by ring sizes: If the Nk value is higher or equal 
        than the threshold, this bond will be defined as an inter-cluster bond.
        The clusters can be later modified to create different representations of the framework, for 
        example by summarizing all organic clusters which are bonded to each other.
        """
        try:
            assert self.threshes
        except:
            self.find_cluster_threshold()
        # remove side chains (unnecessary at the moment)
        if remove_side_chains:
            def forloop():
                broken = False
                for e in self.molg.edges():
                    # since the indices of the edges will change when one edge is removed, the for loop has to be
                    # restarted every time an edge is deleted, or else there will be problems when the next edge 
                    # should be deleted...
                    if self.molg.ep.Nk[e] == 0:
                        # found a sidechain... now use flood_fill to find out which side of the
                        # edge belongs to the sidechain and which to the main framework
                        src = e.source()
                        trg = e.target()
                        self.molg.remove_edge(e)
                        self.molg.vp.filled.set_value(False)
                        struc1 = self.flood_fill(self.molg, src, [])
                        struc2 = self.flood_fill(self.molg, trg, [])
                        # the larger structure is the framework
                        if len(struc1) < len(struc2):
                            for i in reversed(sorted(struc1)):
                                self.molg.remove_vertex(i)
                        elif len(struc2) < len(struc1):
                            for i in reversed(sorted(struc2)):
                                self.molg.remove_vertex(i)
                        else:
                            # both structures have same size
                            pass
                        broken = True
                        break
                return broken
            while forloop():
                pass
        ### set edge filter
        self.molg.vp.filled.set_value(False)
        self.molg.ep.act.set_value(True)
        if self.threshes != []:
            thresh = self.threshes[0]
        else:
            thresh = 0
        for e in self.molg.edges():
            if self.molg.ep.Nk[e] >= thresh:
                self.molg.ep.act[e] = False
        self.molg.set_edge_filter(self.molg.ep.act)
        ### perform flood fill
        clusters = []
        while False in list(self.molg.vp.filled.get_array()):
            vidx = list(self.molg.vp.filled.get_array()).index(0)
            vstart = self.molg.vertex(vidx)
            cluster = self.flood_fill(self.molg, self.molg.vertex(vstart), [])
            cluster = list(map(int, cluster))
            clusters.append(cluster)
        self.molg.clear_filters()
        self.clusters = clusters
        return clusters

    def get_unique_clusters(self):
        """
        get unique clusters
        return their cluster types
        """
        try:
            clusters = self.clusters
        except AttributeError:
            clusters = self.get_clusters()
        logger.info("Get unique building blocks which are distinct by connectivity")
        ms = [self.mol.new_mol_by_index(c) for c in clusters]
        mgs = [molgraph(m) for m in ms]
        molgs = [mg.molg for mg in mgs]
        unique_ms = []
        type_ms = []
        for i,molgi in enumerate(molgs):
            new = True
            for j in unique_ms:
                molgj = molgs[j]
                iso = subgraph_isomorphism(molgi, molgj, vertex_label=(molgi.vp.elem, molgj.vp.elem), max_n=1,
                        subgraph=False)
                if iso != []:
                    new = False
                    break
            if new:
                type_ms.append(len(unique_ms))
                unique_ms.append(i)
            else:
                type_ms.append(type_ms[j])
        return type_ms
    
    def get_cluster_atoms(self):
        """
        Get all atoms in the clusters of the conngraph.
        Needs clusters in self.clusters -> call get_clusters before calling this method!
        
        :Returns: 
        -List of list of vertices of each cluster in the backup "self.keep" graph.
        -List of list of atom ids of each cluster
        """
        try:
            assert self.clusters
        except:
            self.get_clusters()
        midx_list = self.keep.vp.midx.get_array().tolist()
        # set edge filter
        for i, c in enumerate(self.clusters):
            cluster_atoms = self.clusters[i]
            for ia in cluster_atoms:
                via = self.molg.vertex(ia)
                for j in via.all_neighbours():
                    if j not in cluster_atoms:
                        # found external bond, set edge filter here
                        midx = (self.molg.vp.midx[via], self.molg.vp.midx[j])
                        atomid = (midx_list.index(midx[0]), midx_list.index(midx[1]))
                        keepedge = self.keep.edge(atomid[0], atomid[1])
                        self.keep.ep.act[keepedge] = False
        self.keep.set_edge_filter(self.keep.ep.act)
        # then find all connecting atoms
        clusters_vertices = []
        clusters_atoms = []
        self.keep.vp.filled.set_value(False)
        for ic, c in enumerate(self.clusters):
            for vid in c:
                v = self.molg.vertex(vid)
                midx = self.molg.vp.midx[v]
                atomid = midx_list.index(midx)
                atom = self.keep.vertex(atomid)
                if self.keep.vp.filled[atom] == False:
                    this_cluster_verts = self.flood_fill(self.keep, atom, [])
                    clusters_vertices.append(this_cluster_verts)
                    this_cluster_atoms = []
                    for i in this_cluster_verts:
                        this_cluster_atoms.append(self.keep.vp.midx[i])
                        # set atomtypes in self.mol
                        self.mol.atypes[self.keep.vp.midx[i]] = ic
                    clusters_atoms.append(this_cluster_atoms)
        self.keep.clear_filters()
        return clusters_vertices, clusters_atoms
    
    def cluster_conn(self):
        """
        Helper function that returns a list, which describes, which clusters are connected with each other.
        """
        assert self.clusters
        cluster_conn = []
        for i, cluster_atoms in enumerate(self.clusters):
            this_cluster_conn = []
            ext_bond = []
            for ia in cluster_atoms:
                via = self.molg.vertex(ia)
                for j in via.all_neighbours():
                    if j not in cluster_atoms:
                        ext_bond.append(int(str(j)))
            #print("cluster %s consisting of %d atoms is %d times connected" % (str(i), len(cluster_atoms), len(ext_bond)))
            # now check to which clusters these external bonds belong to
            for ea in ext_bond:
                for ji, j in enumerate(self.clusters):
                    if ea in j:
                        this_cluster_conn.append(ji)
            #            print(" -> bonded to cluster ", ji)
                        break
            cluster_conn.append(this_cluster_conn)
        return cluster_conn
    
    def get_bbs(self):
        """
        Returns the building blocks (BBs) of the MOF.
        Since the definition of "building block" is arbitrary, this function will do the following things:
        - if multiple 2-connected clusters are connected to each other, these clusters will be summarized
          as one BB.
        The function will OVERWRITE self.clusters and replace it with the newly generated building blocks!
        """
        try:
            assert self.clusters
        except:
            self.get_clusters()
        # summarize 2-connected clusters:
        stop = False
        while not stop:
            stop = True
            # find external bonds
            cluster_conn = self.cluster_conn()
            # find out if 2-connected clusters are bonded to other 2-connected clusters
            for i, c in enumerate(cluster_conn):
                if len(c) == 2:
                    for b in c:
                        if len(cluster_conn[b]) == 2:
                            # and if they are, create new clusters which contain everything the first clusters contained
                            self.clusters.append(self.clusters[i] + self.clusters[b])
                            # and then remove those clusters
                            if i > b:
                                del self.clusters[i]
                            del self.clusters[b]
                            if i < b:
                                del self.clusters[i]
                            stop = False
                            break
                if not stop:
                    break
        return self.clusters
    
    def detect_organicity(self, organic_elements = ["h", "b", "c", "n", "o", "f", "si", "p", "s", "cl", "as", "se", "br", "i"]):
        """
        Finds out whether a cluster classifies as "organic" or "inorganic".
        
        :Parameters:
        - organic_elements: This is a list which defines, which elements are allowed inside
          an "organic" building block. All element symbols must be lower case. Default:
          H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I
          
        :Returns:
        - self.cluster_organicity: list, each element is True (for organic) or False (for 
          inorganic), and the index is equivalent to the index of the cluster in self.clusters
        """
        try:
            assert self.clusters
        except:
            self.get_clusters()        
        # go through each atom in every cluster and compare the elements
        self.cluster_organicity = []
        for cluster in self.clusters:
            org = True
            for i in cluster:
                v = self.molg.vertex(i)
                midx = self.molg.vp.midx[v]
                if self.mol.elems[midx].lower() not in organic_elements:
                    org = False
                    break
            self.cluster_organicity.append(org)
        return self.cluster_organicity
    
    def get_bbs_by_organicity(self):
        """
        Groups all organic clusters, which are connected with each other, into a single cluster.
        Like self.get_bbs, this method will OVERWRITE self.clusters.
        """
        logger.info('Collapse clusters in grouped superclusters')
        if not hasattr(self,'cluster_organicity'):
            self.detect_organicity()
        stop = False
        while not stop:
            stop = True
            # find external bonds
            cluster_conn = self.cluster_conn()
            # find out if organic clusters are bonded to other organic clusters
            for i, c in enumerate(cluster_conn):
                if self.cluster_organicity[i] == True:
                    for b in c:
                        if self.cluster_organicity[b] == True:
                            # and if they are, create new clusters which contain everything the first clusters contained
                            self.clusters.append(self.clusters[i] + self.clusters[b])
                            # and then remove those clusters
                            if i > b:
                                del self.clusters[i]
                            del self.clusters[b]
                            if i < b:
                                del self.clusters[i]
                            stop = False
                            # recompute organicity, because of the changed indices
                            self.detect_organicity()
                            break
                if not stop:
                    break
        return self.clusters

    def make_topograph(self, verbose=True, allow_2conns=False, color_clusters=True):
        """
        Create the topograph of the topology of the MOF.
        
        :Parameters:
        - verbose: if True, there will be some information printed out
        - allow_2conns: if True, 2-connected vertices will be allowed in the topograph
        """
        try:
            assert self.clusters
        except (AttributeError, AssertionError) as e:
            self.get_clusters()
        if len(self.clusters) == 1:
            if verbose: print("only one cluster is found!")
            tm = copy.deepcopy(self.mol)
            if color_clusters:
                type_clusters = self.get_unique_clusters()
                tm.elems = [vcolor2elem[i] for i in type_clusters]
            tg = topograph(self.mol, allow_2conns)
            tg.make_graph()
            if verbose: print(self.threshes)
            return tg
        tm = molsys.mol()
        tm.force_topo()
        tm.natoms = len(self.clusters)
        tm.set_empty_conn()
        xyz = []
        elems = []
        for i, cluster_atoms in enumerate(self.clusters):
            ext_bond = []
            cidx = []
            for ia in cluster_atoms:
                cidx.append(self.molg.vp.midx[ia])
                via = self.molg.vertex(ia)
                for j in via.all_neighbours():
                    if j not in cluster_atoms:
                        # thus bond is an external bond
                        ext_bond.append(int(str(j)))
            xyz.append(self.mol.get_com(cidx))
            #xyz.append(self.center(cxyz))
            if verbose: print("cluster %s consisting of %d atoms is %d times connected" % (str(i),
                    len(cluster_atoms), len(ext_bond)))
            # now check to which clusters these external bonds belong to
            for ea in ext_bond:
                for ji, j in enumerate(self.clusters):
                    if ea in j:
                        if verbose: print(" -> bonded to cluster ", ji)
                        tm.conn[i].append(ji)
                        break
        for i in range(tm.natoms):
            if len(tm.conn[i]) == 4:
                elems.append('c')
            elif len(tm.conn[i]) == 2:
                elems.append('o')
            else:
                elems.append('n')
        ### check for consistence of conn
            for j in tm.conn[i]:
                if j>i:
                    if not i in tm.conn[j]:
                        if verbose: print("Fragment topology is inconsitent")
        tm.set_xyz(np.array(xyz))
        tm.set_elems(elems)
        tm.set_atypes(tm.natoms*['0'])
        cell = self.mol.cell
        if cell is not None:
            tm.set_cell(cell)
        else:
            tm.set_wrapping_cell()
        tm.ctab = tm.get_conn_as_tab(pconn_flag=False)
        tm.add_pconn()
        tm.set_ctab_from_conn(pconn_flag=True)
        tm.set_etab()
        if color_clusters:
            type_clusters = self.get_unique_clusters()
            tm.elems = [vcolor2elem[i] for i in type_clusters]
        tg = topograph(tm, allow_2conns)
        tg.make_graph()
        if verbose: print(self.threshes)
        return tg
    

class topograph(conngraph):
    """ This class handles the analysis of nets using graph theory. """
    
    def __init__(self, mol, allow_2conns=False):
        """
        :Parameters:
        - mol: molsys.topo object
        - allow_2conns: if False, all 2-connected vertices will be deleted.
        """
        self.mol = mol
        if not allow_2conns:
            self.mol_2conns = copy.deepcopy(self.mol)
            self.midx_2conns = self.remove_2conns_from_mol()
        self.make_graph()
        return
    
    def remove_2conns_from_mol(self):
        """
        Removes all vertices with 2 connecting edges from self.mol
        """
        # make mid-edge atom lists
        delete_list = []
        for i in range(self.mol.natoms):
            if len(self.mol.conn[i]) == 2:
                delete_list.append(i)
        # delete atoms
        self.mol.delete_atoms(delete_list, keep_conn=True)
        ## recompute pconn (since delete_atoms is experimental)
        self.mol.add_pconn()
        self.mol.set_ctab_from_conn(pconn_flag=True)
        self.mol.set_etab_from_tabs()
        return delete_list
    
    def graph2topo(self):
        """
        Returns a molsys.topo object (topology) created from the Graph,
        which can be used to create a graphical representation of the Graph viewable
        in molden, VMD or similar programs.
        """
        return copy.deepcopy(self.mol)
    
    def get_all_cs(self, depth=10, use_atypes=False):
        """
        Calculates all cs (coordination sequence) values of the graph.
        This function just loops over all vertices and calls get_cs for each one
        
        :Parameters:
        - depth: maximum level
        - use_atypes: if this is True, then every vertex with the same atomtype will only be calculated once. 
                      (do NOT use this for topographs deconstructed from a molgraph !!!)
        """
        logger.info('Compute all coordination sequences')
        if use_atypes:
            vertexlist = []
            found_atypes = []
            for i in range(self.mol.natoms):
                if self.mol.atypes[i] not in found_atypes:
                    found_atypes.append(self.mol.atypes[i])
                    vertexlist.append(i)
        else:
            vertexlist = list(range(self.mol.natoms))
        cs_list = []
        for i in vertexlist:
            cs = self.get_cs(depth, i)
            cs_list.append(cs)
        return cs_list

    def get_cs(self, depth, start_vertex=0, start_cell=np.array([0,0,0])):
        """
        Calculates the cs (coordination sequence) values of the vertex specified in start_vertex and start_cell.
        
        :Parameters:
        - depth: maximum level
        - start_vertex: ID of vertex, of which the cs value should be calculated
        - start_cell: starting cell of the function
        """
        def contains(l, obj):
            # the normal construction "if x not in y" does not work if arrays are somehow involved inside a list
            # thus, we need this helper function which is probably terribly slow (if anyone has a better solution, please let me hear it)
            found = False
            for i in l:
                if i[0] == obj[0] and (i[1] == obj[1]).all():
                    found = True
            return found
        # create cs with an appropriate length.
        cs = depth*[0]
        # now, if we want to have a cs_n value, we need to get all neighbours, then do the same thing for all neighbour's
        # neighbours, and continue as many times as necessary.
        # however, we must not start looking at the neighbour's neighbours before we looked at ALL neighbours, and set them
        # into the ignore list!
        ignore_list = [[start_vertex, start_cell]]
        neighbours = [[start_vertex, start_cell]]
        for level in range(depth):
            neighbours2 = []
            #print("--------- level "+str(level)+" -----------")
            #print("neighbours: "+str(neighbours))
            for n in neighbours:
                # get the neighbours of the neighbours and add them to the list neighbours2
                visited = self.get_cs1(n[0], n[1])
                for v in visited:
                    if not contains(neighbours2, v):
                        if not contains(ignore_list, v):
                            neighbours2.append(v)
            #print("ignore_list: " +str(ignore_list))
            # put the neighbours2 into the ignore list
            for n2 in neighbours2:
                ignore_list.append(n2)
            #print("neighbours2: "+str(neighbours2))
            # the neighbours2 are all the vertices which can be reached with the cs_level operation.
            cs[level] = len(neighbours2)
            # if we want to repeat the procedure for the cs_(level+1) operation we have to make the neighbours2 to the neighbours
            neighbours = neighbours2
            # and remove neighbours2 because of side effects
            del(neighbours2)
        return cs

    def get_cs1(self, start_vertex=0, start_cell=np.array([0,0,0])):
        """ This function will return all vertices, which are connected to the vertex start_vertex in the cell start_cell. """
        visited = []
        # loop over all neighbouring clusters and add them to the list
        for nj, j in enumerate(self.mol.conn[start_vertex]):
            current_vertex = j
            current_cell = start_cell + self.mol.pconn[start_vertex][nj]
            visited.append([current_vertex, current_cell])
        return visited

    def get_all_vs(self, use_atypes=False, wells = False, max_supercell_size=20):
        """
        Calculates all vertex symbols of the graph.
        
        :Parameters:
        - use_atypes (bool): if this is True, then every vertex with the same atomtype will only be calculated once.
        - wells (bool): If True, the Wells and the long symbols will be returned. If False, only the long symbols are used
        - max_supercell_size (int): throws OverflowError if supercell_size > max_supercell_size
            (same error as if max_supercell_size is infinite and get_all_vs fails)
        """
        logger.info('Compute all vertex symbols')
        if use_atypes:
            vertexlist = []
            found_atypes = []
            for i in range(self.mol.natoms):
                if self.mol.atypes[i] not in found_atypes:
                    found_atypes.append(self.mol.atypes[i])
                    vertexlist.append(i)
        else:
            vertexlist = list(range(self.mol.natoms))
        vs_list = []
        supercells = [None, copy.deepcopy(self.mol)]
        keep = copy.deepcopy(self.mol)
        for i in vertexlist:
            success = False
            supercell_size = 2
            while not success:
                if supercell_size > max_supercell_size:
                    logger.error("Maximum cell size (%s) reached!" % max_supercell_size)
                    raise OverflowError
                if supercell_size > len(supercells)-1:
                    self.mol = copy.deepcopy(keep)
                    self.mol.make_supercell([supercell_size]*3)
                    supercells.append(copy.deepcopy(self.mol))
                else:
                    self.mol = supercells[supercell_size]
                success = True
                self.make_graph()
                try:
                    ws, ls = self.get_vertex_symbol(i)
                except ValueError:
                    success = False
                    supercell_size += 1
            self.mol = keep
            self.make_graph()
            if wells: 
                vs = (ws, ls)
            else:
                vs = ls
            vs_list.append(vs)
        return vs_list
    
    def get_vertex_symbol(self, start_vertex):
        """
        In the vertex symbol, the number of the shortest ring at each angle of a vertex is given with
        the number of such rings in brackets (originally: subscript, but this can't be realized in
        python).
        Relevant literature: O. Delgado-Friedrichs, M. O'Keeffe, Journal of Solid State Chemistry 178, 2005, p. 2480ff.
        """
        if self.mol.natoms == 1:
            raise ValueError("Topology only consists of only one vertex")
        self.molg.vp.filled.set_value(False)
        self.molg.vp.filled[start_vertex] = True
        vertex_symbol = []
        paths = []
        for source in self.molg.vertex(start_vertex).all_neighbours():
            for target in self.molg.vertex(start_vertex).all_neighbours():
                if source < target:
                    self.molg.set_vertex_filter(self.molg.vp.filled, inverted=True)
                    asp = all_shortest_paths(self.molg, source, target)
                    self.molg.clear_filters()
                    append_list = []
                    for p1 in asp:
                        path = p1.tolist()
                        path = list(map(int, path))
                        p2 = [start_vertex]+path
                        vol = self.get_cycle_voltage(p2)
                        if vol.any() != np.zeros(3).any():
                            raise ValueError("Cycle with non zero voltage detected")
                        path.append(start_vertex)
                        append_list.append(path)
                    if len(append_list) != 0:
                        vertex_symbol.append((len(append_list[0]), len(append_list)))
        ws = self.compute_wells_symbol(vertex_symbol)
        ls =  self.compute_long_symbol(vertex_symbol)
        return ws, ls

    def compute_wells_symbol(self, clist):
        symbol = ""
        if list(clist) == []:
            clist = []
        else:
            clist = np.array(clist)[:,0].tolist()
        sclist = sorted(set(clist))
        for i, s in enumerate(sclist):
            count = clist.count(s)
            if count != 1:
                symbol += "%s^%s." % (s,count)
            else:
                symbol += "%s." % s
        return symbol[:-1]

    def compute_long_symbol(self,clist):
        symbol = ""
        dtype = [("length",int),("number",int)]
        clist = np.array(clist,dtype=dtype)
        sclist = np.sort(clist, order=["length","number"]).tolist()
        for i, s in enumerate(sclist):
            if s[1]==1:
                symbol += "%s." % s[0]
            else:
                symbol += "%s(%s)." % (s[0],s[1]) 
        return symbol[:-1]

    def get_cycle_voltage(self, cycle):
        cycle.append(cycle[0])
        vol = np.zeros(3)
        for i in range(len(cycle)-1):
            cidx = self.mol.conn[cycle[i]].index(cycle[i+1])
            vol += self.mol.pconn[cycle[i]][cidx]
        return vol

    def get_unique_vd(self, cs, vs, atype = True):
        """
        Returns the unique cs values and vertex symbols ("vertex descriptors", vd). 
        :Parameters:
        - cs: list of cs values
        - vs: list of vertex symbols
        - atype: If this is True, the function also changes the atomtypes in the 
          topograph (i.e. the "vertex types"), according to the different vertex 
          descriptors.
        """
        logger.info('Get unique vertex descriptors')
        assert type(cs) == list
        assert type(vs) == list
        assert len(vs) == len(cs)
        atypes = []
        uvd = []
        atcount = 0
        for c,v in zip(cs,vs):
            vd = tuple([tuple(c),v])
            if vd not in uvd: 
                uvd.append(vd)
                atypes.append(str(atcount))
                atcount += 1
            else:
                atypes.append(str(uvd.index(vd)))
        ucs = []
        uvs = []
        for i in uvd:
            ucs.append(i[0])
            uvs.append(i[1])
        if atype: self.mol.set_atypes(atypes)
        return ucs, uvs

    def build_coordination_pattern(self,pattern, cn):
        assert type(pattern) == list
        assert len(pattern) == 2
        assert type(cn) == list
        assert len(cn) == pattern[0]-1 + pattern[1]-1
        assert cn.count(pattern[0]) == cn.count(pattern[1]) == 0
        ### build subgraph
        patg = Graph(directed=False)
        patg.vp.cn = patg.new_vertex_property("int64_t")
        patg.vp.type = patg.new_vertex_property("string")
        for i in pattern:
            v = patg.add_vertex()
            patg.vp.cn[v] = i
            patg.vp.type[v] = "c"
        patg.add_edge(patg.vertex(0),patg.vertex(1))
        for i, c in enumerate(pattern):
            for j in range(c-1):
                v = patg.add_vertex()
                patg.vp.cn[v] = cn[i+j]
                patg.vp.type[v] = "p"
                patg.add_edge(v, patg.vertex(i))
        return patg

    def build_coordination_pattern_from_mol(self,mol):
        atypes = mol.get_atypes()
        types = []
        cns = []
        for a in atypes:
            sa = a.split("_")
            t = sa[0]
            assert ["c","r","p"].count(t) > 0
            cn = int(sa[1])
            types.append(t)
            cns.append(cn)
        mol.addon("graph")
        mol.graph.make_graph()
        mol.graph.molg.vp.cn = mol.graph.molg.new_vertex_property("int64_t")
        for v in mol.graph.molg.vertices():
            mol.graph.molg.vp.cn[v] = cns[int(v)]
            mol.graph.molg.vp.type[v] = types[int(v)]
        return mol.graph.molg

    def search_coordination_pattern(self,patg):
        assert type(patg) == Graph
        self.molg.vp.cn = self.molg.new_vertex_property("int64_t")
        for v in self.molg.vertices():
            self.molg.vp.cn[v] = len(list(v.out_neighbours()))
        maps = subgraph_isomorphism(patg, self.molg, vertex_label =
                (patg.vp.cn, self.molg.vp.cn))
        subs = []
        smaps = []
        for m in maps:
            sl = list(m)
            sl.sort()
            if sl not in subs: 
                subs.append(sl)
                smaps.append(m)
        return smaps

    def collapse_subs(self, maps, patg):
        assert type(patg) == Graph
        assert type(maps) == list
        dl = []
        ### search centers,removals and planets
        for m in maps:
            center = []
            planets = []
            #v = self.molg.add_vertex()
            for v in patg.vertices(): 
                if patg.vp.type[v] == "c":
                    center.append(int(m[v]))
                elif patg.vp.type[v] == "p":
                    planets.append(int(m[v]))
                else:
                    dl.append(int(m[v]))
            if len(center)==1:
                v = self.molg.vertex(center[0])
                cidx = center[0]
            elif len(center) > 1:
                v = self.molg.add_vertex()
                dl += center
                self.mol.set_unit_mass()
                xyz = self.mol.get_com(center)
                self.molg.vp.coord[v] = xyz
                self.mol.add_atom('c','n',xyz)
                cidx = self.mol.natoms-1
            else:
                logger.error("No Center Vertex found")
                return
            for vidx in planets:
                vi = self.molg.vertex(vidx)
                self.molg.add_edge(vi, v)
                self.mol.add_bond(cidx,vidx)
                #self.mol.conn[cidx].append(vidx)
                #self.mol.conn[vidx].append(cidx)
        for v in reversed(sorted(dl)):
            self.molg.remove_vertex(v)
            self.mol.delete_atom(v)
        self.mol.add_pconn()
        self.mol.set_ctab_from_conn(pconn_flag=True)
        self.mol.set_etab_from_tabs()
        return

class topotyper(object):
    """ Wrapper class which combines molgraph and topograph for the deconstruction of MOF structures. """
  
    def __init__(self, mol, split_by_org=True, depth=10, max_supercell_size=20, isum=3, trip=None):
        """
        :Parameters:
        - mol (obj): molsys.mol object which should be deconstructed
        - split_by_org (bool): if True, organicity is used for defining building blocks
        - depth (int): maximum level of coordination sequences 
        - isum (int*): summation of indices
        - trip (nested tuples*): resizing list for make_supercell if vertex symbols method overflows
            * recursive purpose, DO NOT CHANGE
        """
        self.api = None
        #molcopy = copy.deepcopy(mol) #prevents mol pollution if restart ###DOES NOT WORK WITH CIF FILES
        molcopy = copy.copy(mol) #prevents mol pollution if restart
        goodinit = False
        self.mg = molgraph(molcopy)
        while not goodinit:
            if trip is None:
                trinat = triplenats_on_sphere(isum)
            else:
                trinat = trip
            if len(trinat) > 0:
                itri = trinat.pop()
                logger.info("Triplet is: "+str(itri))
                if isum > 3: mol.make_supercell(itri)
                try:
                    self.deconstruct(split_by_org, depth=depth, max_supercell_size=max_supercell_size)
                    goodinit = True
                except OverflowError as e: # specific for supercells
                    logger.info("Deconstruction failed! Resize original cell")
                    self.__init__(mol,split_by_org,depth=depth, max_supercell_size=max_supercell_size, isum=isum,trip=trinat)
            else:
                isum += 1
                logger.info("Resizing list is empty! Increase index summation to %i and create new resizing list" % (isum,))
                self.__init__(mol,split_by_org,depth=depth, max_supercell_size=max_supercell_size, isum=isum)
        return
 
    def deconstruct(self, split_by_org=True, depth=10, max_supercell_size=5):
        """ Perform deconstruction """
        logger.info('Perform topological deconstruction')
        import time; t1 = time.time()
        self.mg.handle_islands(silent=True)
        print('handle_islands %5.1f'  % (time.time()-t1),); t1 = time.time()
        self.mg.determine_Nk()
        print('determine_Nk %5.1f' % (time.time()-t1),); t1 = time.time()
        self.mg.find_cluster_threshold()
        print('find_cluster_threshold %5.1f' % (time.time()-t1),); t1 = time.time()
        self.mg.get_clusters()
        print('get_clusters %5.1f' % (time.time()-t1),); t1 = time.time()
        if split_by_org:
            self.mg.get_bbs_by_organicity()
        else:
            self.mg.get_bbs()
        print('get_bbs %5.1f' % (time.time()-t1),); t1 = time.time()
        self.tg = self.mg.make_topograph(False)
        print('make_topograph %5.1f' % (time.time()-t1),); t1 = time.time()
        #cs = self.tg.get_all_cs(depth=depth)
        #print('get_all_cs %5.1f' % (time.time()-t1),); t1 = time.time()
        #vs = self.tg.get_all_vs(max_supercell_size=max_supercell_size)
        #print('get_all_vs %5.1f' % (time.time()-t1),); t1 = time.time()
        #self.cs, self.vs = self.tg.get_unique_vd(cs, vs)
        return

    def compute_colors(self, sort_flag=True):
        """
        Compute edge coloring (TBI: vertex coloring)
        
        sort_flag (bool): if it is True: colors are sorted according to
            increasing color signature

        >>> import molsys
        >>> from molsys.util import toper
        >>> m = molsys.mol.from_file("jast-1") # as example
        >>> tt = toper.topotyper(m) # may need time
        >>> tt.compute_colors()
        >>> from molsys.util.color import make_mol # TBI: colors I/O
        >>> ecolors = tt.tg.molg.ep.color.a
        >>> n = make_mol(tt.tg.mol, alpha=3, ecolors=ecolors)
        """
        try:
            assert self.bbs
        except:
            self.compute_bbs()
        logger.info('Perform topological coloring')
        ### TBI: vertex coloring
        ### vertices = range(self.tg.mol.natoms) # the clusters
        ### vertexcolors = np.zeros(len(vertices), dtype=int)
        """
        ### PREVIOUS IMPLEMENTATION ###
        #edges = self.tg.mol.ctab
        #edgecolors = np.zeros(len(edges), dtype=int)
        #...
        #for iedge, edge in enumerate(self.tg.molg.edges()):
        #    ...
        #    edgecolors[iedge] = colorsigns[sign]
        ###############################
        """
        list2c = self.tg.midx_2conns
        vertex_bb_list = list(range(len(self.mg.clusters)))
        for i in reversed(sorted(list2c)):
            del vertex_bb_list[vertex_bb_list.index(i)]
        colorsigns = {} # color signatures
        clustsigns = {}
        assert len(self.tg.mol.etab) == len(set(self.tg.mol.etab)), \
            "edges w/ same termini (i.e. when pconn is needed) is NOT supported [TBI]"
        self.tg.molg.ep.color = self.tg.molg.new_edge_property("int")
        for edge in self.tg.molg.edges():
            # v,u are source,target vertex of the edge
            v = vertex_bb_list[int(edge.source())]
            u = vertex_bb_list[int(edge.target())]
            # unique block names
            ubbv = self.bb2ubb[v]
            ubbu = self.bb2ubb[u]
            # signature
            sign = self.get_color_signature(v, u)
            if len(sum(sign,())) == 0:
                # colors are edge midpoints
                # workaround: since edges are no more stored in tg we use
                # directly mg. This will be SLOW
                # TBI: use tg (maybe w/ self.tg.mol_2conns.conn)
                # cluster atoms
                clustv = self.mg.clusters[v]
                clustu = self.mg.clusters[u]
                ## cluster connectivity
                connv = [[self.abb[j] for j in self.mg.mol.conn[i]] for i in clustv]
                connu = [[self.abb[j] for j in self.mg.mol.conn[i]] for i in clustu]
                allconnv = sum(connv,[])
                allconnu = sum(connu,[])
                allconnv_ = [k for k in allconnv if k != v]
                allconnu_ = [k for k in allconnu if k != u]
                ## midpoint cluster
                allconnt = list(set(allconnv_) & set(allconnu_))
                assert len(allconnt) == 1,\
                    "only 1 edge must be btw. 2 vertices"
                t = allconnt[0] # the only one
                ubbt = self.bb2ubb[t]
                sign1 = self.get_color_signature(v, t)
                sign2 = self.get_color_signature(u, t)
                sign = (ubbt,tuple(sorted([sign1,sign2])))
                #if sign1 == sign2:
                #    sign = ubbt ### JUST the cluster type!
                #else:
                #    raise NotImplementedError("TBI: edge midpoints w/ orient colors")
            # order (int) of occurrence assigned as color to the edge
            if sign not in colorsigns:
                colorsigns[sign] = len(colorsigns) # starting from 0
                #clustsigns[sign] = (ubbv, ubbu)
            self.tg.molg.ep.color[edge] = colorsigns[sign]
        if sort_flag is True:
            # sort colors according to color signature lexsorting
            # it is invariant wrt. order of occurrence
            # it sorts just the colortypes
            keys, items = list(zip(*list(colorsigns.items())))
            sorting_dict = dict(list(zip(items,argsorted(keys))))
            self.tg.molg.ep.color.a = [sorting_dict[i] for i in self.tg.molg.ep.color]
        return

    def write_colors(self, foldername="colors", index_run=False, scell=None, sort_flag=True):
        if not "color" in self.tg.molg.ep:
            self.compute_colors(sort_flag=sort_flag)
        if index_run:
            foldername = _checkrundir(foldername)
        else:
            _makedirs(foldername)

        asort = argsorted([(int(e.source()),int(e.target())) for e in self.tg.molg.edges()])
        self.tg.molg.ep.color.a = self.tg.molg.ep.color.a[asort]
        n = make_mol(self.tg.mol, alpha=3, ecolors=self.tg.molg.ep.color.a)
        if scell is not None:
            n.make_supercell(scell)
        n.write("%s%s%s.mfpx" % (foldername, os.sep, "colors"))
        n.write("%s%s%s.txyz" % (foldername, os.sep, "colors"), pbc=False)

        return foldername

    def get_color_signature(self, v, u):
        """
        Get color signature taking the connecting atoms at the boundary of the blocks
        TBI: atom sequence going "inside" the block instead of just connecting atoms

        v(int): vertex with index v
        u(int): vertex with index u
        """
        # cluster atoms
        clustv = self.mg.clusters[v]
        clustu = self.mg.clusters[u]
        # cluster connectivity
        connv = [
            [jc for jc in self.mg.mol.conn[ic] if jc in clustu]
            for ic in clustv
        ]
        connu = [
            [jc for jc in self.mg.mol.conn[ic] if jc in clustv]
            for ic in clustu
        ]
        # cluster connectivity towards each other
        # TBI: atom sequence in that direction
        connv_ = [i for i,ic in enumerate(connv) if ic != []]
        connu_ = [i for i,ic in enumerate(connu) if ic != []]
        # connecting atoms == cluster atoms at the frontier of each other
        clustv_ = [clustv[i] for i in connv_]
        clustu_ = [clustu[i] for i in connu_]
        # elements of the connecting atoms
        # N.B.: sorted for consistency, tupled for hashability as dict key
        # TBI: choice of elemsequences
        elemsv = tuple(sorted([self.mg.mol.elems[i] for i in clustv_]))
        elemsu = tuple(sorted([self.mg.mol.elems[i] for i in clustu_]))
        sign = tuple(sorted([elemsv, elemsu]))
        return sign

    def get_net(self):
        """ Connects to mofplus API to compare the cs and vs value to the database, and return the topology. """
        self.api = user_api()#experimental=False)
        self.nets = self.api.search_cs(self.cs, self.vs)
        return self.nets

    def tg_atypes_by_isomorphism(self):
        """
        This method will modify the atomtypes in the topograph (i.e. "vertex types"): If
        two vertices with the same atomtype have different structures, the atomtype will be
        changed.
        Possible todo: compare the elements with each other
        """
        cv, ca = self.mg.get_cluster_atoms()
        bb_molgraphs = []
        try:
            assert self.mg.cluster_organicity
        except:
            self.mg.detect_organicity()
        # Remove all 2-connected clusters, since they aren't in the topograph
        list2c = self.tg.midx_2conns
        for i in reversed(sorted(list2c)):
            del ca[i]
        # check for isomorphism
        for i, atoms in enumerate(ca):
            m = self.mg.mol.new_mol_by_index(atoms)
            mg = molgraph(m)
            #print("i "+str(i))
            j = len(bb_molgraphs) - 1
            while j >= 0:
                mg2 = bb_molgraphs[j]
                #print("  j "+str(j))
                if mg.mol.natoms > 1 and mg2.mol.natoms > 1:
                    if self.tg.mol.atypes[j] == self.tg.mol.atypes[i]:
                        if not isomorphism(mg.molg, mg2.molg):
                            self.tg.mol.atypes[i] += "#"
                            j = len(bb_molgraphs)
                j -= 1
            bb_molgraphs.append(mg)
        for i in range(len(self.tg.mol.atypes)):
            n = self.tg.mol.atypes[i].count("#")
            if n > 0:
                self.tg.mol.atypes[i] = self.tg.mol.atypes[i].replace("#", "")
                self.tg.mol.atypes[i] += ascii_lowercase[n-1]
        return
    
    def get_all_atomseq(self, depth=10, clusters = None):
        """
        Calculates all atomseq (atom sequences) of the molecule according to their elements..
        This function just loops over all vertices and calls get_atomseq for each one
        
        :Parameters:
        - depth: maximum level
        """
        logger.info('Compute all atom sequences')
        if clusters is None:
            clusters = list(range(self.nbbs))
        atomseqs = []
        for i in clusters:
            atomseq = self.get_atomseq(depth, i)
            atomseqs.append(atomseq)
        self.atomseqs = atomseqs
        return atomseqs

    def get_atomseq(self, depth, start_cluster=0, start_cell=(0,0,0)):
        ###TBI: implement start_cell!
        ###start_cell IS NEEDED for images
        """
        Calculates the as (atom sequence) of the vertex specified in start_vertex and start_cell.
        
        :Parameters:
        - depth: maximum level
        - start_vertex: ID of vertex, of which the cs value should be calculated
        - start_cell: starting cell of the function
        """
        def connected(ia, ignore, return_set=True):
            conn = set(self.mg.mol.conn[ia])
            ignore = set(ignore)
            conn -= ignore
            if return_set:
                return conn
            else:
                return int(*conn)
        atomconn = self.bb2conn[start_cluster]
        nconn = len(atomconn)
        ignore = self.mg.clusters[start_cluster]
        ignore = set(ignore)
        atomseq = [[] for i in range(nconn)]
        for ic,iconn in enumerate(atomconn):
            lever = [iconn] #for depth == 0
            for level in range(depth):
                levseq = []
                leverneigh = set([])
                for cc in lever:
                        neigh = connected(cc,ignore)
                        leverneigh |= neigh
                if neigh:
                    leverneigh -= ignore ###????
                    ignore |= leverneigh
                    levseq.append(tuple(neigh))
                    lever = leverneigh
                else:
                    break
                atomseq[ic] += levseq
            atomseq[ic] = tuple(atomseq[ic]) ###
        return set(atomseq)

    def atomseq2elemseq(self):
        elems = self.mg.mol.elems
        atomseq = self.atomseq
        #elemseq = [[] for i in range(len(atomseq))]
        for i in atomseq:
            for j in i:
                print(j)
                for k in j:
                    print(elems[k], end=" ")
                print()
        #for i,ai in enumerate(atomseq):
        #    print(i)
        #    for j in ai:
        #        for k in j:
        #            print(elems[k], end=" ")
        #        print()
        #    #print(i, ai)
        return

    def compute_bbs(self, org_flag="_ORG", ino_flag="_INO"):
        """Compute building blocks

        >>> import molsys
        >>> from molsys.util import toper
        >>> m = molsys.mol.from_file("jast-1") # as example
        >>> tt = toper.topotyper(m) # may need time
        >>> tt.compute_bbs()
        >>> tt.write_bbs()
        """
        logger.info('Perform topological deconstruction in blocks')
        self.tg_atypes_by_isomorphism()
        cv, ca = self.mg.get_cluster_atoms()
        bbs = []
        organicity = []
        for i, atoms in enumerate(ca):
            m = self.mg.mol.new_mol_by_index(atoms)
            bbs.append(m)
            if self.mg.cluster_organicity[i] == True:
                organicity.append(org_flag)
            else:
                organicity.append(ino_flag)
        # prepare vertex_bb_list (to translate indices of a list with 2-connected clusters to those of one without them)
        list2c = self.tg.midx_2conns
        vertex_bb_list = list(range(len(self.mg.clusters)))
        for i in reversed(sorted(list2c)):
            del vertex_bb_list[vertex_bb_list.index(i)]
        # Use the atomtypes to identify "vertex" BBs
        atomtype_dict = {}
        for i, atype in enumerate(self.tg.mol.atypes):
            try:
                atomtype_dict[atype]
            except KeyError:
                atomtype_dict[atype] = []
            atomtype_dict[atype].append(vertex_bb_list[i])
        unique_bbs = []
        cluster_names = []
        for key in atomtype_dict.keys():
            unique_bbs.append(atomtype_dict[key])
            cluster_names.append(key)
        # determine which of the clusters are "edge" BBs and
        # Check to which vertex BBs the edge BBs are connected
        cluster_conn = self.mg.cluster_conn()
        neighbour_atypes = []
        unique_2c = []
        c2ubb = {}
        for i in list2c:
            neighbour_atype = []
            for cc in cluster_conn[i]:
                for key in atomtype_dict.keys():
                    if cc in atomtype_dict[key]:
                        neighbour_atype.append(key)
                        break
            neighbour_atype = list(sorted(neighbour_atype))
            if neighbour_atype not in neighbour_atypes:
                neighbour_atypes.append(neighbour_atype)
                unique_2c.append(i)
                unique_bbs.append([i])
                c2ubb[i] = len(unique_bbs)-1
                cluster_names.append(neighbour_atype[0]+"-"+neighbour_atype[1])
            else:
                # check for isomorphism
                mg = molgraph(bbs[i])
                isomorphic_with_any = False
                for i2 in unique_2c:
                    mg2 = molgraph(bbs[i2])
                    if mg.mol.natoms > 1 and mg2.mol.natoms > 1:
                        if isomorphism(mg.molg, mg2.molg):
                            isomorphic_with_any = True
                            unique_bbs[c2ubb[i2]].append(i)
                            c2ubb[i] = c2ubb[i2]
                            break
                    else:
                        # this is a workaround, so multiple edge BBs with 1 atom will always be isomorphic...
                        isomorphic_with_any = True
                if not isomorphic_with_any:
                    unique_2c.append(i)
                    unique_bbs.append([i])
                    c2ubb[i] = len(unique_bbs)-1
                    name = neighbour_atype[0]+"-"+neighbour_atype[1]
                    j = 0
                    while name in cluster_names:
                        name = neighbour_atype[0]+"-"+neighbour_atype[1]+ascii_lowercase[j]
                        j += 1
                    cluster_names.append(name)
        self.bbs = bbs
        self.nbbs = len(bbs)
        self.unique_bbs = unique_bbs
        self.cluster_names = cluster_names
        self.organicity = organicity
        self.set_atom2bb()
        self.set_bb2ubb()
        self.detect_connectors()
        ### BUG HERE ###
        #self.detect_all_connectors()
        #self.set_conn2bb()
        return

    def detect_connectors(self):
        for ubbs in self.unique_bbs:
            ubb = ubbs[0] # assumption
            bb = self.bbs[ubb]
            bbatoms_mg = self.mg.clusters[ubb]
            if bb.elems != [self.mg.mol.elems[i] for i in bbatoms_mg]: # this is the assumption
                bb = self.mg.mol.new_mol_by_index(bbatoms_mg)
                # next line is needed since atoms are sorted according to
                #   previous order, not according the given bbatoms_mg order
                bbatoms_mg.sort()
                ### DEBUG ### try to remove the previous line and see what happens
                #for kk,k in enumerate(bbatoms_mg):
                #    print(
                #    "%s == %s? %s" % (
                #        bb.elems[kk], self.mg.mol.elems[k], bb.elems[kk] == self.mg.mol.elems[k]
                #        )
                #)
                ### END DEBUG ###
                self.bbs[ubb] = bb # NOT SURE
            connectors = [] # to be converted later
            connector_atoms = []
            connectors_sign = []
            connectors_type = []
            for ii,(i,j) in enumerate(self.mg.mol.ctab):
                iin = i in bbatoms_mg
                jin = j in bbatoms_mg
                if not iin ^ jin: #both or none
                    continue
                if jin: # reverse
                    i,j = j,i
                ie, je = self.mg.mol.elems[i], self.mg.mol.elems[j]
                iabb, jabb = self.abb[i], self.abb[j]
                iubb, jubb = self.bb2ubb[iabb], self.bb2ubb[jabb]
                connectors.append(bbatoms_mg.index(i))
                connector_atoms.append([bbatoms_mg.index(i)])
                connectors_sign.append((ie,je,iubb,jubb))
            connectors_signtype = list(Counter(connectors_sign).keys())
            sign2type = dict([(e,i) for i,e in enumerate(connectors_signtype)])
            connectors_type = [sign2type[e] for e in connectors_sign]
            # center in box
            bb.set_cell(self.mg.mol.cell)
            bb.periodic = True
            bb.center_com(check_periodic=False)
            bb.wrap_in_box()
            bb.bcond = 0
            bb.addon("bb")
            bb.is_bb = True
            bb.connectors = connectors
            bb.connector_atoms = connector_atoms
            bb.connectors_type = connectors_type
            bb.center_point = 'coc'
    
    def write_bbs(self, foldername="bbs", index_run=False, org_flag="_ORG", ino_flag="_INO"):
        """
        Write the clusters of the molgraph into the folder specified in the parameters.
        The names of the clusters written out will be those of the atomtypes of the vertices
        in the topograph, if they are "vertex" building blocks (i.e. they have more than
        2 neighbours). If they are "edge" building blocks (they have exactly 2 neighbours),
        their name will be the atomtypes of the vertex BBs they are connected to.
        Additionally, at the end of the filename, a string defined by the parameters org_flag
        and ino_flag will be appended to denote whether the BB is organic.
        
        :Parameters:
        - foldername: Name of the folder, in which the mfpx files of the building blocks
          should be saved.
        - org_flag: String, which will be added to the filename if the BB is organic.
        - ino_flag: String, which will be added to the filename if the BB is inorganic.
        """
        # Now write building blocks
        #print("============")
        #print(unique_bbs)
        #print(atomtype_dict)
        #print(organicity)
        #try:
        #    print(neighbour_atypes)
        #except:
        #    print("no 2-conns")
        #print("============")
        if not hasattr(self, "bbs"):
            self.compute_bbs(org_flag=org_flag, ino_flag=ino_flag)
        if index_run:
            foldername = _checkrundir(foldername)
        else:
            _makedirs(foldername)
        for n, i in enumerate(self.unique_bbs):
            m = self.bbs[i[0]]
            m = m.unwrap_box()
            m.write(foldername+"/"+self.cluster_names[n]+self.organicity[i[0]]+".mfpx", "mfpx")
        return foldername
        
    ### BUG HERE, CONNECTIVITY IS CHANGED ###
    #def detect_all_connectors(self):
    #    ###TBI: RETRIEVE EDGES FROM SELF.TG.MOL.CONN
    #    self.edges = set(())
    #    self.bb2conn = []
    #    self.bb2adj = []
    #    self.bb2adjconn = []
    #    for ibb in range(self.nbbs):
    #        iedges, iledges, idedges, icedges = self.detect_connectors(ibb, return_dict=True)
    #        self.edges |= set(iedges)
    #        self.bb2conn.append(iledges)
    #        self.bb2adj.append(idedges)
    #        self.bb2adjconn.append(icedges)
    #    self.edges = list(self.edges)
    #    self.edges.sort()
    #    self.tg.mol.set_ctab(self.edges, conn_flag=True)
    #    self.mol.set_etab_from_tabs()
    #    self.nedges = len(self.edges)
    #    return

    #def detect_connectors(self, ibb, return_dict=False):
    #    batoms = set(self.mg.clusters[ibb])
    #    dedges = {}
    #    cedges = {}
    #    ledges = []
    #    edges = []
    #    for ia in batoms:
    #        ic = set(self.mg.mol.conn[ia])
    #        ic -= batoms #set difference
    #        if ic:
    #            ic = int(*ic) #safety: if more than 1, raise error
    #            dedges[ia] = ic #computed anyway
    #            cedges[self.abb[ic]] = ia
    #            ledges.append(ia)
    #            iedges = [ibb, self.abb[ic]]
    #            iedges.sort()
    #            edges.append(tuple(iedges)) #needs tuple
    #    if return_dict:
    #        return edges, ledges, dedges, cedges
    #    return edges, ledges

    def set_atom2bb(self):
        """from atom index to building block the atom belongs to"""
        self.abb = [None]*self.mg.mol.natoms
        for ibb, bb in enumerate(self.mg.clusters):
            for ia in bb:
                self.abb[ia] = ibb
        return

    ### DOES NOT WORK W/O detect_all_connector
    #def set_conn2bb(self):
    #    """from connector index to building block the atom connects"""
    #    self.conn2bb = [None]*self.mg.mol.natoms
    #    for bba in self.bb2adj:
    #        for c,ca in bba.items():
    #            self.conn2bb[c] = self.abb[ca]
    #    return

    def determine_all_color(self, vtype=0, depth=10):
        """determines colors according to element sequences
        from connectors to blocks"""
        edges = self.edges
        edcol = []
        elemseqs = []
        for e in edges:
            elemseq = self.determine_elemseq(e, vtype, depth=depth)
            elemseqs.append(elemseq)
        unique_elemseqs = []
        for es in elemseqs:
            if es not in unique_elemseqs:
                unique_elemseqs.append(es)
            edcol.append(unique_elemseqs.index(es)) ###SLOW
        self.edcol = edcol
        self.elemseqs = elemseqs
        self.unique_elemseqs = unique_elemseqs
        return

    def set_bb2ubb(self):
        """from building block index to unique building block"""
        self.bb2ubb = [None]*self.nbbs
        for iu,ubb in enumerate(self.unique_bbs):
            for jbb in ubb:
                self.bb2ubb[jbb] = iu
        return
    
    def determine_elemseq(self, edge, vtype=0, depth=10):
        """N.B.: works only with 2 (0/1) colors
        TBI: any number of colors"""
        elems = self.mg.mol.elems
        if self.bb2ubb[edge[0]] == vtype:
            ba,bb = edge
        else: ###AKA### elif self.bb2ubb[edge[1]] == vtype:
            bb,ba = edge
        for atomseq in self.atomseqs[ba]:
            atomseq = list(atomseq)
            if self.abb[atomseq[0][0]] == bb:
                break
        elemseq = []
        for level in range(depth):
            laseq = atomseq[level]
            leseq = []
            for a in laseq:
                leseq.append(elems[a])
            leseq.sort()
            elemseq.append(leseq)
        return elemseq

    def get_vbb(self):
        return self.tg.mol.xyz
    
    def get_vconn(self):
        return self.tg.mol.conn
