# -*- coding: utf-8 -*-

"""

       module to implement an addon feature: graphs using the graph_tool library

       NOTE: this is only imported by __init__.py if graph_tool is present

"""

import graph_tool
from graph_tool import Graph, GraphView
from graph_tool.topology import *
import copy
import numpy as np
import molsys
from molsys.util import elems
from molsys.util.timer import timer, Timer

from tqdm import tqdm

import uuid

import logging
logger = logging.getLogger("molsys.graph")

class graph(object):

    def __init__(self, mol):
        """
        instantiate a graph object which will be attached to the parent mol

        :Parameter:

             - mol : a mol type object
        """
        self._mol = mol
        self.molg = Graph(directed=False)
        # self.molg.vp.type = self.molg.new_vertex_property("string")
        self.molg.vp.molid = self.molg.new_vertex_property("int")
        # defaults
        self.moldg = None
        self.bbg   = None
        logger.debug("generated the graph addon")
        self.timer = Timer("graph addon")
        return

    def make_graph(self, idx = None, rule = 1, hashes = True, omit_x=False):
        """generate a graph for the mol object (atoms should be typed)

        RUB RS 2024: revision of ff assignement .. improve speed of molg generation
                     use ctab to make graph in one shot
                     do not allow idx != None 

        Args:
            idx (list of integer, optional): list of atoms to be converted, None means all. Defaults to None.
            rule (int, optional): which rule to use for the vtypes. Defaults to 1.
            hashes (bool, optional): convert onecoord vertices to a hash in vtype. Defaults to True.
            omit_x (bool, optional): omit all atoms with element X. Defaults to False.

        generate a graph for the mol object (atoms should be typed)
        rule = 0: The vertex property is the element
        rule = 1: The vertex property is the element plus the coordination number
        rule = 2: The vertex property is the full atomtype
        """
        # current revisison: i think we do not need specific indices ... so lets prevent this
        # if idx is None: idx = range(self._mol.natoms)
        assert idx == None, "Making molgraphs with a subset of atoms is not possible .. if this is needed please reimplement"
        self.molg.clear() # allways start from a fresh graph
        # now generate the entire graph from the ctab
        self.molg.add_vertex(n=self._mol.natoms)
        self.molg.add_edge_list(self._mol.ctab)
        # add the atomid as a vertex property
        self.molg.vp.atomid = self.molg.new_vertex_property("int")
        self.molg.vp.atomid.a = np.arange(self._mol.natoms)
        if rule == 0:
            self.molg.vp.type = self.molg.new_vertex_property("string", vals=self._mol.elems)
        elif rule == 2:
            self.molg.vp.type = self.molg.new_vertex_property("string", vals=self._mol.atypes)
        elif rule == 1:
            vtypes = []
            for i in range(self._mol.natoms):
                vtype = self._mol.elems[i]
                if "_" in self._mol.atypes[i]:
                    vtype = self._mol.atypes[i].split("_")[0]
                if hashes:
                    if vtype[-1] == "1":
                        vtype = "#"
                vtypes.append(vtype)
            self.molg.vp.type = self.molg.new_vertex_property("string", vals=vtypes)
        if omit_x:
            delv = [i for i in range(self._mol.natoms) if self._mol.elems[i].lower() == "x"]
            delv.reverse()
            self.molg.remove_vertex(delv)
        logger.info("generated a graph for a mol object with %d vertices" % self.nvertices)
        return

    def write_graph(self, fname=None):
        """
        write the graph to a file

        Args:
            fname (string): filename
        """
        if fname == None:
            fname = self._mol.name + ".gt"
        self.molg.save(fname)
        return

    @property
    def nvertices(self):
        return self.molg.num_vertices()

    def get_iso(self, other):
        """
        get the isomorphism between two graphs

        Args:
            other (graph object): other graph object

        Returns:
            list of lists: list of isomorphisms
        """
        assert hasattr(other, "molg"), "other is not a graph addon object"
        assert type(other.molg) == Graph, "other.molg is not a graph object"
        maps = subgraph_isomorphism(self.molg, other.molg, vertex_label=(self.molg.vp.type, other.molg.vp.type))
        return maps

    def make_comp_graph(self, elem_list = ["c"], idx = None):
        """
        Like make_graph this creates a graph for the mol object, but here we focus on a graph 
        for comparing molecular species in reactions. In the current graph we only look at the C-graph

        """
        if idx is None: idx = range(self._mol.natoms)
        # now add vertices
        self.vert2atom = []  
        ig = 0
        self.molg.clear()  # allways start from a fresh graph
        for i in idx:
            if self._mol.elems[i] in elem_list:
                self.molg.add_vertex()
                self.vert2atom.append(i)
                vtype = self._mol.atypes[i]
                # extract element and coordination number
                if "_" in vtype:
                    vtype = vtype.split("_")[0]
                self.molg.vp.type[ig] = vtype
                ig += 1
        self.nvertices = len(self.vert2atom)
        logger.info("generated a graph for a mol object with %d vertices" % self.nvertices)
        # now add edges ... only bonds between vertices
        for i in range(self.nvertices):
            ia = self.vert2atom[i]
            for ja in self._mol.conn[ia]:
                if ja>=ia:   #we need a .le. here for those atoms/vertices connected to itself twice in different boxes
                    if ja in self.vert2atom:
                        self.molg.add_edge(self.molg.vertex(i), self.molg.vertex(self.vert2atom.index(ja)))
        return

    def plot_graph(self, fname, g = None, label=None, edge_label=None, size=1000, fsize=16, vsize=8, ptype = "pdf",method='arf'):
        """
        plot the graph (needs more tuning options) [via **kwargs? RA]

        :Parameter:
            - fname  : filename (will write filename.pdf)
            - size   : outputsize will be (size, size) in px [default 800]
            - fsize  : font size [default 10]
            - method : placement method to draw graph, can be one of
                       arf
                       frucht
                       radtree
                       sfdp
                       random
        """
        #GS TODO call plot_molgraph to do actual plotting

        import graph_tool.draw as gtd
        if g:
            draw_g = g
        else:
            draw_g = self.molg
        if label:
            vlabel = label
        else:
            vlabel = "type"
        g=draw_g
        if method=='arf':
            pos = gtd.arf_layout(draw_g, max_iter=0)
        elif method=='frucht':
            pos = gtd.fruchterman_reingold_layout(draw_g, n_iter=1000)
        elif method=='radtree':
            pos = gtd.radial_tree_layout(draw_g, draw_g.vertex(0))
        elif method=='sfdp':
            pos = gtd.sfdp_layout(draw_g)
        elif method=='random':
            pos = gtd.random_layout(draw_g)
        else:
            pos=None
        gtd.graph_draw(draw_g,pos=pos, vertex_text=draw_g.vp[vlabel], vertex_font_size=fsize, vertex_size=vsize, \
            output_size=(size, size), output=fname+"."+ptype, bg_color=[1,1,1,1])
        return

    @staticmethod
    def plot_mol_graph(fname, g, label=None, edge_label=None, size=500, fsize=10, vsize=8, ptype = "pdf",method='sfdp'):
        """
        plot the graph (needs more tuning options) [via **kwargs? RA]

        :Parameter:
            - fname  : filename (will write filename.pdf)
            - size   : outputsize will be (size, size) in px [default 800]
            - fsize  : font size [default 10]
            - method : placement method to draw graph, can be one of
                       arf
                       frucht
                       radtree
                       sfdp
                       random
        """
        import graph_tool.draw as gtd
        if label:
            vlabel = label
        else:
            vlabel = "type"

        draw_g = g

        if method=='arf':
            pos = gtd.arf_layout(draw_g, max_iter=0)
        elif method=='frucht':
            pos = gtd.fruchterman_reingold_layout(draw_g, n_iter=1000)
        elif method=='radtree':
            pos = gtd.radial_tree_layout(draw_g, draw_g.vertex(0))
        elif method=='sfdp':
            pos = gtd.sfdp_layout(draw_g)
        elif method=='random':
            pos = gtd.random_layout(draw_g)
        else:
            pos=None
        gtd.graph_draw(draw_g,pos=pos, vertex_text=draw_g.vp[vlabel], vertex_font_size=fsize, vertex_size=vsize, \
            output_size=(size, size), output=fname+"."+ptype, bg_color=[1,1,1,1])
        return

    @staticmethod
    def find_subgraph(graph, subg, graph_property = None, subg_property = None, N=0, verbose=False, keep_order=False):
        """
        use graph_tools subgraph_isomorphism tool to find substructures

        :Parameter:

            - graph : parent graph to be searched
            - subg  : graph to be found
            - N (int): number of subgraphs to find, N=0 is all (defaults to N=0)
            - verbose (bool): print out progress
            - keep_order (bool): keep the order of the vertices in the returned maps (default False)

        :Returns:

            a list of lists with the (sorted) vertex indices of the substructure
        """
        temp_timer = Timer("graph addon: find_subgraph")
        if graph_property is None:
            graph_property = graph.vp.type
        if subg_property is None:
            subg_property = subg.vp.type
        property_maps = (subg_property, graph_property)
        with temp_timer("find_subgraph isom"):
            maps = subgraph_isomorphism(subg, graph, vertex_label=property_maps, generator=True)
        with temp_timer("find_subgraph sortout"):
            subs = set()
            subs_orderd = []
            i = 0
            if verbose:
                print ("Searching for unique subgraphs")
            for m in maps:
                if i%100 == 0 and verbose:
                    print (f"\r processed {i+1} subgraphs", end="")
                i += 1
                m = tuple(m) 
                sm = frozenset(m)
                if keep_order:
                    if sm not in subs:
                        subs_orderd.append(m)
                        subs.add(sm)
                else:
                    # no need to keep the map order to match atoms from sub to graph
                    subs.add(sm)
            if keep_order:
                subs = subs_orderd
            else:
                subs = [tuple(s) for s in subs]
        if verbose:
            temp_timer.report()
        return subs

    # the following static methods are used in the fragments addon (to avoid graphtool dependence of the rest of fragments)
    @staticmethod
    def get_kcore(graph):
        return kcore_decomposition(graph).get_array()


    def find_sub(self, subg, N=0, verbose=False, keep_order=False):
        """
        use graph_tools subgraph_isomorphism tool to find substructures

        :Parameter:

            - subg : graph object (from another molsys) to be searched

        :Returns:

            a list of lists with the (sorted) vertex indices of the substructure
        """
        subs = self.find_subgraph(self.molg, subg.molg, N=N, verbose=verbose, keep_order=keep_order)
        return subs

    def check_sub(self, subg):
        """check if subg found in self.graph
        
        Args:
            subg (mol.graph objects): subgraph to be tested
        """
        subs = self.find_subgraph(self.molg, subg.molg, N=1)
        if subs != []:
            return True
        else:
            return False
        

    def find_fragment(self, frag,add_hydrogen=False):
        """
        find a complete fragment (including the hydrogen atoms not included in the graph)
        Note that the fragment found can be different from the fragment by the number of hydrogen atoms!!

        RS 2024: revision fo fragmentize

        We return the lists of atoms with the hydrogens appended at the end in order to keep
        the ordering from the substructure search. This is important for the annotation of atoms.

        :Parameter:

            - frag : mol object with graph addon to be found

        :Returns:

            a list of lists with the atom indices of the fragment in the full system
        """
        subs = self.find_sub(frag.graph, keep_order=True)
        frags = []
        for s in subs:
            # loop over all vertices
            f = []
            fh = []
            for v in s:
                a = self.molg.vp.atomid[v]
                f.append(a)
                # check all atoms connected to this atom if they are hydrogen
                if add_hydrogen:
                    for ca in self._mol.conn[a]:
                        if self._mol.elems[ca] == "h":
                            fh.append(ca)
            f.extend(fh)            
            frags.append(f)
        return frags

    def util_graph(self, vertices, conn, atom_map=None, vtypes2 = None):
        """generates a fragment or atom graph

        Args:
            vertices (list of strings): vertex identifier
            conn (list of lists): connectivity 
            atom_map (list of list of ints, optional): atoms mapped by this vertex. Defaults to None.
            vtypes2 (list of strings, optional): alternative vertex identifiers. Defaults to None.

        Returns:
            graph: graph object    
        
        RS: this is currently just a helper function that produces the graph and returns it.
            we could consider to store this graph with a name in the graph addon

        """
        if vtypes2 is not None:
            assert len(vtypes2)==len(vertices)
        if atom_map is not None:
            assert len(atom_map) == len(vertices)
        g = Graph(directed=False)
        # now add vertices
        g.vp.type = g.new_vertex_property("string")
        if vtypes2 is not None:
            g.vp.type2 = g.new_vertex_property("string")
        if atom_map is not None:
            g.vp.atom_map = g.new_vertex_property("vector<int>")
        for i, v in enumerate(vertices):
            g.add_vertex()
            g.vp.type[i] = v
            if vtypes2 is not None:
                g.vp.type2[i] = vtypes2[i]
            if atom_map is not None:
                g.vp.atom_map[i] = atom_map[i]
        # now add edges ...
        for i, v in enumerate(vertices):
            for j in conn[i]:
                if j>=i:
                    g.add_edge(g.vertex(i), g.vertex(j))
        return g

    def filter_graph(self, idx):
        """
        filters all atoms besides the given out of the graph
        :Parameters:
            - idx (list): indices of atoms to keep
        """
        # TODO use vert2atom
        assert type(idx) == list
        self.molg.clear_filters()
        filter = self.molg.new_vertex_property("bool")
        filter.set_value(False)
        for i in idx:
            filter[self.molg.vertex(i)]=True
        self.molg.set_vertex_filter(filter)
        return

    def unfilter_graph(self):
        self.molg.clear_filters()
        return

    def get_components(self, fidx= None):
        """Get all the components aka molecules from the atomic graph

        it adds a property map molid to the graph

        Returns:
            number of components found

        """
        if fidx is not None:
            # filter 
            self.filter_graph(fidx)
        label_components(self.molg, vprop=self.molg.vp.molid)
        ncomp = max(self.molg.vp.molid.a)+1
        if fidx is not None:
            self.unfilter_graph()
        comp = list(self.molg.vp.molid.a)
        return ncomp, comp

