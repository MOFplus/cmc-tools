"""

                  extra classes used in findR

                  frame    -> store a frame
                  species  -> store a species in a frame (or just as such)
                  fcompare -> frame comparer
                  revent   -> reaction event

##################### NOTES /TO BE IMPLEMENTED

add xyz coord reference to frame object for mol object generation
-> not completely implemented

add pmol also to species
move molg generation to the frame and species objects

"""

import numpy as np
import os

import molsys
from molsys.util import rotations

from graph_tool import Graph, GraphView
import graph_tool.topology as gtt
import graph_tool.util as gtu
import graph_tool.draw as gtd

import uuid
import graph_tool

#####################  FRAME CLASS ########################################################################################

class frame:
    """container to store info on a frame

    the main attributes are:
    - fid: frame id 
    - xyz:     numpy array (N,3) with the xyz coords of all atoms in this frame
    - elems:   list of elems
    - bondtab: bond table (M,2) with m bonds
    - bondord: bon orders (M)
    """

    def __init__(self, fid, xyz, mol, bondtab, bondord, cutoff=0.5, min_atom=6, min_elems={"c":2}):
        self.fid = fid
        self.xyz = xyz
        self.mol = mol
        self.natoms = self.mol.get_natoms()
        self.bondord = bondord
        self.bondtab = bondtab
        self.maxbond = len(bondord)
        self.specs = {}
        # generate the molecular graph
        molg = Graph(directed=False)
        molg.add_vertex(n=self.natoms)
        molg.vp.aid  = molg.new_vertex_property("int")
        molg.vp.aid.a[:] = np.arange(self.natoms, dtype="int32")
        molg.vp.filt = molg.new_vertex_property("bool")
        molg.vp.mid  = molg.new_vertex_property("int")
        molg.ep.bord = molg.new_edge_property("float")
        molg.ep.filt = molg.new_edge_property("bool")
        for j in range(self.maxbond):
            o = bondord[j]
            # invalid entries are marked with a bondord = -1.0
            if o < 0.0:
                break
            e = bondtab[j]-1
            # TO BE REMOVED ... this is a legacy check for the old incorrect mfp5 files 
            if (e[0] < 0 or e[0] >= self.natoms or e[1] < 0 or e[1] >= self.natoms):
                break
            # END TO BE REMOVED
            newb = molg.add_edge(e[0], e[1])
            isbond = o > cutoff
            molg.ep.bord[newb] = o
            molg.ep.filt[newb] = isbond 
        # apply edge filter
        molg.set_edge_filter(molg.ep.filt)
        self.molg = molg
        self.mid, hist = gtt.label_components(molg, vprop=molg.vp.mid)
        nspecies_all = len(hist)
        nspecies = []
        # collect number of critical elements in the species
        elem_spec = {}
        for e in min_elems.keys():
            spec_ecount = np.zeros(nspecies_all, dtype="int32")
            for i in range(self.natoms):
                if self.mol.elems[i] == e:
                    spec_ecount[self.mid[i]] += 1
            elem_spec[e] = spec_ecount
        for i in range(nspecies_all):
            track = True
            if hist[i] < min_atom:
                track = False
            for e in min_elems.keys():
                if elem_spec[e][i] < min_elems[e]:
                    track = False
            if track:
                nspecies.append(i)
        # append the species
        for s in nspecies:
            self.add_species(s)
        return

    def add_species(self, mid, tracked=True):
        # make all frame species (observed) with a local graph representation
        self.specs[mid] = species(self.fid, mid, self.molg, make_graph=True, tracked=tracked)
        return

    def make_species(self, mid):
        # return a species object (without a local graph)
        return species(self.fid, mid, self.molg, make_graph=False, tracked=False)


    @property
    def nspecies(self):
        return len(self.specs)

    def plot(self, selection=False):
        if selection:
            print ("plotting species %s of frame %d" % (str(selection), self.fid))
            for s in selection:
                gfname = "frame_%d_species_%d.png" % (self.fid, s)
                g = self.specs[s].graph
                pos = gtd.arf_layout(g, max_iter=0)
                gtd.graph_draw(g, pos=pos, vertex_text=g.vp.aid, vertex_font_size=12, vertex_size=8, \
                                           edge_text=g.ep.bord, edge_font_size=10,\
                output_size=(800, 800), output=gfname, bg_color=[1,1,1,1])
        else:            
            print ("plotting all species of frame %d" % self.fid)
            for s in self.specs.keys():
                gfname = "frame_%d_species_%d.png" % (self.fid, s)
                g = self.specs[s].graph
                pos = gtd.arf_layout(g, max_iter=0)
                gtd.graph_draw(g, pos=pos, vertex_text=g.vp.aid, vertex_font_size=12, vertex_size=8, \
                output_size=(200, 200), output=gfname, bg_color=[1,1,1,1])
        return

    def get_border(self, b):
        """get the bond order for a bond b
        
        currently no test is made if the edge exists ... you need to check

        Args:
            b (tuple of indices): atom indices of bond
        """
        e = self.molg.edge(b[0], b[1])
        border = self.molg.ep.bord[e]
        return border

    def make_mol(self, species):
        """generate a single mol object from a list of species for a TS

        NOTE: the additional species added to the frame are considered to be non tracked

        """
        aids = []
        for mid in species:
            if mid not in self.specs:
                self.add_species(mid, tracked=False)
            s = self.specs[mid]
            aids += list(s.aids)
        aids.sort() # This is necessary to ensure same ordering in mol object 
        mol = molsys.mol.from_array(self.xyz[aids])
        mol.set_cell(self.mol.get_cell())
        mol.set_elems([self.mol.elems[i] for i in aids])
        mol.set_real_mass()
        mol.center_com(check_periodic=False)
        mol.apply_pbc()
        mol.make_nonperiodic()
        # rotate into principal axes
        xyz = mol.get_xyz()
        mol.set_xyz(rotations.align_pax(xyz, masses = mol.get_mass()))
        mol.set_atypes(mol.get_elems())
        # add connectivity
        conn = []
        for i in aids:
            v = self.molg.vertex(i)
            conn_i = []
            for j in v.all_neighbors():
                assert int(j) in aids
                conn_i.append(aids.index(int(j)))
            conn.append(conn_i)
        mol.set_conn(conn)
        return mol, aids

    def make_species_mol(self):
        """get a list of mol objects for all species in the frame

        NOTE: we use here the global xyz and pmol -> should be delegated to the species object which needs refactoring
        - we should attacg 
        """
        mols = {}
        for s in self.specs:
            sp = self.specs[s]
            sxyz = self.xyz[list(sp.aids)]
            mols[s] = self.specs[s].make_mol(sxyz, self.mol) 
        return mols

    ### some DEBUG methods ###

    def write_species(self):
        foldername = "frame_%d_species" % self.fid
        os.mkdir(foldername)
        os.chdir(foldername)
        mols = self.make_species_mol()
        for s in mols:
            m = mols[s]
            m.write("spec_%d.mfpx" % s)
        os.chdir("..")

    def get_main_species_formula(self):
        """get the sumformula of the tracked species
        """
        sp = list(self.specs.keys())
        sp.sort()
        sumforms = []
        for s in sp:
            aids = list(self.specs[s].aids)
            elems = [self.mol.elems[i] for i in aids]
            cont  = list(set(elems))
            cont.sort()
            sumform = ""
            for e in cont:
                sumform += "%s%d " % (e, elems.count(e))
            sumforms.append(sumform)
        return sumforms

    def DEBUG_write_as_xyz(self, fname):
        f = open(fname, "w")
        f.write("%d\n\n" % self.natoms)
        for i in range(self.natoms):
            x, y, z = self.xyz[i]
            f.write("%3s %12.6f %12.6f %12.6f\n" % (self.mol.elems[i], x, y, z))
        f.close()
        return

#####################  SPECIES CLASS ########################################################################################

class species:
    """container class to keep species info (per frame!)
    """

    def __init__(self, fid, mid, molg, make_graph=False, tracked = True):
        """init species
        
        Args:
            fid (int): frame number
            mid (int): molecule id ("name" of species from label components)
            molg (graph): molg of the frame
        """
        self.fid = fid
        self.mid = mid
        self.molg = molg # parent molgraph
        # find all vertices in molg that belong to this species mid
        vs = gtu.find_vertex(molg, molg.vp.mid, mid)
        self.aids = set([int(v) for v in vs]) # atomids -> TBI do we need to sort? seems like they are sorted as they come
        self.graph = None
        if make_graph:
            # now make a view of molg for this species         
            self.make_graph()
        self.tracked = tracked
        return

    def make_graph(self):
        vfilt = self.molg.new_vertex_property("bool")
        vfilt.a[:] = False
        for v in self.aids:
            vfilt[v] = True
        self.graph = GraphView(self.molg, vfilt=vfilt)
        return

    @property
    def natoms(self):
        return len(self.aids)

    def make_mol(self, xyz, pmol):
        """generate a mol object for the species from a frame

        we get the current coordinates as xyz and the parent mol object pmol from the mfp5 file
        
        Args:
            xyz (numpy): coordinates of the atoms
            pmol (molsys object): parent mol object from mfp5 file 
        """
        aids = list(self.aids) # in order to map backwards
        self.mol = molsys.mol.from_array(xyz)
        if pmol.get_cell() is not None:
            self.mol.set_cell(pmol.get_cell())
        self.mol.set_elems([pmol.elems[i] for i in aids])
        self.mol.set_real_mass()
        self.mol.center_com(check_periodic=False)
        self.mol.apply_pbc()
        self.mol.make_nonperiodic()
        # rotate into principal axes
        xyz = self.mol.get_xyz()
        self.mol.set_xyz(rotations.align_pax(xyz, masses = self.mol.get_mass()))
        self.mol.set_atypes(self.mol.get_elems())
        # add connectivity
        if self.graph is None:
            self.make_graph()
        ctab = []
        for e in self.graph.edges():
            i = aids.index(int(e.source()))
            j = aids.index(int(e.target()))
            ctab.append([i, j])
        self.mol.set_ctab(ctab, conn_flag=True)
        return self.mol

    def __eq__(self, other):
        """compare two species whether they are equal

        Compares "exactly" .. atom indices (wrt frame) and bonding must match exactly.
        This can not be used to find similar species (same bonding etc but different atom numbers)
        Note: The conformation of the species can be entirely differnt -> we do not compare postions!

        Args:
            other (species object): the other species object

        Returns:
            bool: if they are equal or not equal
        """
        if not type(self) is type(other):
            return NotImplemented
        # check if atom IDs are equal
        if self.aids != other.aids:
            return False
        # generate a set of bonds for each species
        bself  = set([(int(e.source()),int(e.target())) for e in self.graph.edges()])
        bother = set([(int(e.source()),int(e.target())) for e in other.graph.edges()])
        if bself != bother:
            return False

        # if we are here the species are equal
        return True
        
    def __repr__(self):
        return "SPECIES{frame %5d / molid %3d (%4d atoms)}" % (self.fid, self.mid, self.natoms)

#####################  FRAME COMPARE CLASS ########################################################################################

class fcompare:
    """This class compares two frames at different levels of resolution whether species are different
       All info collected during the compairson is kept in this class in order to be exploited later
    """

    def __init__(self, f1, f2):
        """generate comparer
        
        Args:
            f1 (frame object): first frame object
            f2 (frame object): second frame object (to be compared with f1)
        """
        self.f1 = f1
        self.f2 = f2
        # defaults
        self.compare_level = 0 # 0: no comparison, 1: atom id level, 2: connectivity level
        self.umatch_f1 = [s for s in self.f1.specs if self.f1.specs[s].tracked]
        self.umatch_f2 = [s for s in self.f2.specs if self.f2.specs[s].tracked]
        self.aids_match = []
        self.bond_match = []
        self.aids_analyzed = False
        self.bonds_analyzed = False
        self.reacs = []
        self.broken_bonds = []
        self.formed_bonds = []
        self.nreacs = 0 # number of independent reactions for this pair of frames .. should always be 1 (??)
        return

    def report(self, all = False):
        """this method is just to implement all levels of comparison

           it does not return anything and just reports ... meant for debugging
        """
        print("##################################################")
        print("FRAMES        %5d         %5d" % (self.f1.fid, self.f2.fid))
        mf = self.check_aids()
        if all:
            for m in self.aids_match:
                print("          species %5d == species %5d" % m)
                print(" natoms:          %5d            %5d" % (self.f1.specs[m[0]].natoms, self.f2.specs[m[1]].natoms))
        if mf > 0:
            self.analyse_aids()
            for r in self.reacs:
                print ("educts  : %s" % str(list(r[0].keys())))
                print ("products: %s" % str(list(r[1].keys())))
        return

    def check_aids(self, verbose = False):
        """check on level 1 for atom id matches

        this method implements a rather complex logic:
        which species form the educts and products to define a complete reaction event
        """
        if self.compare_level < 1:
            # we need to compare
            for sk1 in self.f1.specs:
                s1 = self.f1.specs[sk1]
                # find a corresponding species in frame 2
                for sk2 in self.umatch_f2:
                    s2 = self.f2.specs[sk2]
                    if s1.aids == s2.aids:      # NOTE: this works because the vertices are always properly sorted
                        # s1 and s2 match -> add to match and to remove
                        self.aids_match.append((sk1, sk2))
                        self.umatch_f1.remove(sk1)
                        self.umatch_f2.remove(sk2)
                        break
            # now matches are found -> set compare level
            self.compare_level = 1
        match_flag = 0 # all is equal on level 1
        if len(self.umatch_f1) > 0 or len(self.umatch_f2) > 0:
            match_flag = 1 # unmatched species!!
        if verbose:
            print ("species in f1   : %s" % str(self.f1.specs.keys()))
            print ("species in f2   : %s" % str(self.f2.specs.keys()))
            print ("unmatched in f1 : %s" % str(self.umatch_f1))
            print ("unmatched in f2 : %s" % str(self.umatch_f2))

        return match_flag

    def analyse_aids(self):
        if self.aids_analyzed:
            return
        if self.compare_level == 0:
            self.check_aids()
        if len(self.umatch_f1)==0 and len(self.umatch_f2)==0:
            # there is nothing to do
            return
        # we have some unmatched species -> there is one (or more) reaction(s) between these frames
        # find groups of species that define a reaction
        #    all atom ids in the union of the educts sets must be also in the products set
        for sk1 in self.umatch_f1:
            # first search for atoms in the unmatched f2 species
            s1 = self.f1.specs[sk1]
            educts = {sk1: s1} # dictionary of species key/species for this reaction
            educt_aids = s1.aids.copy() # set of atomids
            products = {}
            product_aids = set()
            for sk2 in self.umatch_f2:
                s2 = self.f2.specs[sk2]
                common_aids = s1.aids & s2.aids
                if len(common_aids)>0:
                    products[sk2] = s2
                    product_aids |= s2.aids
            # do the following until educt_aids and product_aids match
            while not (educt_aids == product_aids):
                # which atoms are in products that are not in the educt species? --> add them
                for a in product_aids - educt_aids:
                    # to which species in frame 1 does this atom belong to?
                    esk = self.f1.molg.vp.mid[a]
                    # is this already in educts?
                    if esk not in educts:
                        # we need to make a new species object and add it (as non-tracked)
                        educts[esk] = self.f1.make_species(esk)
                        educt_aids |= educts[esk].aids

                        # avoid that found educt is in first list. In that case you will find a reaction twice
                        if esk in self.umatch_f1:
                            self.umatch_f1.remove(esk)

                # which atoms are in the educts that are not in the product species? add them
                for a in educt_aids - product_aids:
                    # to which species in frame 2 does this atom belong to?
                    psk = self.f2.molg.vp.mid[a]
                    # is this already in educts?
                    if psk not in products:
                        # we need to make a new species object and add it (as non-tracked)
                        products[psk] = self.f2.make_species(psk)
                        product_aids |= products[psk].aids

                        # GS: unsure if this is required. Should avoid to assign a product twice
                        if psk in self.umatch_f2:
                            self.umatch_f2.remove(psk)
                        
            self.reacs.append((educts, products))
        # the above will not work if there is no tracked species in umatch_f1 (but in umatch_f2)
        # ... in other words a species has "appeared" or formed by merging two or more untracked species
        if len(self.umatch_f1)==0:
            for sk2 in self.umatch_f2:
                s2 = self.f2.specs[sk2]
                products = {sk2: s2}
                product_aids = s2.aids.copy()
                # now find all the species in frame 1 to match the atoms 
                educts = {}
                educt_aids = set()
                for a in product_aids:
                    esk = self.f1.molg.vp.mid[a]
                    if esk not in educts:
                        educts[esk] = self.f1.make_species(esk)
                        educt_aids |= educts[esk].aids
                # do the following until educt_aids and product_aids match
                while not (educt_aids == product_aids):
                    # which atoms are in products that are not in the educt species? --> add them
                    for a in product_aids - educt_aids:
                        # to which species in frame 1 does this atom belong to?
                        esk = self.f1.molg.vp.mid[a]
                        # is this already in educts?
                        if esk not in educts:
                            # we need to make a new species object and add it (as non-tracked)
                            educts[esk] = self.f1.make_species(esk)
                            educt_aids |= educts[esk].aids


                    # which atoms are in the educts that are not in the product species? add them
                    for a in educt_aids - product_aids:
                        # to which species in frame 2 does this atom belong to?
                        psk = self.f2.molg.vp.mid[a]
                        # is this already in educts?
                        if psk not in products:
                            # we need to make a new species object and add it (as non-tracked)
                            products[psk] = self.f2.make_species(psk)
                            product_aids |= products[psk].aids
                # now add the final results to the reacs list
                self.reacs.append((educts, products))   
        self.nreacs = len(self.reacs)
        self.aids_analyzed = True
        return

    def find_react_bond(self):
        """find a reactive bond in a bimolecular reaction 
        
        Args:
            r (int): index in self.reac to analyse
        """
        assert self.aids_analyzed
        g1 = self.f1.molg
        g2 = self.f2.molg
        for r in range(self.nreacs):
            broken_bonds = []
            formed_bonds = []
            educts, products = self.reacs[r]
            # get all involved atoms
            aids = set()
            for s in educts:
                aids |= educts[s].aids
            for a in aids:
                bonds1 = []
                v1 = g1.vertex(a)
                for e in v1.out_edges(): 
                    bonds1.append(int(e.target()))
                bonds2 = []
                v2 = g2.vertex(a)
                for e in v2.out_edges():
                    bonds2.append(int(e.target()))
                bs = set(bonds1) - set(bonds2)
                fs = set(bonds2) - set(bonds1)
                for b in bs:
                    if b > a:
                        broken_bonds.append((a,b))
                for b in fs:
                    if b > a:
                        formed_bonds.append((a,b))
            self.broken_bonds.append(broken_bonds)
            self.formed_bonds.append(formed_bonds)
        return

    def check_bonds(self, verbose = False):
        """check on level 2 for identical bonds

        we use the existing pairs of species in self.aids_match to identify species that have identical aids
        => now we test if they have the same bonding pattern
        """
        assert self.compare_level > 0
        self.missmatch = []
        for p in self.aids_match:
            # get the species of the pair
            s1 = self.f1.specs[p[0]]
            s2 = self.f2.specs[p[1]]
            # # TBI: this might not be enough 
            # #      do we need elements as vertex properties to be considered?
            # #      what about a tautomerism when the initial and fianl state are symmetric?
            # f = gtt.isomorphism(s1.graph, s2.graph)
            # INDEED: isomoprhism is not strict enough
            # print ("%d %d %d %d %s" % (self.f1.fid, self.f2.fid, p[0], p[1], f))
            if  s1 == s2:
                self.bond_match.append(p)
            else:
                self.missmatch.append(p)
        self.compare_level = 2
        if len(self.missmatch) > 0:
            # unmatched species on level 2
            return 2
        else:
            return 0 # all equal

    def analyse_bonds(self):
        if self.bonds_analyzed:
            return
        if self.compare_level < 2:
            self.check_bonds()
        if len(self.missmatch) == 0:
            return
        # now analyse all the pairs in self.missmatch: they have identical aids but a diffrent bond graph
        self.nreacs = len(self.missmatch)
        for p in self.missmatch:
            # get the species of the pair
            s1 = self.f1.specs[p[0]]
            s2 = self.f2.specs[p[1]]
            self.reacs.append(({p[0]: s1},{p[1]: s2}))
            # get all edges as vertex id tuples (int tuples)
            bs1 = set([(int(e.source()),int(e.target())) for e in s1.graph.edges()])
            bs2 = set([(int(e.source()),int(e.target())) for e in s2.graph.edges()])
            self.broken_bonds.append(list(bs1-bs2)) # broken: bond in frame1 but not in frame2
            self.formed_bonds.append(list(bs2-bs1)) # formed: bond in frame2 but not in frame1
        self.bonds_analyzed = True
        return


    def check(self, verbose = False):
        """check identity of species on all levels (1 and 2) 
        """
        mf = self.check_aids(verbose=verbose)
        if mf > 0:
            return mf
        # aids are equal -> chek bonds
        return self.check_bonds(verbose=verbose)

#####################  REACTIVE EVENT CLASS ########################################################################################
# this class is instantiated with a comparer (between two frames)
# and stores a reactive event 
# it knows the TS_fid and if the event is bi- or unimolecular

class revent:

    def __init__(self, comparer, fR, unimol= False, ireac=0):
        """generate a reaction event object

        TBI: what to do if there are more then one reaction event (in two diffrent tracked species)
             at the same time ... this is properly tracked in the comparer (nreacs>1)
             but this means there are more revents to be generated. should we do this recursively?
             how to store?
        
        Args:
            comparer (fcompare object): comparer that gave a reactive event
            fR (parent findR object): to access process_frame
            unimol (bool, optional): is unimolecular. Defaults to False.
            ireac (int,optional): specifies the reaction event
        """
        self.unimol = unimol
        self.fR = fR
        self.comparer = comparer
        # currently we allow only for single reaction events per frame 
        # this would change if there is more than one tracked species ....
        #assert comparer.nreacs == 1 , "Currently only single reaction events are processed. Error occured for frames %s and %s" % (comparer.f1.fid, comparer.f2.fid)
        #r = 0 # pick first reaction (the only one)
        assert comparer.nreacs > ireac, "Too many reaction events requested..."
        r = ireac
        print("ireac " + str(ireac) + " frames " + str(comparer.f1.fid) + " and " + str(comparer.f2.fid) )
        educts, products = comparer.reacs[r]
        self.broken_bonds     = comparer.broken_bonds[r]
        self.formed_bonds     = comparer.formed_bonds[r]
        f1               = comparer.f1
        f2               = comparer.f2
        # choose which frame we use as TS
        # everything is referenced to f1 which is 0 (f2 is +1)
        # find avereage bond order of reactive bonds at f1/f2
        f1_averborder = 0.0
        if len(self.broken_bonds) >0:
            for b in self.broken_bonds:
                f1_averborder += f1.molg.ep.bord[f1.molg.edge(b[0], b[1])]
            f1_averborder/len(self.broken_bonds)
        f2_averborder = 0.0
        if len(self.formed_bonds) >0:
            for b in self.formed_bonds:
                f2_averborder += f2.molg.ep.bord[f2.molg.edge(b[0], b[1])]
            f2_averborder/len(self.formed_bonds)
        if f1_averborder == 0.0:
            TS_rfid = 1
        elif f2_averborder == 0.0:
            TS_rfid = 0
        else:
            if abs(f1_averborder-0.5) < abs(f2_averborder-0.5):
                # f1 closer to TS
                TS_rfid = 0
            else:
                TS_rfid = 1
        # now store data depending on relative frame id (rfid) of the TS
        if TS_rfid == 0:
            self.TS = f1
            self.PR = f2
            self.ED = self.fR.process_frame(self.TS.fid-1)
            # get corresponding species numbers
            self.TS_spec = educts
            self.PR_spec = products
            loccomp = fcompare(self.ED, self.TS)
            if loccomp.check_aids() == 0:
                # no change in atom ids .. we can use TS species for ED as well
                self.ED_spec = {}
                for e in educts:
                    if e in self.ED.specs:
                        self.ED_spec[e] = self.ED.specs[e]
                    else:
                        self.ED_spec[e] = self.ED.make_species(e)
            else:
                print ("Houston we have a problem!!! species changed between ED and TS")
        else:
            self.ED = f1
            self.TS = f2
            if self.fR.nframes > self.TS.fid+1:
                self.PR = self.fR.process_frame(self.TS.fid+1)
                self.ED_spec = educts
                self.TS_spec = products
                loccomp = fcompare(self.TS, self.PR)
                if loccomp.check_aids() == 0:
                    # no change in atom ids .. we can use TS species for PR as well
                    self.PR_spec = {}
                    for e in products:
                        if e in self.PR.specs:
                            self.PR_spec[e] = self.PR.specs[e]
                        else:
                            self.PR_spec[e] = self.PR.make_species(e)
                else:
                    print ("Houston we have a problem!!! species changed between TS and PR")
            else:
                print("Warning! We run out of frames to process to identify TS and PR.")
        # get TS_fid for ease
        self.TS_fid = self.TS.fid


        return
