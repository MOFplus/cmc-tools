"""topo is an addon for topo mol objects (we assert this at the init)

By adding this addon to a topo mol object you get a number of additional features relevant for working with topologies

Features:
    - lqg: adds the labeled quotient graph object to the mol object
        -> generate a barycentreic embedding
        -> compute the systre key and detect the name of the topo

Note: we store the systrekey internally without the prepended "3 " which is redundant. but for reporting we add it

"""

# from molsys.util.lqg import lqg
from molsys.util import systrekey
from molsys.util.images import idx2arr,arr2idx
from molsys.util import elems

# make a systre key database 
skdb = systrekey.RCSR_db

class topo:

    def __init__(self, mol):
        assert mol.is_topo
        self._mol = mol
        # we need pconn here in any case for many things so add it if it is not there, yet
        if not self._mol.use_pconn:
            self._mol.add_pconn()
        self._lqg = None             # the lqg object of the actual topo object
        self._systrekey = None       # the systrekey lqg object
        self._sk_vmapping = None     # the vertex mapping between lqg and sk
        self._sk_emapping = None     # the edge mapping between lqg and sk
        self._RCSRname   = None      # the RCSR name if known (present in the RSCR.arc file)
        self._info = {} # dictionary with metainfo which can be present or not (most comes from systre runs)
        # color
        self.has_color = False
        return

    def get_info(self, name):
        if name in self._info:
            return self._info[name]
        else:
            return None

    def set_info(self, name, value):
        self._info[name] = value
        return

    def available_info(self):
        return self._name.keys()

    def clear_info(self):
        self._info = {}
        return

    @property
    def lqg(self):
        """method to register the topos labeled quotient graph
        """
        if self._lqg is None:
            edges = []
            labels = []
            for i in range(self._mol.get_natoms()):
                for j,v in enumerate(self._mol.conn[i]):
                    if v >= i:
                        edges.append([i,v])
                        # NOTE: in some cases pconn contains FLOAT numbers which is wrong!!! find out where thsi comes from!! and who did it?
                        labels.append(self._mol.pconn[i][j].astype("int32").tolist())
            self._lqg = systrekey.lqg(edges, labels)
        return self._lqg

    @property
    def systrekey(self):
        """method calls javascript systreKey (must be installed) and computes the systreKey
        """
        if self._systrekey is None:
            lqg = self.lqg
            self._systrekey = lqg.get_systrekey()     
            # store the vertex mapping in fragnumbers
            self._mol.set_fragnumbers(lqg.sk_vmapping)
            # connect mappings
            self._sk_vmapping = lqg.sk_vmapping
            self._sk_emapping = lqg.sk_emapping
        return self._systrekey

    @property
    def RCSRname(self):
        if self._RCSRname is None:
            try:
                self._RCSRname = skdb.get_name(repr(self.systrekey))
            except KeyError:
                self._RCSRname = "unknown"
        return self._RCSRname

    @property
    def sk_vmapping(self):
        if self._sk_vmapping is None:
            # this triggers the computing of the systrekey ... we do not need it because the info is 
            # in the lqg object after that
            sk = self.systrekey
        return self._sk_vmapping

    @property
    def sk_emapping(self):
        if self._sk_emapping is None:
            # this triggers the computing of the systrekey ... we do not need it because the info is 
            # in the lqg object after that
            sk = self.systrekey
        return self._sk_emapping

    @property
    def spgr(self):
        return self.get_info("spgr")
    
    @property
    def transitivity(self):
        return self.get_info("transitivity")

    @property
    def coord_seq(self):
        return self.get_info("coord_seq")

    def set_topoinfo(self, skey=None,
                         vmapping = None,
                         emapping = None, 
                         spgr = None, 
                         RCSRname = None, 
                         coord_seq = None, 
                         transitivity = None,
                         ):
        """
        In case we read a topo object from file this meat info could be just set and no need to 
        runs systreky (which might not even be installed or work)

        RS this is quite complicated .. we should simplify this at some point with args and kwargs
        """
        self._systrekey = skey
        self._sk_vmapping = vmapping
        self._sk_emapping = emapping
        self._RCSRname = RCSRname
        if spgr is not None: 
            self._info["spgr"] = spgr
        if coord_seq is not None: 
            self._info["coord_seq"] = coord_seq
        if transitivity is not None:
            self._info["transitivity"] = transitivity
        return

    def fix_topo_elems(self):
        """call this method to set the elements in a topo file properly (from coord number)
        """
        new_elems = [elems.pse[len(c)] for c in self._mol.conn]
        self._mol.set_elems(new_elems)
        return

    def extend_images(self):
        """helper method to generate a new mol object where the periodic bonds are extended
        to the next image as defined by the pconn
        """
        nm = self._mol.clone()
        nm.unmake_topo()
        del_bonds = []
        for i in range(self._mol.get_natoms()):
            for ij,j in enumerate(self._mol.conn[i]):
                if arr2idx[self._mol.pconn[i][ij]] != 13:
                    # this is a periodic bond .. add image atom
                    xyz_image_j = self._mol.xyz[j]+(self._mol.cell*(self._mol.pconn[i][ij][:,None])).sum(axis=0)
                    new_atom = nm.add_atom(nm.get_elems()[j],nm.atypes[j], xyz_image_j)
                    if j>i:
                        del_bonds.append((i,j))
                    nm.add_bond(i, new_atom)
        for b in del_bonds:
            nm.delete_bond(b[0], b[1])
        return nm

    # colored topo
    #
    # the acab addon is used to make colored topos but we use these data structures
    # to store the coloring info.
    def set_color(self, vcolor, ecolor):
        self.has_color = True
        self.ecolor = ecolor
        self.vcolor = vcolor
        self.nvc = max(vcolor)
        self.nec = max(ecolor)
        assert list(range(self.nvc+1)) == list(set(vcolor))
        assert list(range(self.nec+1)) == list(set(ecolor))
        return

    # remove all vertices of a specific color
    def del_colored_vertices(self, col, del_edges=True):
        # before we can use the mol objects method to delete all vertices of the given vertex color we need to
        # determine all bonds that will go to fix up ecolors
        assert col in self.vcolor
        vremove = [i for i,c in enumerate(self.vcolor) if c == col]
        eremove = [i for i,b in enumerate(self._mol.ctab) if ((b[0] in vremove) or (b[1] in vremove))]
        self._mol.delete_atoms(vremove)
        colors = list(range(self.nvc+1))
        colors.remove(col)
        new_vcolor = [colors.index(c) for i,c in enumerate(self.vcolor) if i not in vremove]
        new_ecolor = [c for i,c in enumerate(self.ecolor) if i not in eremove]
        self.set_color(new_vcolor, new_ecolor)
        self.set_topoinfo(skey = "invalid") # this invalidates all the topoinfo, which is useless after this crippling
        self.clear_info()
        # reset element codes
        self.fix_topo_elems()
        return

    # remove all edges of a given color
    def del_colored_edges(self, col):
        assert col in self.ecolor
        eremove = []
        for i, e in enumerate(self._mol.etab):
            if self.ecolor[i] == col:
                # remove this edge
                self._mol.delete_bond(e[0], e[1])
                eremove.append(i)
        # now recreate etab, since delete_bonds operates only on conn and pconn we need to repait pimages as well
        self._mol.set_pimages_from_pconn()
        self._mol.set_etab_from_conns()
        # now remove entries in ecolor
        self.ecolor = [c for c in self.ecolor if c != col]
        self.set_topoinfo(skey = "invalid") # this invalidates all the topoinfo, which is useless after this crippling
        self.clear_info()
        # reset element codes
        self.fix_topo_elems()
        return