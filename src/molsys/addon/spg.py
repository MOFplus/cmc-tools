# -*- coding: utf-8 -*-
"""

    spg

    implements an addon to access the features of the spglib within molsys
    https://atztogo.github.io/spglib/

    you need a recent spglib (>= 1.9.0) because the python import has changed at some point
    it can be installed via pip (pip install spglib)

    comment by JK: i've added spacegroups.py in util containing a (an incomplete) list of spacegroup
    strings with the corresponding spacegroup numbers.
    molsys.util.spacegroups (herein imported as spacegroup) call via:
    spacegroup.get_spacegroup_number(sgname). returns None if not in dict.

Created on Wed Dec  7 15:44:36 2016

@author: rochus
"""

import spglib
import numpy
from molsys.util import elems
from molsys.util import spacegroups
import molsys
import sys
import numpy as np
import copy

import logging
logger = logging.getLogger("molsys.spg")
logger.setLevel(logging.INFO)

from molsys.util.misc import sort_by_columns, argsort_by_columns, sort_by_columns_in_place
from molsys.util.sysmisc import isatty

def get_frac_match(frac, sym, thresh=5e-6, eps=1e-8):
    symperm = []
    x = frac[np.newaxis,:]-sym[:,np.newaxis]
    whereint = np.where(np.isclose(np.round(x), x, atol=eps))
    x[whereint] = np.round(x[whereint]) + eps
    #x -= np.floor(x) ### np.round does not work! [?RA] MAYBE WE FLOOR x+.5
    #symperm = np.where((x<thresh).all(axis=-1))[-1]
    x -= np.floor(x+.5) ### np.round does not work! [?RA] MAYBE WE FLOOR x+.5
    symperm = np.where((abs(x)<thresh).all(axis=-1))[-1]
    return symperm

class spg:

    def __init__(self, mol):
        """
        generate a spg object

        :Parameters:

            - mol: mol object to be kept as a parent ref
        """
        self.mol = mol
        self.spgcell = None # tuple of (lattice, position, numbers) as used in spglib
        self.spg_version = spglib.get_version()
        self.symprec = 1.0e-2
        logger.info("Addon spg loaded (version %d.%d.%d)" % self.spg_version)
        return

    def set_symprec(self, thresh):
        """
        set the symmetry threshold
        """
        self.symprec = thresh
        return

    def get_symprec(self):
        """
        get the symmetry threshold
        """
        return self.symprec

    def generate_spgcell(self, omit=[]):
        """
        Generate the spglib specific representation of the structure
        (Needs to be called before any other call to spg methods)
         :Parameters:

             - omit : a list of either integers (atomic numbers) or element strings to be omited [optional]
        """
        # convert element symbols into atomic numbers
        if len(omit)>0:
            new_omit = []
            for e in omit:
                if type(e)==str and len(e)==0: continue ###needed?
                if type(e) != type(1):
                    new_omit.append(elems.number[e.lower()])
                else:
                    new_omit.append(e)
            omit = new_omit
        lattice = numpy.array(self.mol.get_cell(), order="C", dtype="double")
        pos     = self.mol.get_frac_xyz()
        pos     = pos%1.0
        # pos     = pos%1.0
        num     = self.mol.get_elems_number()
        pos_rem = []
        num_rem = []
        for i in range(self.mol.natoms):
            if num[i] not in omit:
                pos_rem.append(pos[i])
                num_rem.append(num[i])
        pos_rem = numpy.array(pos_rem, order="C", dtype="double")
        num_rem = numpy.array(num_rem, dtype="intc")
        self.spgcell = (lattice, pos_rem, num_rem)
        spglib.standardize_cell(self.spgcell, no_idealize=True) #is that needed?
        return

    def get_spacegroup(self):
        """
        determine the space group of the current system
        returns a tuple with the symbol and the integer number
        """
        try:
            assert self.spgcell != None
        except:
            self.generate_spgcell()
        result = spglib.get_spacegroup(self.spgcell, symprec=self.symprec)
        result = result.split()
        symbol = result[0]
        number = int(result[1][1:-1])
        if number == 1:
            logger.warning("symmetry detection claims it's P1")
        else:
            logger.info('detected spacegroup %s %i with symprec=%5.4f' % (symbol, number, self.symprec))
        return (symbol, number)

    def equivalent_sites(self, scaled_positions, onduplicates='error',
                         symprec=1e-3):
        """Returns the scaled positions and all their equivalent sites.
		Shamelessly and gratefully copied from ase.spacegroup
        Parameters:

        scaled_positions: list | array
            List of non-equivalent sites given in unit cell coordinates.
        onduplicates : 'keep' | 'replace' | 'warn' | 'error'
            Action if `scaled_positions` contain symmetry-equivalent
            positions:

            'keep'
               ignore additional symmetry-equivalent positions
            'replace'
                replace
            'warn'
                like 'keep', but issue an UserWarning
            'error'
                raises a SpacegroupValueError

        symprec: float
            Minimum "distance" betweed two sites in scaled coordinates
            before they are counted as the same site.

        Returns:

        sites: array
            A NumPy array of equivalent sites.
        kinds: list
            A list of integer indices specifying which input site is
            equivalent to the corresponding returned site.

        Example:

        >>> from ase.spacegroup import Spacegroup
        >>> sg = Spacegroup(225)  # fcc
        >>> sites, kinds = sg.equivalent_sites([[0, 0, 0], [0.5, 0.0, 0.0]])
        >>> sites
        array([[ 0. ,  0. ,  0. ],
               [ 0. ,  0.5,  0.5],
               [ 0.5,  0. ,  0.5],
               [ 0.5,  0.5,  0. ],
               [ 0.5,  0. ,  0. ],
               [ 0. ,  0.5,  0. ],
               [ 0. ,  0. ,  0.5],
               [ 0.5,  0.5,  0.5]])
        >>> kinds
        [0, 0, 0, 0, 1, 1, 1, 1]
        """
        scaled = np.array(scaled_positions, ndmin=2)
        ### EXPERIMENTAL #######################################################
        rots, transs = zip(*self.sg.get_symop())
        rots = np.array(rots, ndmin=3)
        transs = np.array(transs, ndmin=2)
        batchdot = np.einsum('aij,bj->abi', rots, scaled)
        #### TO DEBUG / TO DEMONSTRATE THIS ### -> put elsewhee [RA]
        #allfine = True
        #for i,rot in enumerate(rots):
        #    for j,pos in enumerate(scaled):
        #        standard = np.dot(rot,pos)
        #        experimental = batchdot[i,j]
        #        if not np.all(np.isclose(standard, experimental, atol=symprec)):
        #            print("ERROR: dot product standard[%i,%i] != experimental[%i,%i]:\n\
        #                %s != %s" % (i,j,i,j,standard,experimental))
        #            allfine = False
        #if not allfine:
        #    import pdb; pdb.set_trace()
        #### END DEBUG / DEMONSTRATION #####
        sites = np.mod(batchdot + transs[:,np.newaxis], 1.)
        diff = scaled[np.newaxis,:] - sites
        diff -= (diff + 0.5 + symprec) // 1. + symprec
        mask = np.all(abs(diff) < 2*symprec, axis=-1) # which elements project to 
        where = np.argwhere(mask)
        ########################################################################
        kinds = []
        kinds_all = []
        kinds_table = -np.ones(batchdot.shape[:2], dtype=int)
        sites = []
        for kind, pos in enumerate(scaled):
            for rot, trans in self.sg.get_symop():
                site = np.mod(np.dot(rot, pos) + trans, 1.)
                if not sites:
                    sites.append(site)
                    kinds.append(kind)
                    kinds_all.append(set([kind]))
                    continue
                t = site - sites
                mask = np.all((abs(t) < symprec) |
                              (abs(abs(t) - 1.0) < symprec), axis=1)
                if np.any(mask):
                    inds = np.argwhere(mask)[0]
                    ind = inds[0] # the first
                    if kinds[ind] == kind:
                        pass
                    elif onduplicates == 'keep':
                        pass
                    elif onduplicates == 'replace':
                        kinds[ind] = kind
                    elif onduplicates == 'warn':
                        warnings.warn('scaled_positions %d and %d '
                                      'are equivalent' % (kinds[ind], kind))
                    elif onduplicates == 'error':
                        raise SpacegroupValueError(
                            'scaled_positions %d and %d are equivalent' % (
                                kinds[ind], kind))
                    elif onduplicates == 'return': # new, count as "keep" for kinds
                        kinds_all[ind].add(kind)
                    else:
                        raise SpacegroupValueError(
                            'Argument "onduplicates" must be one of: '
                            '"keep", "replace", "warn" or "error".')
                else:
                    sites.append(site)
                    kinds.append(kind)
                    kinds_all.append(set([kind]))
        self.rotations = rots
        self.translations = transs
        if onduplicates == 'return': # new
            return np.array(sites), kinds, kinds_all
        else:
            return np.array(sites), kinds

    def cif_connectivity(self, new_xyz, atypes):
        def cif_column(column):
            """
            label: labels of atoms involved in the bonds according to unsymmetrized cif file
            sym: symmetries of positions of atoms in the bonds
            op: symmetry operations as per order of symmetry
            pos: image positions according to CIF convention
            abc: image position according to molsys convention
            frac: fractional coordinates of bonding atoms
            rots: rotations of atoms in the bonds
            transs: translations of atoms in the bonds
            atoms: atoms in the bonds
            xyz: coordinates of atoms in the bonds
            diff: difference in coordinates between the new symmetric sites and the seleceted
                bonding atoms
            mask: True if
            """
            label = self.mol.cifdata["_geom_bond_atom_site_label_%i" % (column + 1)]
            sym = self.mol.cifdata["_geom_bond_site_symmetry_%i" % (column + 1)]
            sym_ar = np.array(sym, str)
            sym_split = np.char.split(sym_ar,'_')
            sym_list = sym_split.tolist()
            op, pos = zip(*sym_list)
            op = np.array(op, dtype=int)
            op -= 1 # python convention
            ### symmetry positions 1
            pos = np.array(pos, dtype=int)
            apos = pos//100
            bpos = pos//10  -  10*apos
            cpos = pos      - 100*apos - 10*bpos
            abc = np.vstack([apos, bpos, cpos]).T
            abc -= 5 # cif convention
            abc = abc.astype(float)
            ### general ###
            frac = np.vstack([atypes2frac[k] for k in label])
            rots = self.rotations[op]
            transs = self.translations[op]
            #find allowed bond types
            atoms = [unsym_label.index(i) for i in label]
            xyz = frac_xyz[atoms]
            xyz = np.einsum('aij,aj->ai', rots, xyz) + transs
            diff = xyz[:,np.newaxis] - new_xyz
            diff -= (diff + 0.5) // 1
            mask = np.all(abs(diff) < self.symprec, axis=-1) # which elements project to 
            where = np.argwhere(mask)
            import pdb; pdb.set_trace()
            return where[:,1]
        frac_xyz = self.mol.get_frac_xyz()
        atypes2frac = dict(zip(atypes, frac_xyz))
        unsym_label = self.mol.cifdata['_atom_site_label']
        iwhere = cif_column(0)
        jwhere = cif_column(1)
        import pdb; pdb.set_trace()
        return new_ctab
    
    def cif_connectivity_alternative(self):
        # get made bonds -> extend at all possible bonds with found symmetry.
        raise NotImplementedError
        dx = frac_xyz[jatoms] - frac_xyz[iatoms]
        d = np.linalg.norm(dx,axis=1)
        du = np.unique(d.round(4)) # unique distances (fourth decimal) -> unique bond types
        btypes = []
        for u in du:
            k = np.where(np.isclose(u, d, atol=1e-3))[0][0] # the first
            ik, jk = iatoms[k], jatoms[k]
            iat, jat = elems[ik], elems[jk]
            btypes.append([u]+sorted([iat, jat]))
        #find symmetrized bonds
        new_elems = self.mol.elems[:]
        dx = new_xyz[:,np.newaxis]-new_xyz[np.newaxis,:]
        dx -= np.around(dx)
        d = np.linalg.norm(dx,axis=2)
        for u,iat,jat in btypes:
            dev = abs(d-u)
            bwhere = np.where(np.isclose(dev, 0, atol=1e-4))
            bselect = bwhere[0] < bwhere[1] # prevent double bonds
            bwhere = bwhere[0][bselect].tolist(), bwhere[1][bselect].tolist()
            bonds = zip(*bwhere)
            bt = [iat, jat]
            #check whether the bonds connect the right couple of elements
            #N.B.: reverse list so that last can be removed w/o messing with idxs
            for k in range(len(bonds))[::-1]:
                i,j = bonds[k]
                if sorted([new_elems[i], new_elems[j]]) != bt: #then remove
                    bonds.pop(k)
            new_ctab += bonds

    def make_P1(self, spgnum=-1, sg_setting=1, onduplicates="keep", conn_thresh=0.1, symprec=1e-6):
        """
        to be implemented by Julian from his topo tools

        :Parameters:

            - spgnum : integer space group number

        :KNOWN BUGS:
            - scaled_positions could be equivalent from a cif file, so it fails to make_P1
        """
        if onduplicates == "return":
            raise NotImplementedError("\"return\" mode on duplicates")
        # how to convert international spgnum to hall number
        # apply operations to self.mol and generate a new mol object
        # use detect_conn etc to complete it.
        
        #Okay, what i did was to use ASE as:
        try: 
            from ase.spacegroup import Spacegroup
        except:
            logger.error('make_P1 requires ASE (i.e. ase.lattice.spacegroup) to function properly')
            return
        
        # 1) if provided, use the spacegroup set by the user
        # 2) the spacegroup number is supplied with the cif
        # 3) there is at least the H-M symbol of it, try to find it via the dictionary!
        
        if spgnum == -1:
            try:
                spgnum_cif = int(self.mol.cifdata['_symmetry_int_tables_number'])
                try:
                    spgsym_cif = self.mol.cifdata['_symmetry_space_group_name_H-M'].replace(' ','')
                except:
                    sgs = spacegroups.spacegroups
                    spgsym_cif = str([i for i in sgs.keys() if sgs[i] == spgsym_cif][0])
                logger.info('using spacegroup number from cif file: %i (%s)' % (spgnum_cif,spgsym_cif))
                spgnum=spgnum_cif
            except:
                try:
                    spgsym_cif = self.mol.cifdata['_symmetry_space_group_name_H-M'].replace(' ','')
                    spgnum_cif = spacegroups.get_spacegroup_number(spgsym_cif)
                    if spgnum_cif ==None: 
                        logger.error('spacegroup %s could not be found, add it to spacegroups.py ?!' %spgsym_cif)
                        logger.error('make_P1 failed')
                        return False
                    logger.info('using spacegroup symbol from cif file, sgnumber looked up: %i (%s)' % (spgnum_cif,spgsym_cif))
                    spgnum = spgnum_cif
                except:
                    logger.error('I do not have any spacegroup informations, make_P1 failed!')
                    return False
        
        dataset = spglib.get_symmetry_from_database(spgnum)
        #print(dataset) ### DEBUG
        
        try:
            self.sg = Spacegroup(spgnum,setting=sg_setting)#,sprec = 1e-3)
        except Exception:
            self.sg = Spacegroup(spgnum)#,sprec = 1e-3)
        
        new_xyz = []
        new_elems = []
        new_atypes = []
        new_fragtypes = []
        new_fragnumbers = []
        frac_xyz = self.mol.get_frac_xyz()
        old_atypes = self.mol.atypes[:]
        try:
            if onduplicates == "return":
                new_xyz, kinds, kinds_all = self.equivalent_sites(frac_xyz, symprec=symprec, onduplicates=onduplicates)
                kinds_all = [tuple(k) for k in kinds_all]
                dkinds_ = {tuple(ka):k for ka,k in zip(kinds_all,kinds)}
                dkinds = {v:k for k,v in dkinds_.items()}
                assert len(dkinds) == len(dkinds_)
            else:
                new_xyz, kinds = self.equivalent_sites(frac_xyz, symprec=symprec, onduplicates=onduplicates)
        except Exception as e:
            import sys
            logger.error('could not get equivalent sites, '+str(sys.exc_info()[1]))
            return False
        #now do the new elems and stuff:
        for i,k in enumerate(kinds):
            new_elems.append(self.mol.elems[k])
            new_atypes.append(self.mol.atypes[k])
            new_fragtypes.append(self.mol.fragtypes[k])
            new_fragnumbers.append(0)
        # keep to detect new connectivity
        elems = self.mol.elems[:]
        ctab = self.mol.ctab[:]
        self.mol.set_natoms(len(new_xyz))
        self.mol.set_xyz_from_frac(new_xyz)
        self.mol.set_elems(new_elems)
        self.mol.set_atypes(new_atypes)
        self.mol.set_fragtypes(new_fragtypes)
        self.mol.set_fragnumbers(new_fragnumbers)
        self.mol.set_empty_conn() # for debugging
        # now we try to get the connectivity right and find duplicates during the search
        if onduplicates == "return":
            new_ctab = self.cif_connectivity(new_xyz, old_atypes)
            self.mol.set_ctab(new_ctab, conn_flag=True)
            self.mol.remove_duplicates()
            return True
        else:
            self.mol.detect_conn(thresh = conn_thresh, remove_duplicates = True)
            return True

    def get_primitive_cell(self):
        """
        get the primitve cell as a new mol object
        """
        assert self.spgcell != None
        new_spgcell = spglib.find_primitive(self.spgcell)
        if new_spgcell is None:
            logger.error("Search for primitive cell failed with symprec %f" % self.symprec)
            return
        print(new_spgcell[0])
        print(new_spgcell[2])
        new_mol = molsys.mol()
        new_mol.set_natoms(len(new_spgcell[2]))
        new_mol.set_cell(new_spgcell[0])
        new_mol.set_xyz(new_mol.get_xyz_from_frac(new_spgcell[1]))
        #new_mol.set_xyz_from_frac(new_spgcell[1])
        new_mol.set_elems_number(new_spgcell[2])
        # now add the connectivity
        new_mol.detect_conn()
        # RS: we could do atomtyping ... but this would have to be a method of mol ...
        new_mol.set_atypes(["0"]*new_mol.get_natoms())
        new_mol.set_nofrags()
        return new_mol

    def get_symmetry(self):
        """
        returns lists of rotations, translations and equivalent atoms according to the spgcell
        n.b.: spgcell must be generated with generate_spgcell
        example:
        >>> import molsys
        >>> import numpy as np
        >>> m = molsys.mol()
        >>> m.read(filename)
        >>> m.addon("spg")
        >>> m.spg.generate_spgcell()
        >>> sym = m.spg.get_symmetry()
        >>> n=0 #just an example, n could be any btw. 0 and len(sym)-1
        >>> rota, tran = sym['rotations'][n], sym['translations'][n]
        >>> new_vector = rota*old_vector[:,np.newaxis] + tran
        """
        logger.info("Find space group symmetries")
        sym = spglib.get_symmetry(self.spgcell)
        logger.info("Found %s symmetries and %s equiv. atom/s" % \
            (len(sym['rotations']), len(sym['equivalent_atoms'])))
        return sym['rotations'], sym['translations'], sym['equivalent_atoms']

    def generate_symmetries(self, eps=1e-8):
        """
        Generate list of coordinates by symmetries
        scale (same scale as per supercell) ###TBI: non-orthorombic cells
        """
        logger.info("Get space group symmetries")
        ### INITIALIZE ###
        frac = self.mol.get_frac_xyz()
        self.syms = []
        self.argsyms = []
        ### GENERATE SPACE POINT GROUP SYMMETRIES ###
        self.generate_spgcell()
        self.symops = self.get_symmetry() ###TBI equivalent atoms
        ops = self.symops[:-1] ### REMOVE EQUIVALENT ATOMS
        for i,(rot,tra) in enumerate(zip(*ops)):
            sym = np.tensordot(frac, rot, axes=1)+tra
            ###sym -= sort_by_columns(sym)[0] #np.min(sym,axis=0) ### EXPERIMENTAL ###
            self.syms.append(sym)
            self.argsyms.append(argsort_by_columns(sym))

    # @timer("generate symmetry permutations")
    def generate_symperms(self, thresh=5e-6, eps=1e-8):
        """
        Each symmetry permutation stores the indices that would sort an array
        according to each symmetry operation in the symmetry space group.

        >>> m.addon('spg')
        >>> m.spg.generate_spgcell()
        >>> m.spg.generate_symmetries()
        >>> m.spg.generate_symperms()

        ### OLD LOOP IMPLEMENTATION ###
        >>> for i,isym in enumerate(self.syms):
        >>>     symperm = []
        >>>     for c in isym:
        >>>         frac -= c
        >>>         fracadd = abs(frac)+.5
        >>>         gap = (abs(fracadd-np.floor(fracadd)-.5) < eps)
        >>>         sype = np.where(gap.all(axis=1))[0][0]
        >>>         symperm.append(sype)
        it is slow with respect to the broadcasting
        
        N.B. to avoid that -0. = 0 an older implementation used np.isclose to substitute 0/1 floats to exact 0/1 integers
        """
        if not hasattr(self,""): self.generate_symmetries()
        logger.info("Set %s space group symmetry permutations" % len(self.syms))
        natoms = self.mol.natoms
        tot = len(self.syms)
        sp  = np.empty((0,self.mol.natoms),int)
        sp_ = sp.copy()
        frac = self.mol.get_frac_xyz()
        for i,isym in enumerate(self.syms):
            if (i+1)%100==0 or i+1 == tot:
                if isatty():
                    sys.stdout.write("\r")
                sys.stdout.write("%s of %s symmetry permutations" % (i+1,tot))
                if not isatty():
                    sys.stdout.write("\n")
                elif i+1 == tot:
                    sys.stdout.write("\n")
            isp = get_frac_match(frac, isym, thresh, eps)
            if isp.shape[0] == natoms: 
                ### STANDARD: match position -> match permutation index ###
                sp = np.vstack([sp, isp]) if sp.size else isp
            else:
                ### EXPERIMENTAL, w/ pivot and distances ###
                x = frac[np.newaxis,:]-isym[:,np.newaxis]
                x[np.where(np.isclose(x,.5))] = .5+eps
                x-=np.floor(x+.5)
                for j in range(x.shape[0]):
                    p = 0 #pivot, any
                    w_ = np.where(np.isclose(x, x[j,p]).all(axis=-1))[-1] #[0] is [0,..,N]
                    if w_.shape[0] == natoms:
                        if np.unique(w_).shape[0] == natoms: ### sanity check
                            sp_ = np.vstack([sp_, w_]) if sp_.size else w_
        if sp_.size > 0:
            sp_ = np.vstack({tuple(row) for row in sp_}) if len(sp_.shape) == 2 else sp_
        symperms = np.vstack([sp,sp_])
        self.symperms = symperms
        return symperms

    def find_symmetry(self, xyzref):
        """
        If a match is found, return True. Else, return False.
        """
        logger.info("Seeking symmetry match")
        match = False
        for i,isp in enumerate(self.symperms):
            if np.isclose(self.mol.xyz[isp], xyzref).all():
                match = True
                logger.info("Find symmetry!\nIndex: %d\nPermutation: %s" % (i,isp))
                return i, isp
        if match == False:
            logger.info("No symmetry found")
            raise ValueError("No symmetry found")

    def find_symmetry_from_frac(self, fracref):
        """
        If a match is found, return True. Else, return False.
        N.B.:
        self.find_symmetry_from_colors(sel.colors[m.spg.symperms[i]],symperms=self.symperms) == i
        """
        logger.info("Seeking symmetry match")
        match = False
        frac = self.mol.get_frac_xyz()
        for i,isp in enumerate(self.symperms):
            if np.isclose(frac[isp], fracref).all():
                match = True
                logger.info("Find symmetry!\nIndex: %d\nPermutation: %s" % (i,isp))
                return i, isp
        if match == False:
            logger.info("No symmetry found")
            raise ValueError("No symmetry found")

    def find_symmetry_from_colors(self, colref=None, symperms = None):
        """
        If a match is found, return True. Else, return False.
        """
        if symperms is None: symperms = self.symperms
        if colref is None: colref = self.mol.colors #deprecated [RA]
        logger.info("Seeking symmetry match")
        match = False
        col = self.mol.colors ###???
        for i,isp in enumerate(symperms): ###???
            if np.isclose(col[isp], colref).all():
                match = True
                logger.info("Find symmetry!\nIndex: %d\nPermutation: %s" % (i,isp))
                return i, isp
        if match == False:
            logger.info("No symmetry found")
            raise ValueError("No symmetry found")

    def get_frac_match(frac, sym, thresh=5e-6, eps=1e-8):
        """retrieve equivalent 
        
        [description]
        
        Arguments:
            frac {(N,3) numpy array of floats} -- 
            sym  {(N,3) numpy array of floats} -- 
        
        Keyword Arguments:
            thresh {[type]} -- [description] (default: {5e-6})
            eps {[type]} -- [description] (default: {1e-8})
        
        Returns:
            [type] -- [description]
        """
        symperm = []
        x = frac[np.newaxis,:]-sym[:,np.newaxis]
        whereint = np.where(np.isclose(np.round(x), x, atol=eps))
        x[whereint] = np.round(x[whereint]) + eps
        x -= np.floor(x) ### np.round does not work! [?RA]
        symperm = np.where((x<thresh).all(axis=-1))[-1]
        return symperm

    def generate_symmetry_dataset(self,eps=1e-13):
        """
            Set up the data necessary to exploit symmetry within weaver.
            transformations in the end contains a set of rotations and translations
            for every vertex.

        """
        self.dataset = spglib.get_symmetry_dataset(self.spgcell)
        self.RT = [(r, t) for r, t in zip(self.dataset['rotations'], self.dataset['translations'])]
        self.R = self.dataset['rotations']
        self.T = self.dataset['translations']
        equiv = self.dataset['equivalent_atoms'].tolist()
        equiv_set = list(set(equiv))
        base_indices = [equiv.index(i) for i in equiv_set]
        derived_indices = [[i for i,e in enumerate(equiv) if ((e == j) and (i != -1))] for j in equiv_set]
        
        transformations = {}
        frac = copy.copy(self.spgcell[1])
        for ie,e in enumerate(equiv_set):
            exyz = frac[e]
            M_exyz = (numpy.dot(self.R,exyz)+self.T) #% 1.0
            for id,d in enumerate(derived_indices[ie]):
                dxyz  = frac[d]
                distvects = M_exyz-dxyz
                whereint = np.where(np.isclose(np.round(distvects), distvects, atol=eps))
                distvects[whereint] = np.round(distvects[whereint]) + eps
                distvects -= numpy.floor(distvects)
                
                dists = numpy.linalg.norm(distvects,axis=1)
                idx_dmin  = [i for i,dist in enumerate(dists) if dist < 1e-5][0]
                #idx_dmin = numpy.argmin(dists)
                if dists[idx_dmin] > 1e-8:
                    raise ValueError('no transformation found! for %i ' % (d,))
                #print id,d,numpy.min(dists), numpy.argmin(dists)
                transformations[d] = self.RT[idx_dmin]
        self.transformations = transformations
        self.equiv = equiv
        self.equiv_set = equiv_set
        self.base_indices = base_indices
        self.derived_indices = derived_indices
        
        return transformations
