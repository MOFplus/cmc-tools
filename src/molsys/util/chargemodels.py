"""
    
                chargemodels

    BFJ 2023 : implementation of fluctuating charge models (geometric and topologic)

    used for one-shot or on-the-fly charge (and electrostatic energy/force) generation

"""


import sys
import numpy as np
from collections import OrderedDict
from functools import partial
from molsys.util.constants import *
from molsys.util.chargeinteractions import *
#from scipy.sparse.linalg import spsolve

# define some extra constants here
CONVERT_EV_KCPM = electronvolt / kcalmol


class chargemodel:
    """
    the base class for all fluctuating charge models
    initializes attributes shared by all models and contains methods shared by these
    dummy methods are defined for methods that are shared by all models but are implemented differently
    """

    def __init__(self, mol, dtype, cutoff, q_tot):
        """
        Parameters:
            mol (molsys.mol): mol instance containing the atomistic system
            dtype (str)     : charge distribution type used in the model
            cutoff (float)  : distance cutoff for computation of matrix elements in the child classes
            q_tot (int)     : total charge of the system
        """
        assert dtype in ["gauss", "gauss_wolf", "gauss_dsf", "slater", "point"], f"Unknown charge distribution type {dtype}"
        self.mol = mol
        self.cutoff = cutoff
        self.q_tot = q_tot
        self.dtype = dtype
        self.xyz = self.mol.get_xyz()
        self.atypes = self.mol.get_atypes()
        self.natoms = self.mol.get_natoms()
        self.q = np.zeros([self.natoms], "d")
        self.energy = 0.0
        self.force = np.zeros([self.natoms], "d")
        self.field = np.array([0.0, 0.0, 0.0], "d")
        self.pot_field = np.zeros([self.natoms], "d")
        return

    def set_field(self, field_vect):
        """
        sets the external electric field and computes the consequent potential felt by each atom that needs to be added to the electronegativity vector
        Parameters:
            field_vect (float): vector containing the external electric field in Volt / Angstrom
        """
        self.field = np.array(field_vect)
        self.pot_field = (CONVERT_EV_KCPM * self.field * self.xyz).sum(axis=1)
        return

    def setup_EN(self):
        """
        sets up the electronegativity vector of the molecular system
        """
        return

    def setup_J(self):
        """
        sets up the hardness matrix of the molecular system
        """
        return
    
    def reset_Jii(self):
        """
        convenience method that resets the diagonal elements of the already setup hardness matrix
        """
        return
    
    def setup_q_base(self):
        """
        sets up the base charges of the molecular system (TODO: not yet implemented)
        """
        return

    def setup_SLE(self):
        """
        sets up the system of linear equations of the respective model
        to be called from within solve method
        seperated this from solve method for inheritance reasons
        """
        return

    def solve(self):
        """
        solves the system of linear equations of the respective model and yields the atomic charges
        """
        return
    
    def calc(self):
        """
        convenience method that calls all methods setting up and solving the SLE
        """
        return

    def compute_energy(self):
        """
        computes the electronic energy according to the model
        """
        return self.energy

    def compute_force(self):
        """
        computes the contributions to the forces on each atom due to the model
        """
        return self.force

    def read_parameters(self):
        """
        reads in the parameters needed by the model from one or more external files
        """
        return
    
    def generate_init_qeq_params(self, fill_values=[1.0, 0.0, 0.0, 0.0, 0.0]):
        """
        returns a complete qeq parameter dictionary with all parameters set according to fill_values
        Parameters:
            fill_values (list, opional): list containing values to fill qeq parameters with
        """
        params_qeq = OrderedDict()
        for at in dict.fromkeys(self.atypes):
            params_qeq[at] = [v for v in fill_values]
        return params_qeq


class qeq(chargemodel):
    """
    the standard QEq class using real interatomic distances for the computation of the offdiagonal elements of hardness matrix J
    inherits from base chargemodel class
    makes use of the inter module for reading and storage of parameters and computation of J
    """

    def __init__(self, mol, dtype, cutoff, q_tot, qeq_parfile=None):
        """
        Parameters (new):
            qeq_parfile (str, optional): name of the file containing the qeq parameters
                                         if none is given all qeq parameters are initialized as 0.0,
                                         default is None
        """
        super().__init__(mol, dtype, cutoff, q_tot)
        try:
            # probably not best practice
            global inter
            import inter
        except:
            raise ImportError('ImportError: Impossible to load pair-interaction module ("inter")')
        # set default (zero) parameters
        params = self.generate_init_qeq_params()
        if qeq_parfile != None:
            params_read = self.read_parameters(qeq_parfile)
            # overwrite only needed parameters -> flexibility regarding extra parameters
            for k in params.keys():
                params[k] = params_read[k]
        self.Jij = inter.Jij(self.mol, self.cutoff, 1.0, params, dtype=dtype, outf=sys.stdout)
        self.Jii_self = self.Jij.calc_Jii_self()
        return
    
    def setup_EN(self):
        """
        sets up the electronegativity vector of the molecular system
        """
        self.EN_vec = np.array([self.Jij.params[self.atypes[i]][2] for i in range(self.natoms)], "d")
        self.EN_vec += self.pot_field
        return

    def setup_J(self):
        self.Jij.calc()
        self.J_mat = self.Jij.get_nonsparse()
        # eigval_J = np.linalg.eigvals(self.Jij_mat)
        # self.eigval_J_min = eigval_J.min()
        return
    
    def reset_Jii(self):
        Jii = np.array([self.Jii_self[self.atypes[i]] 
                        + self.Jij.params[self.atypes[i]][1] 
                        for i in range(self.natoms)], "d")
        np.fill_diagonal(self.J_mat, Jii)
        return

    def setup_SLE(self):
        # setup QEq system of linear equations H*s=x
        na = self.natoms
        N = na + 1
        H = np.zeros([N, N], "d")
        x = np.zeros([N], "d")
        # fill H matrix
        H[:na, :na] = self.J_mat[:, :]
        H[na, :na] = -1.0
        H[:na, na] = -1.0
        # fill x vector
        x[:na] = -self.EN_vec[:]
        x[na] = -self.q_tot
        return H, x

    def solve(self):
        # get H and x
        H, x = self.setup_SLE()
        # solve the system of linear equations
        s = np.linalg.solve(H, x)
        # extract charge from solution vector
        self.q = s[:self.natoms]
        return s
    
    def calc(self):
        self.setup_EN()
        self.setup_J()
        return self.solve()

    def compute_energy(self):
        E_EN = np.dot(self.EN_vec, self.q)
        E_J = 0.5 * np.dot(np.dot(self.J_mat, self.q), self.q)
        self.energy = E_EN + E_J
        return self.energy

    def compute_force(self):
        F_J = self.Jij.get_force(self.q)
        self.force = F_J
        return self.force

    def read_parameters(self, qeq_parfile):
        qeq_params = inter.read_Jij_params(qeq_parfile)
        return qeq_params


class qeq_core_valence(qeq):

    """
    a qeq class that computes the J matrix and solves the qeq SLE under the assumption of seperated core (point) and valence (distributed) charges
    inherits from the qeq class
    """

    def __init__(self, mol, dtype, cutoff, q_tot, qeq_parfile=None):
        assert dtype in ["slater", "gauss_dsf"], "Only Slater-type or Gauss-DSF-type valence charge distributions supported for now"  # TODO
        super().__init__(mol, dtype, cutoff, q_tot, qeq_parfile)
        # need to get this guy back into our namespace
        params = self.Jij.params
        cv_type = dtype + "_point"
        self.Jij_cv = inter.Jij(self.mol, self.cutoff, 1.0, params, dtype=cv_type, outf=sys.stdout)
        self.Jii_self_cv = self.Jij_cv.calc_Jii_self()
        self.Jij_core = inter.Jij(self.mol, self.cutoff, 1.0, params, dtype="point", outf=sys.stdout)
        return

    def setup_J(self):
        super().setup_J()
        self.Jij_cv.calc()
        self.J_cv_mat = self.Jij_cv.get_nonsparse()
        self.Jij_core.calc()
        self.J_core_mat = self.Jij_core.get_nonsparse()
        return
    
    def reset_Jii(self):
        return super().reset_Jii()

    def setup_q_core(self):
        """
        sets up the core charge vector of the molecular system
        """
        self.q_core = np.array([self.Jij.params[self.atypes[i]][4] for i in range(self.natoms)])
        return

    def setup_SLE(self):
        H, x = qeq.setup_SLE(self)
        x[:self.natoms] -= np.dot(self.J_cv_mat, self.q_core)
        x[self.natoms] += self.q_core.sum()
        return H, x

    def solve(self):
        s = super().solve()
        self.q += self.q_core
        return s
    
    def calc(self):
        self.setup_q_core()
        return super().calc()

    def compute_energy(self):
        E_EN = np.dot(self.EN_vec, self.q)
        q_val = self.q - self.q_core
        E_J_val = np.dot(np.dot(self.J_mat, q_val), q_val)
        E_J_cv = np.dot(np.dot(self.J_cv_mat, self.q_core), q_val)
        E_J_core = np.dot(np.dot(self.J_core_mat, self.q_core), self.q_core)
        E_J = 0.5 * (E_J_val + 2.0 * E_J_cv + E_J_core)
        self.energy = E_EN + E_J
        return self.energy

    def compute_force(self):
        q_val = self.q - self.q_core
        F_val = self.Jij.get_force(q_val)
        F_cv = self.Jij_cv.get_force(q_val, self.q_core)
        F_core = self.Jij_core.get_force(self.q_core)
        self.force = F_val + F_cv + F_core
        return self.force


class acks2(qeq):

    """
    the standard ACKS2 class
    inherits from qeq class
    makes use of the inter module for reading and storage of parameters and computation of hardness matrix J and non-interacting response matrix X
    an autocomplete feature for incomplete Xij parameter sets is implemented (e.g. have parameters for dimer, want to set up for larger cluster),
    where equivalent atom pairs are detected and parameters are distributed according to the found equivalencies
    makes use of the graph-tools module for detection of these equivalencies
    """

    def __init__(self, mol, dtype, cutoff, q_tot, qeq_parfile=None, inter_cutoff=10000, acks2_parfile=None):
        """
        Parameters (new):
            inter_cutoff (int, optional) : number of consecutive bonds between two atoms starting at which intramolecular non-interacting
                                           response matrix elements are considered intermolecular (exponentially decaying with distance)
                                           default is 10000 (arbitrarily high)
            acks2_parfile (str, optional): name of the file containing the non-interacting response parameters,
                                           if none is given all non-interacting response parameters are initialized as 0.0,
                                           default is None
        """
        super().__init__(mol, dtype, cutoff, q_tot, qeq_parfile)
        try:
            # probably still not best practice
            global grt
            import graph_tool as grt
        except:
            raise ImportError("ImportError: Impossible to load graph-tool")
        # set up the molecular graph
        self.mol.addon("graph")
        self.mol.graph.make_graph(rule=2, hashes=False)
        self.molg = self.mol.graph.molg
        # compute matrix containing number of bonds between each atom pair
        self.compute_nbondmat(inter_cutoff)
        # generate unique atomtypes
        self.utypes = []
        for n, t in enumerate(self.atypes):
            self.utypes.append(t + "%" + str(n+1))
        # get equivalent atom pairs
        self.detect_Xij_equivs()
        # set default (zero) parameters
        params = self.generate_zero_Xij_params()
        if acks2_parfile != None:
            # read parameters from file
            params_read = inter.read_Xij_params(acks2_parfile)
            # NOTE: ad-hoc solution to keep autocomplete situational (replacing missing params with None and checking in the end)
            # other option would be to always autocomplete
            upairs = params.keys()
            upairs_read = params_read.keys()
            for up in upairs:
                if up in upairs_read:
                    params[up] = params_read[up]
                else:
                    params[up] = None
            if None in params.values():
                print("WARNING: Incomplete Xij parameter set! Equalizing missing parameters with existing equivalencies and setting remainder to 0.0.")
                # autocomplete missing parameters
                params = self.autocomplete_Xij_params(params)
        self.Xij = inter.Xij(self.mol, self.cutoff, 1.0, params, outf=sys.stdout)
        self.u = np.zeros([self.natoms], "d")
        return
    
    def reset_Jii(self):
        return super().reset_Jii()

    def setup_X(self):
        """
        sets up the non-interacting response matrix of the molecular system
        """
        self.Xij.calc()
        self.X_mat = self.Xij.get_nonsparse()
        return

    def setup_SLE(self):
        # setup ACKS2 system of linear equations H*s=x
        na = self.natoms
        N = (na + 1) * 2
        H = np.zeros([N, N], "d")
        x = np.zeros([N], "d")
        # fill H matrix
        H[:na, :na] = self.J_mat[:, :]
        H[na, :na] = -1.0
        H[:na, na] = -1.0
        H[:na, na+1:N-1] = -np.eye(na)
        H[na+1:N-1, :na] = -np.eye(na)
        H[na+1:N-1, na+1:N-1] = self.X_mat[:, :]
        H[N-1, na+1:N-1] = -1.0
        H[na+1:N-1, N-1] = -1.0
        # fill x vector
        x[:na] = -self.EN_vec[:]
        x[na] = -self.q_tot
        return H, x

    def solve(self):
        s = super().solve()
        na = self.natoms
        N = (na + 1) * 2
        # extract potential from solution vector
        self.u = s[na+1:N-1]
        return s
    
    def calc(self):
        self.setup_X()
        return super().calc()

    def compute_energy(self):
        super().compute_energy()
        E_u = -np.dot(self.u, self.q)
        E_X = 0.5 * np.dot(np.dot(self.X_mat, self.u), self.u)
        E_ACKS2 = E_u + E_X
        self.energy += E_ACKS2
        return

    def generate_zero_Xij_params(self):
        """
        returns a complete non-interacting response parameter dictionary with all parameters set to 0.0
        """
        params_Xij = OrderedDict()
        for i in range(self.natoms):
            ui = self.utypes[i]
            for j in range(i+1, self.natoms):
                uj = self.utypes[j]
                uij = ui + ":" + uj
                uji = uj + ":" + ui
                params_Xij[uij] = [0.0, 0.0]
                params_Xij[uji] = [0.0, 0.0]
        return params_Xij

    def autocomplete_Xij_params(self, params_Xij):
        """
        autocompletes incomplete non-interacting response parameter sets according to self.Xij_equivs
        fills entirely missing non-interacting response parameters with 0.0
        """
        params_Xij_auto = self.generate_zero_Xij_params()
        upairs = params_Xij.keys()
        for up in upairs:
            pars = params_Xij[up]
            # not sure if this will ever be True, but just in case
            if pars != None:
                params_Xij_auto[up] = pars
        for up in self.Xij_equivs.keys():
            if up in upairs:
                # equalize equivalent parameters
                pars = params_Xij[up]
                for up_equiv in self.Xij_equivs[up]:
                    params_Xij_auto[up_equiv] = pars
        return params_Xij_auto

    def compute_nbondmat(self, inter_cutoff):
        """
        computes the matrix storing the numbers of consecutive bonds along the shortest path between all atom pairs
        """
        nbondmat = grt.topology.shortest_distance(self.molg)
        nbondmat = nbondmat.get_2d_array(range(self.natoms))
        self.nbondmat = np.where(nbondmat >= inter_cutoff, 0, nbondmat)
        return self.nbondmat

    def detect_Xij_equivs(self):
        """
        determines equivalent atom pairs from self.nbondmat to reduce parameter space
        """
        self.Xij_equivs = {}
        upairs = []
        for i in range(self.natoms):
            ui = self.utypes[i]
            for j in range(i+1, self.natoms):
                uj = self.utypes[j]
                uij = ui + ":" + uj
                uji = uj + ":" + ui
                upairs.append(uij)
                upairs.append(uji)
        while len(upairs) > 0:
            upi = upairs.pop(0)
            self.Xij_equivs[upi] = []
            ui_1, ui_2 = upi.split(":")
            ai_1, ni_1 = ui_1.split("%")
            ai_2, ni_2 = ui_2.split("%")
            nbonds_i = self.nbondmat[int(ni_1)-1, int(ni_2)-1]
            for upj in upairs[:]:
                uj_1, uj_2 = upj.split(":")
                aj_1, nj_1 = uj_1.split("%")
                aj_2, nj_2 = uj_2.split("%")
                if [ai_1, ai_2] in [[aj_1, aj_2], [aj_2, aj_1]]:
                    nbonds_j = self.nbondmat[int(nj_1)-1, int(nj_2)-1]
                    if nbonds_i == nbonds_j:
                        self.Xij_equivs[upi].append(upj)
                        upairs.remove(upj)
        return


class topoqeq(chargemodel):

    """
    A QEq class using topological distances for the computation of the offdiagonal elements of J
    inherits from base chargemodel class and borrows some methods from the qeq sibling class
    NOTE: inheriting from the qeq class should be about as code efficient as the current version, it's a matter of choice
    makes use of the graph-tools module for operations on the molecular systems graph representation
    NOTE: many desired functionalities are not yet implemented and will be added over time
    """

    def __init__(self, mol, dtype, cutoff, q_tot, rule=2, from_ff=False, qeq_parfile=None, bl_file=None):
        """
        Parameters (new):
            rule (int, optional)       : sets level of atom type diversification, 1 - full atom type, 2 - hybridization type, 3 - element type
            from_ff (bool, optional)   : (TODO: not yet functional) if set to True, r_0 values are taken from loaded ff addon,
                                         else they are taken from the internal chargeparam database
            qeq_parfile (str, optional): name of the file containing the qeq parameters,
                                         if none is given they are taken from the internal chargeparam database
            bl_file (str, optional)    : name of the file containing the topological bond lengths,
                                         if none is given they are taken from the internal chargeparam database
        """
        super().__init__(mol, dtype, cutoff, q_tot)
        assert rule in [0,1,2], f"Unknown rule {rule} for molecular graph vertices"
        try:
            # probably still not best practice
            global grt
            import graph_tool as grt
        except:
            raise ImportError("ImportError: Impossible to load graph-tool")
        self.rule = rule
        self.from_ff = from_ff
        self.params = OrderedDict()
        # assign calculation methods for the elements of the hardness matrix depending on distribution type
        self.calc_Jij, self.calc_Jii_self = self.assign_calc_methods()
        # set up the molecular graph
        self.mol.addon("graph")
        self.mol.graph.make_graph(rule=self.rule, hashes=False)
        self.molg = self.mol.graph.molg
        # override the atomtypes according to chosen rule
        self.atypes = [atype for atype in self.molg.vp["type"]]
        # load topological bondlengths from file if given
        if bl_file != None:
            self.bondlengths = self.read_bondlengths(bl_file)
        # computing and storing the distance matrix here seems convenient for now
        self.distmat = self.compute_distmat()
        # load qeq parameters either from file or from database
        self.params = self.generate_init_qeq_params()
        if qeq_parfile == None:
            #TEMPORARY HACK: REMOVE DATABASE OPTION
            #from molsys.util.chargeparam import topoqeq_params
            ## TODO: this is an inbetween solution for fitting, should maybe add a from_db flag
            #params = {}
            #for at in dict.fromkeys(self.atypes):
            #    try:
            #        params[at] =  topoqeq_params[rule][at]
            #    except:
            #        pass
            #        #print("WARNING: Missing qeq parameters for atype {}, will be set to default values. Charges will be nonsense!".format(at))
            pass
        else:
            params = self.read_parameters(qeq_parfile)
        for k in self.params.keys():
            try:
                self.params[k] = params[k]
            except:
                pass
        return
    
    def setup_EN(self):
        self.EN_vec = np.array([self.params[self.atypes[i]][2] for i in range(self.natoms)], "d")
        self.EN_vec += self.pot_field
        return

    def setup_J(self):
        """
        new numpyified version which is much faster than nested loop for large systems
        for any distribution type except slater: slighly faster for single methanol (6 atoms)
        """
        self.J_mat = np.zeros((self.natoms, self.natoms))
        sigmas = np.array([self.params[at][0] for at in self.atypes])
        sigmamat_i, sigmamat_j = np.meshgrid(sigmas, sigmas)
        mask = (self.distmat != 0.0) & (self.distmat != np.inf)
        self.J_mat[mask] = self.calc_Jij(sigmamat_i[mask], sigmamat_j[mask], self.distmat[mask])
        self.reset_Jii()
        # NOTE: next step is actually only necessary for slater, where J needs to be symmetrized for cases where id_o12 is not empty
        # to reproduce numerics of legacy/inter version one needs to instead overwrite the [j, i] elements with their [i, j] counterparts
        # will not do that for now
        self.J_mat = 0.5 * (self.J_mat + self.J_mat.T)
        return

    def setup_J_legacy(self):
        """
        old version using nested loop
        """
        self.J_mat = np.zeros((self.natoms, self.natoms))
        for i in range(self.natoms):
            atype_i = self.atypes[i]
            sig_i = self.params[atype_i][0]
            for j in range(i+1, self.natoms):
                atype_j = self.atypes[j]
                sig_j = self.params[atype_j][0]
                r = self.distmat[i, j]
                if r != np.inf:
                    rij = self.calc_Jij(sig_i, sig_j, r)
                    self.J_mat[i, j] = rij
                    self.J_mat[j, i] = rij
            self.J_mat[i, i] = self.params[atype_i][1] + self.calc_Jii_self(sig_i)
        return
    
    def reset_Jii(self):
        sigmas = np.array([self.params[at][0] for at in self.atypes])
        Jii_corr = np.array([self.params[at][1] for at in self.atypes])
        Jii_self = self.calc_Jii_self(sigmas)
        Jii = Jii_self + Jii_corr
        np.fill_diagonal(self.J_mat, Jii)
        return

    def setup_SLE(self):
        """borrowed from qeq"""
        return qeq.setup_SLE(self)

    def solve(self):
        """borrowed from qeq"""
        return qeq.solve(self)
    
    def calc(self):
        """borrowed from qeq"""
        return qeq.calc(self)

    def compute_energy(self):
        """borrowed from qeq"""
        return qeq.compute_energy(self)

    def read_parameters(self, qeq_parfile):
        qeq_params = OrderedDict()
        with open(qeq_parfile, "r") as parfile:
            lines = filter(None, (line.strip() for line in parfile))
            lines = (line.split() for line in lines if not line.startswith("#"))
            for line in lines:
                atype = line[0]
                par = list(map(float, line[1:]))
                qeq_params[atype] = par
        return qeq_params
    
    def read_bondlengths(self, bl_file):
        bondlengths = OrderedDict()
        with open(bl_file, "r") as parfile:
            lines = filter(None, (line.strip() for line in parfile))
            lines = (line.split() for line in lines if not line.startswith("#"))
            for line in lines:
                atype_i = line[0]
                atype_j = line[1]
                atat_ij = (atype_i, atype_j)
                par = float(line[2])
                bondlengths[atat_ij] = par
        return bondlengths

    def compute_bondlengths_from_cov_radii(self):
        from molsys.util.elems import get_covdistance
        bondlengths = {}
        uniques = list(set(self.atypes))
        nunq = len(uniques)
        if self.rule == 0:
            for ni in range(nunq):
                ati = uniques[ni]
                for nj in range(ni, nunq):
                    atj = uniques[nj]
                    dist = get_covdistance((ati, atj))
                    bondlengths[(ati, atj)] = dist
                    bondlengths[(atj, ati)] = dist
        else:
            for ni in range(nunq):
                ati = uniques[ni]
                ei = ati.split("_")[0][:-1]
                for nj in range(ni, nunq):
                    atj = uniques[nj]
                    ej = atj.split("_")[0][:-1]
                    dist = get_covdistance((ei, ej))
                    bondlengths[(ati, atj)] = dist
                    bondlengths[(atj, ati)] = dist
        return bondlengths

    def compute_distmat(self):
        """
        computes the topological distance matrix for an existing molecular graph
        for now the topological bondlengths are taken from the generic database in chargeparam.py
        """
        bondlength = self.molg.new_edge_property("double")
        bondlengths = self.compute_bondlengths_from_cov_radii()
        for edge in self.molg.edges():
            atype_i = self.molg.vp.type[edge.source()]
            atype_j = self.molg.vp.type[edge.target()]
            atat_ij = (atype_i, atype_j)
            atat_ji = (atype_j, atype_i)
            try:
                bondlength[edge] = bondlengths[atat_ij]
            except:
                bondlength[edge] = bondlengths[atat_ji]
        nbondmat = grt.topology.shortest_distance(self.molg)
        nbondmat = nbondmat.get_2d_array(range(self.natoms))
        id_zeros = nbondmat > self.cutoff
        nbondmat[id_zeros] = 0
        #n_zeros = np.count_nonzero(nbondmat==0)
        #print(n_zeros/nbondmat.size)
        #if n_zeros/nbondmat.size  >= 2/3:
        #    self.solvefunc = spsolve
        #    #self.solvefunc = np.linalg.solve
        #else:
        #    self.solvefunc = np.linalg.solve
        self.molg.edge_properties["bondlengths"] = bondlength
        distmat = grt.topology.shortest_distance(self.molg, weights=bondlength)
        # these two steps are necessary to reproduce the outputs of shortest_distance with specific source+target
        distmat = distmat.get_2d_array(range(self.natoms))
        #self.distmat = np.where(distmat > self.cutoff, np.inf, distmat)
        distmat[id_zeros] = np.inf
        self.distmat = distmat
        return self.distmat

    def assign_calc_methods(self):
        """
        assigns the interaction functions imported from chargeinteractions.py depending on chosen distribution type
        """
        if self.dtype == "gauss":
            self.calc_Jij = calc_Jij_gauss
            self.calc_Jii_self = calc_Jii_self_gauss
        elif self.dtype == "gauss_wolf":
            self.calc_Jij = partial(calc_Jij_gauss_wolf, cutoff=self.cutoff)
            self.calc_Jii_self = partial(calc_Jii_self_gauss_wolf, cutoff=self.cutoff)
        elif self.dtype == "gauss_dsf":
            self.calc_Jij = partial(calc_Jij_gauss_dsf, cutoff=self.cutoff)
            self.calc_Jii_self = partial(calc_Jii_self_gauss_dsf, cutoff=self.cutoff)
        elif self.dtype == "slater":
            self.calc_Jij = calc_Jij_slater
            self.calc_Jii_self = calc_Jii_self_slater
        elif self.dtype == "point":
            self.calc_Jij = calc_Jij_point
            self.calc_Jii_self = calc_Jii_self_point
        return self.calc_Jij, self.calc_Jii_self
