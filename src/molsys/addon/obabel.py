"""obabel addon

   This addon allows access to various features of the openbabel library for molsys mol objects.
   You must have openbabel V3.X installed and currently only non-periodic molecules are supported

   Current focus of the addon: SMILES and canonical smiles etc
   TBI: FF optimization and conformer search

"""

from openbabel import openbabel as ob
from openbabel import pybel
import numpy as np
from ctypes import *

ob_log_handler = pybel.ob.OBMessageHandler()

# helper class for rings
class ring:

    def __init__(self, pyb_ring, mol):
        self.mol = mol
        self.pyb_ring = pyb_ring
        self.atoms = set([i for i in range(self.mol.natoms) if self.pyb_ring.IsInRing(i+1)])
        self.size = self.pyb_ring.Size()
        # test for aromaticity
        print ("detect arom in ring %s" % str(self.atoms))
        self.arom = self.pyb_ring.IsAromatic()
        print ("openabel says %s" % self.arom)
        if not self.arom:
            # still test atomtypes
            arom = True
            for a in self.atoms:
                at = self.mol.atypes[a].split("_")[0]
                print ("atom %d is %s" % (a, at))
                if not at in ("c3", "n2", "s2", "o2"):
                    arom = False
                    print ("setting arom to False")
            if arom:
                self.arom = True
        print ("Final result %s" % self.arom)
        self.ringsys = None
        return
    
class obabel:

    def __init__(self, mol, loglevel = 0):
        # set log level of global logghandler (this is a global setting!)
        ob_log_handler.SetOutputLevel(0)
        assert mol.periodic == False and mol.bcond == 0
        self._mol = mol
        # generate the pybel object 
        molstring = mol.to_string(ftype="txyz", plain=True)
        self.pybmol = pybel.readstring("txyz", molstring)
        # defaults
        self._smiles = None
        self._cansmiles = None
        self._ff = "uff"
        self.pff = None
        # get atomIDs (differnt from indices)
        self.atomIDs = [a.OBAtom.GetId() for a in self.pybmol.atoms]
        return

    @property
    def ff(self):
        return self._ff

    @ff.setter
    def ff(self, forcefield):
        assert forcefield in pybel.forcefields
        self._ff = forcefield
        return

    @property
    def smiles(self):
        if self._smiles == None:
            self._smiles = self.pybmol.write("smi")[:-2]
        return self._smiles

    @property
    def cansmiles(self):
        if self._cansmiles == None:
            self._cansmiles = self.pybmol.write("can")[:-2]
        return self._cansmiles

    def smiles2filename(self, smi):
        """converts a smiles string to smoething useable as a filename

        we replace
        * -> x
        / -> y
        \ -> z
        # -> ^
        (do we need to replace parantheses?)

        Args:
            smi (string): smiles or canonical smiles
        """
        rep = (
            ("*", "x"),
            ("/", "y"),
            ('\\', 'z'),
            ('#', '^'),
        )
        for r in rep:
            smi = smi.replace(*r)
        return smi

    def get_aromatic(self):
        """get indices of all aromatic ring atoms

        NOTE: openbabel seems not to properly detect all heteraromatic rings ... not clear to me why.
        
        Returns:
            list: atom indices
        """
        aromatic = [a.OBAtom.GetIndex() for a in self.pybmol.atoms if a.OBAtom.IsAromatic()]
        return aromatic

    def get_metal(self):
        """get indices of all metal atoms
        
        Returns:
            list: atom indices
        """
        metal = [a.OBAtom.GetIndex() for a in self.pybmol.atoms if a.OBAtom.IsMetal()]
        return metal

    def get_hybridization(self):
        """get hybridization state of the atoms
        """
        hyb = [a.hyb for a in self.pybmol.atoms]
        return hyb

    def get_obtype(self):
        """get openbabel type of the atoms
        """
        obtyp = [a.type for a in self.pybmol.atoms]
        return obtyp

    def get_bond_order(self):
        obbonds = []
        oborder = []
        for a in self.pybmol.atoms:
            bnds = []
            ordr = []
            for na in ob.OBAtomAtomIter(a.OBAtom):
                bnds.append(na.GetIdx()-1)
                b = a.OBAtom.GetBond(na)
                ordr.append(b.GetBondOrder())
            obbonds.append(bnds)
            oborder.append(ordr)
        return (obbonds, oborder)

    def determine_rings(self):
        self.rings = []
        for r in self.pybmol.OBMol.GetSSSR():
            self.rings.append(ring(r, self._mol))
        # now determine aromatic ringsystems
        links = []
        self.arom_rings = [r for r in self.rings if r.arom]
        for r in self.arom_rings:
            print ("aromatic ring %s" % r.atoms)
        self.arom_ringsys = []
        arng_idx = 0
        largest_j = -1
        for i,r1 in enumerate(self.arom_rings):
            for ii,r2 in enumerate(self.arom_rings[i+1:]):
                j = ii+i+1
                if not r1.atoms.isdisjoint(r2.atoms):
                    # print ("link between %d and %d" % (i,j))
                    if r1.ringsys is None:
                        # this is a new ringsystem
                        new_ringsys = [i, j]
                        r1.ringsys = arng_idx
                        r2.ringsys = arng_idx
                        self.arom_ringsys.append(new_ringsys)
                        arng_idx += 1
                    else:
                        # this is an exisiting ringsystem .. find it
                        for k, rgsys in enumerate(self.arom_ringsys):
                            if i in rgsys:
                                break
                        # add j to rgsys k
                        rgsys.append(j)
                        r2.ringsys = k
                        if j > largest_j:
                            largest_j = j
        # print ("linked arom rings %s" % self.arom_ringsys)
        # now add all remaining non-connected rings to individual ringsystems
        k = largest_j+1
        for i,r in enumerate(self.arom_rings):
            if r.ringsys is None:
                self.arom_ringsys.append([i])
                k += 1
                r.ringsys = k
        print ("all aromatic rings %s" % self.arom_ringsys)
        return

    def get_aromatic_ringsystems(self):
        self.determine_rings()
        rgsys = []
        for s in self.arom_ringsys:
            aidx = set()
            for ri in s:
                aidx |= self.arom_rings[ri].atoms
            rgsys.append(sorted(list(aidx)))
        rgsys_clean = []
        for i in rgsys:
            if i not in rgsys_clean:
                rgsys_clean.append(i)
        rgsys = rgsys_clean
        return rgsys

    def plot_svg(self,fname):
        self.pybmol.write(format="svg",filename=fname,overwrite=True) 
        return
   
    def check_chirality(self):
        # a helper to get atomIds coords:
        def _get_coords_from_ID(atomID):
            idx = self.atomIDs.index(atomID)
            return self._mol.xyz[idx]
        ccenters = {}
        abs_config = {}
        m = self.pybmol.OBMol
        facade = ob.OBStereoFacade(m)
        for iat in range(1,m.NumAtoms()+1):
            id = m.GetAtom(iat).GetId()
            tetstereo = facade.GetTetrahedralStereo(id)
            if tetstereo is not None:
                config = tetstereo.GetConfig()
                ccenters[iat] = [id, config.from_or_towards, config.refs]
                A = _get_coords_from_ID(config.refs[0])
                B = _get_coords_from_ID(config.refs[1])
                C = _get_coords_from_ID(config.refs[2])
                D = _get_coords_from_ID(config.from_or_towards)
                # compute sign of volume of tetrahedron
                M = np.vstack([A - D, B - D, C - D])
                vol = np.linalg.det(M)
                abs_config[iat] = "R" if vol > 0 else "S"
        return ccenters, abs_config
    
    # def calc_CIP_prio(self, idx):
    #     """compute the CIP prios of the tetrahedral neighbors

    #     Args:
    #         idx (int): atom index

    #     Returns:
    #         list of atom idices in order of CIP prio: last is lowest
    #     """
    #     from graph_tool import Graph
    #     from molsys.util import elems
    #     g = Graph(directed=False)
    #     g.add_vertex(self._mol.natoms)
    #     g.add_edge_list(self._mol.ctab)
    #     g.vp.anum = g.new_vertex_property("int")
    #     for v in g.vertices():
    #         g.vp.anum[v] = elems.number[self._mol.elems[int(v)]]
    #     nei1 = [int(n) for n in g.get_out_neighbors(idx)]
    #     lev1 = [g.vp.anum[n] for n in nei1]
    #     prio1 = np.argsort(lev1)
    #     # check if we need a second level
    #     skip = []
    #     for an in lev1:
    #         if an not in skip and lev1.count(an) > 1:
    #             print ("secondlevel for atom number %d" % an)
    #             # need second level
    #             dups = [i for i,v in enumerate(lev1) if v == an]
    #             skip.append(an)
    #             nei2 = [nei1[i] for i in dups]
    #             lev2 = []
    #             for j in nei2:
    #                 lev2.append([g.vp.anum[n] for n in g.get_out_neighbors(j) if n != idx])
    #             print (dups, nei2, lev2)

    #     print (nei1, lev1, prio1)

            


    ################ force field stuff ##############################

    def get_obmol_coords(self):
        """ Get OBMol coords and put them into the parent mol object

        Needs to be called after pybel or openbabel operations that chenged the coords
        """
        coords = []
        for a in ob.OBMolAtomIter(self.pybmol.OBMol):
            coords.append([a.GetX(), a.GetY(), a.GetZ()])
        coords = np.array(coords)
        self._mol.set_xyz(coords)
        return

    # this is a black box in pybel using steepest descent  ... fixed iterations
    # advantage: no need to setup ff
    def localopt(self, steps=500):
        self.pybmol.localopt(forcefield=self.ff, steps=steps)
        self.get_obmol_coords()
        return

    def setup_ff(self):
        from openbabel import OBForceField
        self.pff = OBForceField.FindForceField(self.ff)
        assert (self.pff.Setup(self.pybmol.OBMol))
        self.pff.SetLogLevel(ob.OBFF_LOGLVL_LOW)
        self.pff.SetLogToStdOut()
        return

    def get_ff_energy(self, all_params=False):
        assert self.pff
        if all_params:
            self.pff.SetLogLevel(ob.OBFF_LOGLVL_HIGH)
        e = self.pff.Energy(True)
        if all_params:
            self.pff.SetLogLevel(ob.OBFF_LOGLVL_LOW)        
        return e

    def get_ff_gradient(self):
        """ get obabel ff gradient
        """
        gradp = self.pff.GetGradientPtr()
        # Set number of elements to receive
        nelem = self._mol.natoms*3
        # int returns the actual address in memory
        p = (c_double * nelem).from_address(int(gradp))
        grad = np.array(p).reshape(self._mol.natoms,3)
        return grad

    def ff_opt(self, steps=None, steps_per_atom=10, maxsteps=10000, gconv=0.005):
        if steps is None:
            steps = steps_per_atom*self._mol.natoms
        self.pff.SteepestDescent(steps, 1.0e-10)
        energy = self.get_ff_energy()
        grad = self.get_ff_gradient()
        rmsgrad = np.sqrt(np.linalg.norm(grad))
        print ("obabel ff_opt  energy %15.5f rmsgrad %12.6f" % (energy, rmsgrad))
        while rmsgrad > gconv:
            self.pff.SteepestDescent(steps, 1.0e-10)
            energy = self.get_ff_energy()
            grad = self.get_ff_gradient()
            rmsgrad = np.sqrt(np.linalg.norm(grad))/(3*self._mol.natoms)
            print ("obabel ff_opt  energy %15.5f rmsgrad %12.6f" % (energy, rmsgrad))
        print ("Converged!")
        # get coords from ff object back into OBMol
        self.pff.GetCoordinates(self.pybmol.OBMol)
        # now get OBMol coords back into mol object
        self.get_obmol_coords()
        return energy, rmsgrad
        
    def get_ff_hessian(self, delta=0.001):
        hess = []
        for a in ob.OBMolAtomIter(self.pybmol.OBMol):
            print ("computing atom %d" % a.GetIndex())
            keep = [a.GetX(), a.GetY(), a.GetZ()]
            # X
            a.SetVector(keep[0]+delta, keep[1], keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_xp = self.get_ff_energy()
            grad_xp = self.get_ff_gradient()
            a.SetVector(keep[0]-delta, keep[1], keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_xm = self.get_ff_energy()
            grad_xm = self.get_ff_gradient()
            a.SetVector(keep[0], keep[1], keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            hx = (grad_xm.ravel()-grad_xp.ravel())/(2.0*delta)
            hess.append(hx)
            # Y
            a.SetVector(keep[0], keep[1]+delta, keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_yp = self.get_ff_energy()
            grad_yp = self.get_ff_gradient()
            a.SetVector(keep[0], keep[1]-delta, keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_ym = self.get_ff_energy()
            grad_ym = self.get_ff_gradient()
            a.SetVector(keep[0], keep[1], keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            hy = (grad_ym.ravel()-grad_yp.ravel())/(2.0*delta)
            hess.append(hy)
            # Z
            a.SetVector(keep[0], keep[1], keep[2]+delta)
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_zp = self.get_ff_energy()
            grad_zp = self.get_ff_gradient()
            a.SetVector(keep[0], keep[1], keep[2]-delta)
            self.pff.SetCoordinates(self.pybmol.OBMol)
            energy_zm = self.get_ff_energy()
            grad_zm = self.get_ff_gradient()
            a.SetVector(keep[0], keep[1], keep[2])
            self.pff.SetCoordinates(self.pybmol.OBMol)
            hz = (grad_zm.ravel()-grad_zp.ravel())/(2.0*delta)
            hess.append(hz)
        hess = np.array(hess)
        # measure symmetry
        dev = np.linalg.norm(hess - hess.T)
        # make symmetric
        hess = (hess + hess.T)/2.0
        return hess






