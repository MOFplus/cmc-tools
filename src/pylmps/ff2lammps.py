# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:37:25 2017

@author: rochus


              ff2lammps
              
class to be instantiated with an exisiting mol object and paramters already assinged
it will write a data and a lamps input file              

"""

import numpy as np
import string
import copy

import molsys
import molsys.util.elems as elements
from molsys.addon import base
from molsys.util.timer import timer, Timer

import logging
logger = logging.getLogger('molsys.ff2lammps')


mdyn2kcal = 143.88
angleunit = 0.02191418
rad2deg = 180.0/np.pi 

from .util import rotate_cell
from . import lammps_pots

class ff2lammps(base):
    
    def __init__(self, mol, setup_FF=True, reax=False, print_timer=True):
        """
        setup system and get parameter 
        
        :Parameters:
        
            - mol: mol object with ff addon and params assigned
            - setup_FF [bool]: defaults to True, skip FF setup when False
            - reax [bool]: defaults to False: if True then ReaxFF is used

        In case of ReaxFF no ff addon is present and no bonds/angles/dihedrals/oops are written
        The atom_style is charge (no mol-ID)

        """
        super(ff2lammps,self).__init__(mol)
        # generate a timer
        self.timer = Timer("ff2lammps")
        self.timer.start()
        self.print_timer = print_timer
        # generate the force field
        if setup_FF != True:
            return
        # general settings                
        self._settings = {}
        # set defaults
        self._settings["cutoff"] = 12.0
        self._settings["cutoff_inner"] = 10.0
        self._settings["cutoff_coul"] = 12.0
        self._settings["parformat"] = "%15.8g"
        self._settings["vdw_a"] = 1.84e5
        self._settings["vdw_b"] = 12.0
        self._settings["vdw_c"] = 2.25
        self._settings["vdw_dampfact"] = 0.25
        self._settings["vdw_smooth"] = 0.9
        self._settings["coul_smooth"] = 0.9
        self._settings["coul_dampfact"] = 0.05
        self._settings["use_angle_cosine_buck6d"] = True
        self._settings["kspace_method"] = "ewald"
        self._settings["kspace_prec"] = 1.0e-6
        self._settings["use_improper_umbrella_harmonic"] = False # default is to use improper_inversion_harmonic
        self._settings["origin"] = "zero"
        self._settings["bcond"] = 0
        self._settings["flowchrg"] = False
        self._settings["delta_mlp"] = None
        self._settings["mlp_pth"] = None
        self._settings["mlp_extra_cmd"] = None
        self._settings['chargegen'] = None
        self._settings['topoqeq_par'] = None
        self._settings["cutoff_reapr"] = 6.0 # @KK, this should eventually be tied to the potential via the .par file
        self._settings["acceptor_elems"] = ["zn"]
        self._settings["donor_elems"] = ["n"]
        # set up lammps_pots
        self.lammps_pots = lammps_pots.lpots()
        lammps_pots.register_defaults(self.lammps_pots)
        # init some basic stuff always needed
        self.ricnames = ["bnd", "ang", "dih", "oop", "cha", "vdw"]
        self.nric = {}
        self.par_types = {}
        self.rics = {}
        for r in self.ricnames:
            self.nric[r] = 0
            self.par_types[r] = {}
            self.rics[r] = {}
        self.reax=False
        if reax:
            self.reax = True
            self.plmps_atypes = list(set(self._mol.get_elems()))
            self.plmps_atypes.sort()
            self.plmps_mass = {}
            for at in self.plmps_atypes:
                self.plmps_mass[at] = elements.mass[at]
            self.timer.stop()
            return
        with self.timer("setup pair pots"):
            self._mol.ff.setup_pair_potentials()
        # set up the molecules
        with self.timer("molecules addon"):
            self._mol.addon("molecules")
            self._mol.molecules()
        # make lists of paramtypes and conenct to mol.ff obejcts as shortcuts
        self.par = {}
        self.parind = {}
        self.npar = {}
        for r in self.ricnames:
            self.par[r]       = self._mol.ff.par[r]
            self.parind[r]    = self._mol.ff.parind[r]
            self.rics[r]      = self._mol.ff.ric_type[r]
            # sort identical parameters (sorted) using a tuple to hash it into the dict par_types : value is a number starting from 1
            par_types = {}
            i = 1
            iric = 0
            with self.timer("par loop %s" % r):
                for pil in self.parind[r]:
                    if pil:
                        pil.sort()
                        tpil = tuple(pil)
                        # we have to check if we have none potentials in the par structure, then we have to remove them
                        if len(tpil) == 1 and self.par[r][tpil[0]][0] == 'none': 
                            continue
                        else:
                            iric += 1
                        if not tpil in par_types:
                            par_types[tpil] = i
                            i += 1
            self.par_types[r] = par_types
            self.npar[r] = i-1
            self.nric[r] = iric
        # map additional nonbonded types
        self.par["vdwpr"] = self._mol.ff.par["vdwpr"]
        self.par["chapr"] = self._mol.ff.par["chapr"]
        self.par["reapr"] = self._mol.ff.par["reapr"]
        # we need to verify that the vdw types and the charge types match because the sigma needs to be in the pair_coeff for lammps
        # thus we build our own atomtypes list combining vdw and cha and use the mol.ff.vdwdata as a source for the combined vdw params
        # but add the combined 1.0/sigma_ij here
        self.plmps_atypes = []
        self.plmps_elems = []
        self.plmps_pair_data = {}
        self.plmps_mass = {} # mass from the element .. even if the vdw and cha type differ it is still the same atom
        for i in range(self._mol.get_natoms()):
            vdwt = self.parind["vdw"][i][0]
            chrt = self.parind["cha"][i][0]
            at = vdwt+"/"+chrt
            if not at in self.plmps_atypes:
                #print("new atomtype %s" % at)
                self.plmps_atypes.append(at)
                #self.plmps_elems.append(self._mol.elems[i].title()+str(len(self.plmps_atypes)))
                self.plmps_elems.append(self._mol.elems[i].title())
                # extract the mass ...
                etup = vdwt.split("->")[1].split("|")[0]
                etup = etup[1:-2]
                e = etup.split("_")[0]
                e = [x for x in e if x.isalpha()]
                #self.plmps_mass[at] = elements.mass[e]
                try:
                    self.plmps_mass[at] = elements.mass[self.plmps_elems[-1].lower()]
                except:
                    self.plmps_mass[at] = 1.0
                #print("with mass %12.6f" % elements.mass[e])
#        for i, ati in enumerate(self.plmps_atypes):
#            for j, atj in enumerate(self.plmps_atypes[i:],i):
#                vdwi, chai = ati.split("/")
#                vdwj, chaj = atj.split("/")
#                vdwpairdata = self._mol.ff.vdwdata[vdwi+":"+vdwj]
#                sigma_i = self.par["cha"][chai][1][1]
#                sigma_j = self.par["cha"][chaj][1][1]
#                # compute sigma_ij
#                sigma_ij = np.sqrt(sigma_i*sigma_i+sigma_j*sigma_j)
#                # vdwpairdata is (pot, [rad, eps])
#                pair_data = []
#                pair_data.append(vdwpairdata)
#                #pair_data = copy.copy(vdwpairdata[1])
#                pair_data.append(1.0/sigma_ij)
#                self.plmps_pair_data[(i+1,j+1)] = pair_data
        # add settings from ff addon
        self.plmps_map_atypes2ids = {}
        for i, at in enumerate(self.plmps_atypes):
            short_at = at[at.find("(")+1:at.find(")")]
            self.plmps_map_atypes2ids[short_at] = i

        for k,v in list(self._mol.ff.settings.items()):
            self._settings[k]=v
        if self._settings["chargetype"]=="gaussian":
            assert self._settings["vdwtype"]=="exp6_damped" or self._settings["vdwtype"]=="wangbuck"
        self.timer.stop()
        return 

    def adjust_cell(self):
        if self._settings["bcond"] > 0:
            fracs = self._mol.get_frac_xyz()
            cell  = self._mol.get_cell()
            self.tilt = 'small'
            # now check if cell is oriented along the (1,0,0) unit vector
            if np.linalg.norm(cell[0]) != cell[0,0]:
                rcell = rotate_cell(cell)
                self._mol.set_cell(rcell, cell_only=False)
            else:
                rcell = cell
            lx,ly,lz,xy,xz,yz = rcell[0,0],rcell[1,1],rcell[2,2],rcell[1,0],rcell[2,0],rcell[2,1]
                # system needs to be rotated
#                rcell=np.zeros([3,3])
#                A = cell[0]
#                B = cell[1]
#                C = cell[2]
#                AcB = np.cross(A,B)
#                uAcB = AcB/np.linalg.norm(AcB)
#                lA = np.linalg.norm(A)
#                uA = A/lA
#                lx = lA
#                xy = np.dot(B,uA)
#                ly = np.linalg.norm(np.cross(uA,B))
#                xz = np.dot(C,uA)
#                yz = np.dot(C,np.cross(uAcB,uA))
#                lz = np.dot(C,uAcB)
                # check for tiltings
            if abs(xy)>lx/2: 
                #logger.warning('xy tilting is too large in respect to lx')
                self.tilt='large'
            if abs(xz)>lx/2: 
                #logger.warning('xz tilting is too large in respect to lx')
                self.tilt='large'
            if abs(yz)>lx/2: 
                #logger.warning('yz tilting is too large in respect to lx')
                self.tilt='large'
            if abs(xz)>ly/2: 
                #logger.warning('xz tilting is too large in respect to ly')
                self.tilt='large'
            if abs(yz)>ly/2:
                #logger.warning('yz tilting is too large in respect to ly')
                self.tilt='large'
            # check if celldiag is positve, else a left hand side basis is formed
            if rcell.diagonal()[0]<0.0: raise IOError('Left hand side coordinate system detected')
            if rcell.diagonal()[1]<0.0: raise IOError('Left hand side coordinate system detected')
            if rcell.diagonal()[2]<0.0: raise IOError('Left hand side coordinate system detected')
#            self._mol.set_cell(rcell, cell_only=False)
#                import pdb; pdb.set_trace()
        return

    @staticmethod
    def cell2tilts(cell):
        return [cell[0,0],cell[1,1],cell[2,2],cell[1,0],cell[2,0],cell[2,1]]


    def setting(self, s, val):
        if not s in self._settings:
            self.pprint("This settings %s is not allowed" % s)
            return
        else:
            self._settings[s] = val
            return
        
    @timer("write data")
    def write_data(self, filename="tmp.data"):
        if self.mpi_rank > 0:
            # only head node writes data 
            return
        self.data_filename = filename
        f = open(filename, "w")
        # write header 
        if self.reax:
            header = "LAMMPS data file for mol object using ReaxFF\n\n"
        else: 
            header = "LAMMPS data file for mol object with MOF-FF params from www.mofplus.org\n\n"
        header += "%10d atoms\n"      % self._mol.get_natoms()
        if self.nric['bnd'] != 0: header += "%10d bonds\n"      % self.nric['bnd']
        if self.nric['ang'] != 0: header += "%10d angles\n"     % self.nric['ang']
        if self.nric['dih'] != 0: header += "%10d dihedrals\n"  % self.nric['dih']
        if self.nric['oop'] != 0:
            if self._settings["use_improper_umbrella_harmonic"] == True:
                header += "%10d impropers\n"  % (self.nric['oop']*3) # need all three permutations
            else:
                header += "%10d impropers\n"  % self.nric['oop']            
        # types are different paramtere types 
        header += "%10d atom types\n"       % len(self.plmps_atypes)
        if len(self.par_types["bnd"]) != 0: header += "%10d bond types\n"       % len(self.par_types["bnd"]) 
        if len(self.par_types["ang"]) != 0: header += "%10d angle types\n"      % len(self.par_types["ang"])
        if len(self.par_types["dih"]) != 0: header += "%10d dihedral types\n"   % len(self.par_types["dih"])
        if len(self.par_types["oop"]) != 0: header += "%10d improper types\n\n" % len(self.par_types["oop"])
        self.adjust_cell()
        xyz = self._mol.get_xyz()
        if (self._settings["bcond"] == 0) and (self._mol.periodic == False):
            # in the nonperiodic case center the molecule in the origin
            # JK: Lammps wants a box also in the non-periodic (free) case.
            # RS: we add this only if we did not already read a cell from the mol object
            if self._settings["origin"] == "zero":
                self._mol.translate(-self._mol.get_com())
                self._settings["origin"] = "center" 
            cmax = xyz.max(axis=0)+10.0
            cmin = xyz.min(axis=0)-10.0
            tilts = (0.0,0.0,0.0)
        elif self._settings["bcond"] < 2:
            # orthorombic/cubic bcond or bcond=0 but a cell exists in the mol object
            cell = self._mol.get_cell()
            if self._settings["origin"] == "zero":
                cmax = cell.diagonal()
                cmin = np.zeros([3])
            else:
                cmax = cell.diagonal()*0.5
                cmin = -cmax
            tilts = (0.0,0.0,0.0)
        else:
            # triclinic bcond
            cell = self._mol.get_cell()
            if self._settings["origin"] == "zero":
                cmin = np.zeros([3])
                cmax = cell.diagonal()
            else:
                cmax = cell.diagonal()*0.5
                cmin = -cmax
            tilts = (cell[1,0], cell[2,0], cell[2,1])
        if self._settings["bcond"] >= 0:
            header += '%12.6f %12.6f  xlo xhi\n' % (cmin[0], cmax[0])
            header += '%12.6f %12.6f  ylo yhi\n' % (cmin[1], cmax[1])
            header += '%12.6f %12.6f  zlo zhi\n' % (cmin[2], cmax[2])
        if self._settings["bcond"] > 2:
            header += '%12.6f %12.6f %12.6f  xy xz yz\n' % tilts
        # NOTE in lammps masses are mapped on atomtypes which indicate vdw interactions (pair potentials)
        #   => we do NOT use the masses set up in the mol object because of this mapping
        #   so we need to extract the element from the vdw paramter name which is a bit clumsy (DONE IN INIT NOW)
        header += "\nMasses\n\n"        
        for i, at in enumerate(self.plmps_atypes):
            header += "%5d %10.4f # %s\n" % (i+1, self.plmps_mass[at], at)
        f.write(header)
        # write Atoms
        # NOTE ... this is MOF-FF and we silently assume that all charge params are Gaussians!!
        f.write("\nAtoms\n\n")
        if self.reax:
            elems = self._mol.get_elems()
            for i in range(self._mol.get_natoms()):
                at = elems[i]
                atype = self.plmps_atypes.index(at)+1
                x,y,z = xyz[i]
                # for reaxff chrg = 0.0 becasue it is set by Qeq
                #   ind  atype chrg x y z # comment
                chrg = 0.0
                f.write("%10d %5d %12.8f %12.6f %12.6f %12.6f\n" % (i+1, atype, chrg, x,y,z))
        else:
            charges = self.get_charges()
            # write atoms with charges
            for i in range(self._mol.get_natoms()):
                vdwt  = self.parind["vdw"][i][0]
                chat  = self.parind["cha"][i][0]
                at = vdwt+"/"+chat
                atype = self.plmps_atypes.index(at)+1
                molnumb = self._mol.molecules.mgroups["molecules"].whichmol[i]+1
                x,y,z = xyz[i]
                chrg  = charges[i]
                #   ind  atype molnumb chrg x y z # comment
                f.write("%10d %5d %5d %12.8f %12.6f %12.6f %12.6f # %s\n" % (i+1, molnumb, atype, chrg, x,y,z, vdwt))
            chargesum = charges.sum()
            if abs(chargesum) > 1e-8:
              self.pprint("The total charge of the system is: %12.8f" % chargesum)
        # write bonds
        if len(self.rics["bnd"]) != 0: f.write("\nBonds\n\n")
        for i in range(len(self.rics["bnd"])):
            bndt = tuple(self.parind["bnd"][i])
            a,b  = self.rics["bnd"][i]
            if bndt in list(self.par_types['bnd'].keys()):
                f.write("%10d %5d %8d %8d  # %s\n" % (i+1, self.par_types["bnd"][bndt], a+1, b+1, bndt))
        # write angles
        if len(self.rics["ang"]) != 0: f.write("\nAngles\n\n")
        for i in range(len(self.rics["ang"])):
            angt = tuple(self.parind["ang"][i])
            a,b,c  = self.rics["ang"][i]
            if angt in list(self.par_types['ang'].keys()):
                f.write("%10d %5d %8d %8d %8d  # %s\n" % (i+1, self.par_types["ang"][angt], a+1, b+1, c+1, angt))
        # write dihedrals
        if len(self.rics["dih"]) != 0: f.write("\nDihedrals\n\n")
        for i in range(len(self.rics["dih"])):
            diht = tuple(self.parind["dih"][i])
            a,b,c,d  = self.rics["dih"][i]
            if diht in list(self.par_types['dih'].keys()):
                f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["dih"][diht], a+1, b+1, c+1, d+1, diht))
        # write impropers/oops
        if len(self.rics["oop"]) != 0: f.write("\nImpropers\n\n")
        for i in range(len(self.rics["oop"])):            
            oopt = tuple(self.parind["oop"][i])
            if oopt:
                a,b,c,d  = self.rics["oop"][i]
                if oopt in list(self.par_types['oop'].keys()):
                    f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, b+1, c+1, d+1, oopt))
                    if self._settings["use_improper_umbrella_harmonic"] == True:
                        # add the other two permutations of the bended atom (abcd : a is central, d is bent)
                        f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, d+1, b+1, c+1, oopt))
                        f.write("%10d %5d %8d %8d %8d %8d # %s\n" % (i+1, self.par_types["oop"][tuple(oopt)], a+1, c+1, d+1, b+1, oopt))
        f.write("\n")
        f.close()
        return

    def parf(self, n):
        pf = self._settings["parformat"]+" "
        return n*pf

    @timer("write 2 internal")
    def write2internal(self, lmps, pair=False, charge=False):
        """rewrite parameters into internal strcutures for fitting

        Args:
            lmps (lmps object): lammps instance to write to
            pair (bool, optional): also rewrite vdw pair potentials. Defaults to False.
            charge (bool, optional): also rewrite charges. Defaults to False.
        """
        if pair:
            pstrings = self.pairterm_formatter()
            for p in pstrings: lmps.lmps.command(p)
        if charge:
            pstrings = self.charge_formatter()
            for p in pstrings: lmps.lmps.command(p)
        for ict in ['bnd','ang','dih','oop']:
            for bt in list(self.par_types[ict].keys()):
                bt_number = self.par_types[ict][bt]
                for ibt in bt:
                    pot_type, params = self.par[ict][ibt]
                    pstring = self.lammps_pots.get_lammps_in(ict, pot_type, bt_number, ibt, params, comments=False)
                    lmps.lmps.commands_string(pstring+"\n")
        return


    # RS (Nov 2020) revision of charges -> add delta charges
    #      works for write_data where not just charges are printed
    #      the follwoing routines are for fitting and only set charges 
    #      some obsolete code is removed
    def charge_formatter(self):
        pstrings = []
        charges = self.get_charges()
        for i in range(self._mol.get_natoms()):
            pstrings.append("set atom %5d charge %12.8f" % (i+1, charges[i]))
        return pstrings

    def get_charges(self):
        if "charge" not in self._mol.loaded_addons:
            self._mol.addon("charge")
        if self._mol.charge.method == None:
            if "chargegen" in self._settings:
                if self._settings["chargegen"] == "topoqeq":
                    assert "topoqeq_par" in self._settings, "topoqeq parameter file is missing"
                    self.pprint("ff2lammps: generating charges from topoqeq using params from %s" % self._settings["topoqeq_par"])
                    charges = self._mol.charge.get_from_topoqeq(qeq_parfile=self._settings["topoqeq_par"])
                elif self._settings["chargegen"] == "ff":
                    charges = self._mol.charge.get_from_ff()
                elif self._settings["chargegen"] == "zero":
                    charges = np.zeros([self._mol.get_natoms()])
                else:
                    print ("WARNING no chargegen defined in par file! Using old default get from ff")
                    charges = self._mol.charge.get_from_ff()
            else:
                # no charges set up, yet .. get from ff (this will be the default)
                charges = self._mol.charge.get_from_ff()
        else:
            charges = self._mol.charge.q
        self.zero_charges = self._mol.charge.zero_charges()
        self.pprint ("ff2lammps: getting charges from %s" % self._mol.charge.method)
        return charges

    def get_charges_old(self):
        charges = np.zeros(self._mol.get_natoms()) # we need to compute the array first and only then we can write it out
        # set up the delta charges dictionary
        delta_chrg = {}
        for k in self.par["chapr"]:
            if self.par["chapr"][k][0] == "delta":
                delta = self.par["chapr"][k][1][0]
                at1, at2 = k.split("(")[1].split(")")[0].split(",")
                delta_chrg[at1] = (at2, delta)
        # compute charges
        conn = self._mol.get_conn()
        for i in range(self._mol.get_natoms()):
            chat  = self.parind["cha"][i][0]
            chrgpar    = self.par["cha"][chat]
            #assert chrgpar[0] == "gaussian", "Only Gaussian type charges supported"   # also "point" should work -> To be checked
            assert chrgpar[0] in ["gaussian","point"], "Only Gaussian type charges supported"   # TODO
            charges[i] += chrgpar[1][0]
            # check if chat in delta_chrg
            chat_red = chat.split("(")[1].split(")")[0]
            if chat_red in delta_chrg:
                at2, delta = delta_chrg[chat_red]
                # chek if any of the bonded atoms is of type at2
                for j in conn[i]:
                    if repr(self._mol.ff.ric.aftypes[j]) == at2:   # Note: aftypes are aftype objects and not strings .. we call repr() to get the string
                        # print ("atom %s connected to atom %s ..apply delta %f" % (chat_red, at2, delta))
                        charges[i] += delta
                        charges[j] -= delta
        if (charges == 0.0).all():
            self.zero_charges = True
        else:
            self.zero_charges = False
        return charges

    def pairterm_formatter(self, hybrid_style = "", comment = False):
        # this method behaves different thant the other formatters because it
        # performs a loop over all pairs
        # recompute pairdata by the combination rules
        #
        # with hybrid_style we can make each pair_coeff part of a hybrid pair_style
        self._mol.ff.setup_pair_potentials()
        pstrings = []
        #TODO recompute pair data before
        for i, ati in enumerate(self.plmps_atypes):
            for j, atj in enumerate(self.plmps_atypes[i:],i):
                # compute the pair data relevant stuff directly here
                vdwi, chai = ati.split("/")
                vdwj, chaj = atj.split("/")
                vdw = self._mol.ff.vdwdata[vdwi+":"+vdwj]
                if self._settings["chargetype"] == "gaussian":
                    sigma_i = self.par["cha"][chai][1][1]
                    sigma_j = self.par["cha"][chaj][1][1]
                    # compute sigma_ij
                    alpha_ij = 1.0/np.sqrt(sigma_i*sigma_i+sigma_j*sigma_j)
                if self._settings["vdwtype"]=="exp6_damped":
                    if vdw[0] == "buck6d":
                        r0, eps = vdw[1]
                        A = self._settings["vdw_a"]*eps
                        B = self._settings["vdw_b"]/r0
                        C = eps*self._settings["vdw_c"]*r0**6
                        D = 6.0*(self._settings["vdw_dampfact"]*r0)**14
                        if "pressure_bath_atype" in self._settings.keys():
                            a1 = ati.count(self._settings["pressure_bath_atype"]) > 0
                            a2 = atj.count(self._settings["pressure_bath_atype"]) > 0
                            if (a1 and not a2) or (not a1 and a2):
                                print ("DEBUG : one is a pressure bath atom %s %s" % (ati, atj))
                                C = 0.0
                                D = 0.0
                        elif "decouple_atypes" in self._settings.keys():
                            # len(self._settings["decouple_atypes"]) > 0: 
                            print (ati, atj)
                            decatyp = self._settings["decouple_atypes"]
                            # decouple all atypes in the set from the rest (this can be one or any)
                            if ((ati in decatyp) + (atj in decatyp)) == 1:
                                A = 0.0
                                C = 0.0
                                D = 0.0
                        #pstrings.append(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s) % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))
                        #f.write(("pair_coeff %5d %5d " + self.parf(5) + "   # %s <--> %s\n") % (i+1,j+1, A, B, C, D, alpha_ij, ati, atj))
                    elif vdw[0] == "buck":
                        A,B,C = vdw[1]
                        D = 0.
                    elif vdw[0] == "buck6de":
                        A,B,C,D = vdw[1]
                    elif vdw[0] == "lbuck":
                        sigma, epsilon, gamma = vdw[1]
                        A = 6*epsilon*np.exp(gamma)/(gamma-6)
                        B = gamma/sigma
                        C = gamma*epsilon*sigma**6/(gamma-6)
                        D = 0.
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(5) + "   # %s <--> %s") % (i+1,j+1, hybrid_style, A, B, C, D, alpha_ij, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(5)) % (i+1,j+1, hybrid_style, A, B, C, D, alpha_ij))                #f.write("pair_style buck/coul/long %10.4f\n\n" % (14.0))
                elif self._settings["vdwtype"] == "wangbuck":
                    if vdw[0]=="wbuck":
                        A,B,C = vdw[1]
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(4) + "   # %s <--> %s") % (i+1,j+1, hybrid_style, A, B, C, alpha_ij, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(4)) % (i+1,j+1, hybrid_style, A, B, C, alpha_ij))
                elif self._settings["vdwtype"]=="buck":
                    if vdw[0] == "buck":
                        A,B,C = vdw[1]
                        B=1./B
                    elif vdw[0] == "lbuck":
                        sigma, epsilon, gamma = vdw[1]
                        A = 6*epsilon*np.exp(gamma)/(gamma-6)
                        B = gamma/sigma
                        C = gamma*epsilon*sigma**6/(gamma-6)
                        D = 0.
                        B = 1./B
                    elif vdw[0] =="mm3":
                        r0, eps = vdw[1]
                        A = self._settings["vdw_a"]*eps
                        B = self._settings["vdw_b"]/r0
                        B = 1./B
                        C = eps*self._settings["vdw_c"]*r0**6
                    else:
                        raise ValueError("unknown pair potential")
                    if comment:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(3) + "   # %s <--> %s") % (i+1,j+1, hybrid_style, A, B, C, ati, atj))
                    else:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(3)) % (i+1,j+1, hybrid_style,  A, B, C))
                elif self._settings["vdwtype"]=="lj_12_6":
                    if vdw[0] == "lj_12_6":
                        #if len(vdw) <= 3:
                        r0 , eps = vdw[1]
                        sig = r0/(2.0**(1.0/6.0))
                        if comment:
                            pstrings.append(("pair_coeff %5d %5d %s " + self.parf(3) + "   # %s <--> %s") % (i+1,j+1, hybrid_style, eps, sig, alpha_ij, ati, atj))
                        else:
                            pstrings.append(("pair_coeff %5d %5d %s " + self.parf(3)) % (i+1,j+1, hybrid_style, eps, sig, alpha_ij))
                    else:
                        pstrings.append(("pair_coeff %5d %5d %s " + self.parf(3)) % (i+1,j+1, hybrid_style, A, B, C))
                elif self._settings["vdwtype"]=="lj":
                    if vdw[0] == "lj":
                        sig, eps = vdw[1]
                        if comment:
                            pstrings.append(("pair_coeff %5d %5d %s " + self.parf(2) + "   # %s <--> %s") % (i+1,j+1, hybrid_style, eps, sig, ati, atj))
                        else:
                            pstrings.append(("pair_coeff %5d %5d %s " + self.parf(2)) % (i+1,j+1, hybrid_style, eps, sig))
                    else:
                        raise ValueError("unknown potential")
                elif self._settings["vdwtype"]=="lj_mdf":
                    if vdw[0] == "lj":
                        sig, eps = vdw[1]
                        cut1 = self._settings["cutoff_inner"]
                        cut2 = self._settings["cutoff"]
                        cut_coul = self._settings["cutoff_coul"]
                        if comment:
                            if hybrid_style != "":
                                pstrings.append(("pair_coeff %5d %5d %s " + self.parf(2) + "   # %s <--> %s") % (i+1,j+1, "lj/mdf", eps, sig, ati, atj))
                                pstrings.append(("pair_coeff %5d %5d %s " + "   # %s <--> %s") % (i+1,j+1, hybrid_style, ati, atj))
                            else:
                                pstrings.append(("pair_coeff %5d %5d " + self.parf(2) + "   # %s <--> %s") % (i+1,j+1, eps, sig, ati, atj))
                        else:
                            if hybrid_style != "":
                                pstrings.append(("pair_coeff %5d %5d %s " + self.parf(2)) % (i+1,j+1, "lj/mdf", eps, sig))
                                pstrings.append(("pair_coeff %5d %5d %s ") % (i+1,j+1, hybrid_style))
                            else:
                                pstrings.append(("pair_coeff %5d %5d " + self.parf(2)) % (i+1,j+1, eps, sig))
                    else:
                        raise ValueError("unknown potential")
                else:
                    raise ValueError("unknown pair setting")
        return pstrings




    @timer("write input")
    def write_input(self, filename = "lmp.input", header=None, footer=None, kspace=False, noheader=False, boundary=None):
        """
        NOTE: add read data ... fix header with periodic info
        """
        assert self.reax != True, "Not to be used with ReaxFF"
        if self.mpi_rank > 0: 
            # only head node writes data
            return
        self.input_filename = filename
        f = open(filename, "w")
        # write standard header
        if not noheader:      
            f.write("clear\n")
            f.write("units real\n")
            # if we use nequip as delta-mlp then we have to set "newton off" already here before the sim box is defined.
            if self._settings["delta_mlp"] == "nequip":
                f.write("newton off\n") 
            # 
            if boundary:
                assert len(boundary)==3
            else:
                if self._settings["bcond"] == 0:
                    boundary = ("f", "f", "f")
                else:
                    boundary = ("p", "p", "p")
            f.write("boundary %s %s %s\n" % tuple(boundary))
            f.write("atom_style full\n")
            if self._settings["bcond"] > 2:
                f.write('box tilt large\n')
            f.write("read_data %s\n\n" % self.data_filename)
            f.write("neighbor 2.0 bin\n\n")
            # extra header
            if header:
                hf = open(header, "r")
                f.write(hf.readlines())
                hf.close()
        f.write("\n# ------------------------ %s FORCE FIELD ------------------------------\n" % self._mol.ff.par.FF)
        ###########################################################################################################################
        #    pair style
        #
        #   TODO ... clean up this entire mess ... what styles do we have? what settings?
        #            if there is more than one make it a hybrid

        # RS 2023 Rev 1: collect pair style settings in string variables and then wite to file.
        pair_style_vdw = ""
        pair_style_coul = ""
        pair_extra = ""
        pair_modify = ""

        # set vdW type
        if self._settings["vdwtype"] in ["buck", "wangbuck", "buck6d"]:
            pair_style_vdw = self._settings["vdwtype"]
        elif self._settings["vdwtype"] == "exp6_damped":
            pair_style_vdw = "buck6d"                    # this is a HACK .. we define the vdw type at differnt points ... this should go
        elif self._settings["vdwtype"] == "lj_12_6":
            pair_style_vdw = "lj/charmm"                    # is this right? what does it mean?
        elif self._settings["vdwtype"] == "lj":
            pair_style_vdw = "lj/cut"                    # is this right? what does it mean?
        elif self._settings["vdwtype"] == "lj_mdf":
            pair_style_vdw = "lj/mdf"                    # this is right. it means nothing. nothing means anything.
        else:
            raise NotImplementedError
        
        if pair_style_coul == "lj/cut" and self._settings["tailcorr"]:
            pair_modify = "pair_modify tail yes"
        
        # set coul type
        if not self.zero_charges:
            if kspace:                
                # use kspace for the long range electrostatics and the corresponding long for the real space pair
                pair_extra = ("kspace_style %s %10.4g" % (self._settings["kspace_method"], self._settings["kspace_prec"]))
                # for DEBUG f.write("kspace_modify gewald 0.265058\n")
                if self._settings["chargetype"] == "gaussian":
                    pair_style_coul = "coul/gauss/long"
                elif self._settings["chargetype"] == "point":
                    pair_style_coul = "coul/long"
            else:
                # use shift damping (dsf)
                if self._settings["chargetype"] == "gaussian":
                    # do we have a floating charge model?
                    if self._settings["flowchrg"]:
                        pair_style_coul = self._settings["flowchrg"]
                    else:
                        pair_style_coul = "coul"
                    pair_style_coul += "/gauss/dsf"
                elif self._settings["chargetype"] == "point":
                    pair_style_coul = "coul/dsf"
            # add coreshell
            if self._settings["coreshell"]:
                pair_style_coul += "/cs"

        # join pair_style
        if pair_style_coul != "":
            pair_style = pair_style_vdw + "/" + pair_style_coul
        else:
            pair_style = pair_style_vdw

        # check if this pair style is allowed (exists in lammps_pots dictionary pair_styles)
        assert pair_style in lammps_pots.allowed_pair_styles
        cutoffs = [("%10.5f" % self._settings[c]) for c in lammps_pots.allowed_pair_styles[pair_style] if self._settings[c] != None]
        cutoffs = " ".join(cutoffs)

        # now check if we have an MLP overlay
        hybrid_flag = False
        hybr_ps = ""
        pair_mlp_cutoff = ""
        pair_mlp_coeff = ""
        pair_reapr_coeff = []
        if self._settings["delta_mlp"]:
            hybr_ps = pair_style
            cap_atypes = [e.capitalize() for e in self.plmps_elems]
            atypes = " ".join(cap_atypes)
            if self._settings["delta_mlp"] in ["nequip", "pace"]:
                # use nequip/pace as additional MLP
                pair_mlp_coeff = "pair_coeff  * * %s %s %s" % (self._settings["delta_mlp"], self._settings["mlp_pth"], atypes)
            elif self._settings["delta_mlp"] == "n2p2":
                raise NotImplementedError
            else:
                raise NotImplementedError
            f.write("\npair_style hybrid/overlay %s %s %s %s" % (pair_style, cutoffs, self._settings["delta_mlp"], pair_mlp_cutoff))
        elif pair_style_vdw == "lj/mdf": #@KK unsure if this merge is fine
            hybr_ps = pair_style_coul
            hparams = [("%10.5f" % self._settings[c]) for c in lammps_pots.allowed_pair_styles[pair_style] if self._settings[c] != None]
            cutoffs_vdw = " ".join(hparams[:2])
            hparams_coul = " ".join(hparams[2:])
            if pair_style_coul != "":
                f.write("\npair_style hybrid/overlay %s %s %s %s\n" % (pair_style_vdw, cutoffs_vdw, pair_style_coul, hparams_coul))
            else:
                f.write("\npair_style %s %s\n" % (pair_style_vdw, cutoffs_vdw))
            hybrid_flag = True
        if self.par["reapr"] != {}:
            for par_descriptor, (pot_name, rea_params) in self.par["reapr"].items():
                lammps_name = self.lammps_pots.rics["reapr"][pot_name].lammps_name
                pair_atom_types:list = par_descriptor[par_descriptor.find("(")+1:par_descriptor.find(")")].split(",")
                pair_atom_types_ids = [self.plmps_map_atypes2ids[name] for name in pair_atom_types]
                sorted_container = sorted(zip(pair_atom_types_ids, pair_atom_types)) # start sort
                pair_atom_types_ids, pair_atom_types = map(list,zip(*sorted_container)) # are now jointly sorted from low id to high id

                first_elem = pair_atom_types[0].split("_")[0][:-1].lower()
                if first_elem in self._settings["donor_elems"]:
                    donor_indicator = "i"
                else:
                    donor_indicator = "j"
                pair_reapr_coeff.append(f"pair_coeff {pair_atom_types_ids[0]+1:5d} {pair_atom_types_ids[1]+1:5d} {lammps_name}     {donor_indicator:5s} {'    '.join(map(str,rea_params))}")

                if hybrid_flag is True:
                    f.write(f"{lammps_name} {self._settings['cutoff_reapr']}")
                else:
                    hybrid_flag = True
                    hybr_ps = pair_style
                    f.write(f"\npair_style hybrid/overlay {pair_style} {cutoffs}  {lammps_name}  {self._settings['cutoff_reapr']}")
                
        if hybrid_flag is True:
            f.write("\n") # end the hybrid/overlay line
        else:
            f.write("\npair_style %s %s\n" % (pair_style, cutoffs))
        
        if len(pair_extra) > 0 and not noheader:
            f.write("\n %s \n\n" % pair_extra)
        if len(pair_modify) > 0:
            f.write("\n %s \n\n" % pair_modify)

        pairstrings = self.pairterm_formatter(comment = True, hybrid_style=hybr_ps)
        pairstrings = pairstrings + [pair_mlp_coeff] + pair_reapr_coeff


        for s in pairstrings: 
            if s == "":
                continue
            f.write((s+"\n"))
        ###########################################################################################################################
        #
        # bond potentials TODO: do we have potentially cross terms here? i guess not
        #
        self.lammps_pots.init_ric("bnd")
        pstrings = []
        for bt in self.par_types["bnd"]:
            bt_number = self.par_types["bnd"][bt]
            for ibt in bt:
                pot_type, params = self.par["bnd"][ibt]
                pstrings.append(self.lammps_pots.get_lammps_in("bnd", pot_type, bt_number, ibt, params))
        styles, cross = self.lammps_pots.finish_ric("bnd")
        # write bond styles
        if len(self.par_types["bnd"]) > 0:
            f.write("\nbond_style hybrid %s\n\n" % " ".join(styles))
        # write all the bond coeffs to the file
        pstrings = "\n".join(pstrings)
        f.write(pstrings + "\n")
        ###########################################################################################################################
        #
        # angle style
        #
        self.lammps_pots.init_ric("ang")
        pstrings = []
        for at in self.par_types["ang"]:
            at_number = self.par_types["ang"][at]
            for iat in at:
                pot_type, params = self.par["ang"][iat]
                pstrings.append(self.lammps_pots.get_lammps_in("ang", pot_type, at_number, iat, params))
        styles, cross = self.lammps_pots.finish_ric("ang") # get info on missing cross terms .. we fill these with zeros
        # write styles 
        if len(self.par_types["ang"]) > 0:
            f.write("\nangle_style hybrid %s\n\n" % " ".join(styles))
        # write all the angle coeffs to the file
        pstrings = "\n".join(pstrings)
        f.write(pstrings + "\n")
        # fill remaining cross terms
        for c in cross:
            # we use the same varable names as above
            pot_type, iat, at_number = c
            params = [0.0] * self.lammps_pots.rics["ang"][pot_type].nparams
            pstring = self.lammps_pots.get_lammps_in("ang", pot_type, at_number, iat, params)
            f.write(pstring + "\n")
        ###########################################################################################################################
        #        
        # dihedral style
        #
        self.lammps_pots.init_ric("dih")
        pstrings = []
        for dt in self.par_types["dih"]:
            dt_number = self.par_types["dih"][dt]
            for idt in dt:
                pot_type, params = self.par["dih"][idt]
                pstrings.append(self.lammps_pots.get_lammps_in("dih", pot_type, dt_number, idt, params))
        styles, cross = self.lammps_pots.finish_ric("dih") # get info on missing cross terms .. we fill these with zeros
        # write styles 
        if len(self.par_types["dih"]) > 0:
            f.write("\ndihedral_style hybrid %s\n\n" % " ".join(styles))
        # write all the angle coeffs to the file
        pstrings = "\n".join(pstrings)
        f.write(pstrings + "\n")
        for c in cross:
            # we use the same varable names as above
            pot_type, idt, dt_number = c
            params = [0.0] * self.lammps_pots.rics["dih"][pot_type].nparams
            pstring = self.lammps_pots.get_lammps_in("dih", pot_type, dt_number, idt, params)
            f.write(pstring + "\n")
        ###########################################################################################################################
        #        
        # improper/oop style
        self.lammps_pots.init_ric("oop")
        pstrings = []
        for it in self.par_types["oop"]:
            it_number = self.par_types["oop"][it]
            for iit in it:
                pot_type, params = self.par["oop"][iit]
                pstrings.append(self.lammps_pots.get_lammps_in("oop", pot_type, it_number, iit, params))
        styles, cross = self.lammps_pots.finish_ric("oop")
        # write styles 
        if len(self.par_types["oop"]) > 0:
            f.write("\nimproper_style hybrid %s\n\n" % " ".join(styles))
        # write all the angle coeffs to the file
        pstrings = "\n".join(pstrings)
        f.write(pstrings + "\n")
        # NOTE : no cross for oops (right?)
        #
        ###########################################################################################################################
        #
        #   special bonds -> TODO check with pots on consistence of special bonds
        #
        f.write("\nspecial_bonds lj %4.2f %4.2f %4.2f coul %4.2f %4.2f %4.2f\n\n" %
            (self._settings["vdw12"],self._settings["vdw13"],self._settings["vdw14"],
            self._settings["coul12"],self._settings["coul13"],self._settings["coul14"]))
        f.write("# ------------------------ %s FORCE FIELD END --------------------------\n" % self._mol.ff.par.FF)
        # write footer
        if footer:
            ff = open(footer, "r")
            f.write(ff.readlines())
            ff.close()
        f.close()
        return

    def report_timer(self):
        if self.mpi_rank == 0:
            if self.print_timer is True:
                self.timer.report()


#### HPC mode hack #####################################

# in the hpc mode we have a mol object and all data only on the head node (which writes to file)
# in order to simplify things the follwoing ff2lammps_stub is just an empty object with the same API as the 
# original, which is doing nothing when setting() or write_data() and write_input() is called. 
# it will give an error on the nodes if wirte_internal is used, since this is used in fitting and should be called
# only when NOT running in hpc_mode

class ff2lammps_stub(base):
    
    def __init__(self, mol, **kwargs):
        return
    
    def setting(self, s, val):
        return
    
    def write_data(self, filename="tmp.data"):
        return
    
    def write_input(self, filename = "lmp.input", **kwargs):
        return
    
    def report_timer(self):
        return
