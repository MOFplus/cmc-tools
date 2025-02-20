"""

                   lammps potentials

    this module implements two classes:
    - lpot: info on a potential available in lammps with the MOF+ name, the lammps name, the number and types of params etc.
    - lpots: a dictionary type structure of lpot entries

    usage:
    lammps_pots = lpots()
    register_default_lammps_pots(lammps_pots)

    the point is to have clever defaults in lpot so that a minimal set of input is sufficient but keep maximum flexibility
    in addition an instance of lpots is instantiated and filled with entries.
    by exposing this object to the user it is possible to extend the catalog with additional entries before the lammps setup

    RS RUB 2022


    A note on units:
    for added terms pelase check to have units consistent with the existing ones .. they are checked purely as strings!

"""

import numpy as np

ric2lammps = {'bnd' : 'bond_coeff',
              'ang' : 'angle_coeff',
              'dih' : 'dihedral_coeff',
              'oop' : 'improper_coeff'}

def special_format(f):
    fs = ""
    for e in f:
        if e == "f":
            fs += "%12.6f "
        elif e == "i":
            fs += "%5d"
        else:
            pass
    return fs
    

class lpot:

    def __init__(self,
                ric,
                name,
                nparams=None,
                params=None,
                pform=None,
                units=None,
                lammps_name=None,
                range=None,
                bounds=None,
                param_converter=None,
                special_writer=None,
                reverse_mask=None, # index mask to revert the params if the order of atoms is reverted (only bnd, ang, dih)
                is_cross=False,
                cross_parent=None,
                cross_equiv=None, # dicitonary to map equivalent cross params
                ):
        self.ric = ric
        self.name = name
        if params == None:
            assert nparams != None
            self.params = [("p%d" % i) for i in range(nparams)]
            self.nparams = nparams
        else:
            self.params = params
            self.nparams = len(self.params)
        self.pform = pform
        if units == None:
            self.units = [""]* self.nparams
        else:
            assert len(units) == self.nparams
            self.units = units
        if lammps_name == None:
            self.lammps_name = name
        else:
            self.lammps_name = lammps_name
        self.lammps_name = self.lammps_name.replace("_", " ") # currently only one !! underscore is allowed which should be sufficient
        # TODO add or invent range and bounds for the fitting
        self.range = range
        self.bounds = bounds
        # special functions to massage units or compute lammps params from FF params
        self.param_converter = param_converter
        self.special_writer = special_writer
        # mask to juggle the params if the order of atoms is reverted
        self.reverse_mask = reverse_mask
        # flags and info if this is a cross term and needs filling with zeros
        self.is_cross = is_cross
        self.cross_parent = cross_parent
        self.is_cross_parent = False
        self.cross_equiv = cross_equiv
        return

    def get_input(self, i, params):
        """write the lammps input string to be used in lmps.command or written to in file

        Args:
            i (int): number of the potential type for the given ric
            params (list of floats): parameters with len of nparams

        Returns:
            (string) input string       
        """
        assert len(params) == self.nparams, "Problem with %s, params %s" % (self.name, str(params))
        if self.param_converter != None:
            params = self.param_converter(params) # it is possible that the number of params is not nparams any more
        if self.special_writer != None:
            # we use the special write now (here all is possible, this is registered on a per potential basis, so ric and all else is known)
            return self.special_writer(i, params)
        else:
            if self.pform != None:
                frmt = special_format(self.pform)
            else:
                frmt = len(params)*"%12.6f "
            return "%s %5d %s %s" % (ric2lammps[self.ric], i, self.lammps_name, frmt % tuple(params))

        
    def revert_params(self, params):
        """revert the params if the order of atoms is reverted

        Args:
            params (list of floats): parameters with len of nparams

        Returns:
            (list of floats) reverted parameters
        """
        if self.reverse_mask != None:
            return [params[i] for i in self.reverse_mask]
        else:
            return params   

class lpots:

    def __init__(self):
        self.rics = {}
        self.current_ric = None
        return

    def register(self, lp):
        if lp.ric not in self.rics:
            self.rics[lp.ric] = {}
        rpots = self.rics[lp.ric]
        assert lp.name not in rpots
        rpots[lp.name ] = lp
        if lp.is_cross:
            if rpots[lp.cross_parent].is_cross_parent == False:
                rpots[lp.cross_parent].is_cross_parent = []
            rpots[lp.cross_parent].is_cross_parent.append(lp.name) 
        return
        
    def init_ric(self, ric):
        """init registering cross terms and lammps potential tapes for this ric

        after using this you should use get_lammps_in only for this ric, until calling finish_ric 

        Args:
            ric (string): start registering for this ric
        """
        self.current_ric = ric
        self.lammps_pots = []
        self.cross_pot_parents = []
        self.cross_pot = []
        return

    def get_lammps_in(self, ric, pot, i, ptype, params, comments=True):
        assert pot in self.rics[ric] # assume the rics are ok
        lp = self.rics[ric][pot]
        if self.current_ric != None:
            assert ric == self.current_ric
            # get root potential (for cross terms!)
            lpot = lp.lammps_name.split(" ")[0]
            if lpot not in self.lammps_pots:
                self.lammps_pots.append(lpot)
            if lp.is_cross_parent != False:
                self.cross_pot_parents.append((ptype, lp.is_cross_parent, i))
            if lp.is_cross:
                self.cross_pot.append(ptype)
        lammpsin = lp.get_input(i, params)
        if comments:
            lammpsin += "   # %s" % ptype
        return lammpsin

    def finish_ric(self, ric):
        """finsh up for a ric

        Args:
            ric (string): name of ric we are working on (the one used in init_ric_cross)

        Returns:
            a tuple with two sublists
            list1: a list of lammps potential types
            list2: a list of tuples with (pottype, potname, i) to generate dummy cross term entries with zeros to keep lammps happy
                    e.g. ("strbnd", 'strbnd->(c3_c1o2@co2,c3_c3@ph,c3_c2h1@ph)|CuPW', 2) 
        """
        assert self.current_ric == ric
        # now we need to check if there are cross_parents which do not have the corresponding cross terms defined
        # if this is not the case we need to fill up with zeros
        # the return contains all the info fo the necessary zeros to be written
        cross_info = []
        for p in self.cross_pot_parents:
            ptype = p[0].split("->")[1]
            for cp in p[1]:
                cptype = cp + "->" + ptype
                if cptype not in self.cross_pot:
                    cross_info.append((cp, cptype, p[2]))
        self.current_ric = None
        return (self.lammps_pots, cross_info)


#####################################################################################################################################
#                
#             register potentials for lammps
# 
# ###################################################################################################################################
 

mdyn2kcal = 143.88  # this is not very accurate but in order to keep numbers correspond the "legacy" factor is used.
angleunit = 0.02191418
rad2deg = 180.0/np.pi 

def register_defaults(lp):
    """call this function with a lpots object to fill it with the defaults

    Args:
        lp (lpots object): will be filled with default lammps potentials
    """
    #################################################################################################################################
    #
    # bond

    # mm3
    def par_mm3_bnd(p):
        r0 = p[1]
        K2 = p[0]*mdyn2kcal/2.0 
        K3 = K2*(-2.55)
        K4 = K2*(2.55**2.)*(7.0/12.0)
        return (r0, K2, K3, K4)    
    lp.register(lpot("bnd",
                    "mm3",
                    params=("kb", "r0"),
                    units=("mdyne/A", "A"),
                    lammps_name="class2",
                    param_converter=par_mm3_bnd,
    ))


    # class2 (this is identical to quartic but with kcal/mol units .. default lammps setup)
    lp.register(lpot("bnd",
                    "class2",
                    params=("r0", "k2", "k3", "k4"),
                    units=("A", "kcal/(mol*A^2)", "kcal/(mol*A^3)", "kcal/(mol*A^4)"),
    ))


    # quartic
    def par_quartic_bnd(p):
        r0 = p[1]
        K2 = p[0]*mdyn2kcal/2.0 
        K3 = -1*K2*p[2]
        K4 = K2*(2.55**2.)*p[3]
        return (r0, K2, K3, K4)    
    lp.register(lpot("bnd",
                    "quartic",
                    params=("kb", "r0", "kb3", "kb4"),
                    units=("mdyne/A", "A", "1/A", "1/A^2"),
                    lammps_name="class2",
                    param_converter=par_quartic_bnd,
    ))

    # harm 
    lp.register(lpot("bnd",
                    "harm",
                    params=("kb", "r0"),
                    units=("mdyne/A", "A"),
                    lammps_name="harmonic",
                    param_converter=lambda p: (p[0]*mdyn2kcal/2.0, p[1]),
    ))

    # morse 
    lp.register(lpot("bnd",
                    "morse",
                    params=("kb", "r0", "Ed"),
                    units=("mdyne/A", "A", "kcal/mol"),
                    param_converter=lambda p: (p[2], np.sqrt(p[0]*mdyn2kcal/2.0/p[2]), p[1]),  # check alpha!!!
    ))

    #################################################################################################################################
    #
    # angle

    # mm3
    def par_mm3_ang(p):
        th0 = p[1]
        K2  = p[0]*mdyn2kcal/2.0 
        K3 = K2*(-0.014)*rad2deg
        K4 = K2*5.6e-5*rad2deg**2
        K5 = K2*-7.0e-7*rad2deg**3
        K6 = K2*2.2e-8*rad2deg**4
        return (th0, K2, K3, K4, K5, K6)
    lp.register(lpot("ang",
                    "mm3",
                    params=("ka", "a0"),
                    units=("mdyn*A/rad^2", "deg"),
                    lammps_name="class2/p6",
                    param_converter=par_mm3_ang,
    ))

    # mm3 cross strbnd
    #   since strbnd in our current definition (we do not want to break this) needs to generate two angle_coeff lines we 
    #   need a special writer here
    def par_mm3_strbnd(p):
        ksb1, ksb2, kss = np.array(p[:3])*mdyn2kcal
        r01, r02 = p[3:5]
        a0 = p[5]  # looks like we do not need this but it is taken from the parent pot .. could we assert this to be equal?
        return (ksb1, ksb2, kss, r01, r02)
    def write_mm3_strbnd(i, p):
        lmps_string = "angle_coeff %5d class2/p6 bb %12.6f %12.6f %12.6f\n" % (i, p[2], p[3], p[4])
        lmps_string += "angle_coeff %5d class2/p6 ba %12.6f %12.6f %12.6f %12.6f" % (i, p[0], p[1], p[3], p[4])
        return lmps_string
    lp.register(lpot("ang",
                    "strbnd",
                    params=("ksb1", "ksb2", "kss", "r01", "r02", "a0"),
                    units=("mdyn*A/rad^2", "mdyn*A/rad^2", "mdyn/A", "A", "A", "deg"),
                    special_writer=write_mm3_strbnd,
                    lammps_name="class2/p6", # we need this name despite the special writer to collect all the pots
                    param_converter=par_mm3_strbnd,
                    reverse_mask=[1, 0, 2, 4, 3, 5],  # this is the mask to revert the params if the order of atoms is reverted
                    is_cross=True,
                    cross_parent="mm3"
    ))

    # fourier
    lp.register(lpot("ang",
                    "fourier",
                    params=("V", "a0", "n", "fcoul", "fvdw"),  
                    units=("kcal/mol", "deg", "", "", ""),
                    lammps_name="cosine/buck6d",
                    param_converter= lambda p: (0.5*p[0]*angleunit*rad2deg*rad2deg/p[2], p[2], p[1]),
                    pform="fif"
    ))

    # quartic (class2) NOTE we use lammps units and order here intead of old mm3/tinker style from pydlpoly
    lp.register(lpot("ang",
                    "class2",
                    params=("a0", "k2", "k3", "k4"),
                    units=("deg", "kcal/(mol*rad^2)", "kcal/(mol*rad^3)", "kcal/(mol*rad^4)",),
    ))
    lp.register(lpot("ang",
                    "class2_bb",
                    params=("m", "r1", "r2"),
                    units=("kcal/(mol*A^2)", "A", "A"),
                    reverse_mask=[0, 2, 1],
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv = {1: [("ba", 2)], 2: [("ba", 3)]}
    ))
    lp.register(lpot("ang",
                    "class2_ba",
                    params=("n1", "n2", "r1", "r2"),
                    units=("kcal/(mol*A*rad)", "kcal/(mol*A*rad)", "A", "A"),
                    reverse_mask=[1, 0, 3, 2],
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv = {2: [("bb", 1)], 3: [("bb", 2)]}
    ))
    # added by GS: TODO check correctness
    lp.register(lpot("ang",
                    "cosine_periodic",
                    params=("C", "B", "n"),  
                    units=("kcal/mol", "", ""),
                    lammps_name="cosine/periodic",
                    pform="fii"
    ))
    lp.register(lpot("ang",
                    "fourier_uff",
                    params=("K", "C0", "C1", "C2"),  
                    units=("kcal/mol", "", "", ""),
                    lammps_name="fourier",
    ))
    # simple harmonic
        # harm 
    lp.register(lpot("ang",
                    "harm",
                    params=("ka", "a0"),
                    units=("mdyne*A/rad^2", "deg"),
                    lammps_name="harmonic",
                    param_converter=lambda p: (p[0]*mdyn2kcal/2.0, p[1]),
    ))



    #################################################################################################################################
    #
    # dihedral

    # cos3
    lp.register(lpot("dih",
                    "cos3",
                    params=("V1", "V2", "V3"),
                    units=("kcal/mol", "kcal/mol", "kcal/mol"),
                    lammps_name="opls",
                    param_converter=lambda p: (p[0], p[1], p[2], 0.0),
    ))

    # cos4
    lp.register(lpot("dih",
                    "cos4",
                    params=("V1", "V2", "V3", "V4"),
                    units=("kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol"),
                    lammps_name="opls",
    ))

    # class2 from lammps with cross terms 
    lp.register(lpot("dih",
                    "class2",
                    params=("k1", "phi1", "k2", "phi2", "k3", "phi3"),
                    units=("kcal/mol", "deg", "kcal/mol", "deg", "kcal/mol", "deg"),             
    ))
    lp.register(lpot("dih",
                    "class2_mbt",
                    params=("A1", "A2", "A3", "r2"),
                    units=("kcal/(mol*A)", "kcal/(mol*A)", "kcal/(mol*A)", "A"),
                    is_cross=True,
                    cross_parent="class2"
    ))
    lp.register(lpot("dih",
                    "class2_ebt",
                    params=("B1", "B2", "B3", "C1", "C2", "C3", "r1", "r3"),
                    units=("kcal/(mol*A)", "kcal/(mol*A)", "kcal/(mol*A)", "kcal/(mol*A)", "kcal/(mol*A)", "kcal/(mol*A)", "A", "A"),
                    reverse_mask=[3, 4, 5, 0, 1, 2, 7, 6],  # RS this needs verfication
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv={6: [("bb13", 1)], 7: [("bb13", 2)]},
    ))    
    lp.register(lpot("dih",
                    "class2_at",
                    params=("D1", "D2", "D3", "E1", "E2", "E3", "a1", "a2"),
                    units=("kcal/(mol*rad)", "kcal/(mol*rad)", "kcal/(mol*rad)", "kcal/(mol*rad)", "kcal/(mol*rad)", "kcal/(mol*rad)", "deg", "deg"),
                    reverse_mask=[3, 4, 5, 0, 1, 2, 7, 6],  # RS this needs verfication
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv={6: [("aat", 1)], 7: [("aat", 2)]},
    ))
    lp.register(lpot("dih",
                    "class2_aat",
                    params=("M", "a1", "a2"),
                    units=("kcal/(mol*rad^2)", "deg", "deg"),
                    reverse_mask=[0, 2, 1],  # RS this needs verfication
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv={1: [("at", 6)], 2: [("at", 7)]},
    ))
    lp.register(lpot("dih",
                    "class2_bb13",
                    params=("N", "r1", "r3"),
                    units=("kcal/(mol*A^2)", "A", "A"),
                    reverse_mask=[0, 2, 1],  # RS this needs verfication
                    is_cross=True,
                    cross_parent="class2",
                    cross_equiv={1: [("ebt", 6)], 2: [("ebt", 7)]},
    ))
    # added by GS: TODO check correctness
    lp.register(lpot("dih",
                    "harmonic",
                    params=("K", "d", "n"),  
                    units=("kcal/mol", "", ""),
                    lammps_name="harmonic",
                    pform="fii"
    ))





    #################################################################################################################################
    #
    # oop

    # harm 
    lp.register(lpot("oop",
                    "harm",
                    params=("ko", "o0"),
                    units=("mdyne*A/rad^2", "deg"),
                    lammps_name="inversion/harmonic",
                    param_converter=lambda p: (p[0]*mdyn2kcal*1.5, p[1]),
    ))
    # added by GS: TODO check correctness
    lp.register(lpot("oop",
                    "fourier",
                    params=("K", "C0", "C1", "C2", "all"),  
                    units=("kcal/mol", "", "", "", ""),
                    lammps_name="fourier",
                    pform="ffffi"
    ))


    # class2 from lammps with cross term
    lp.register(lpot("oop",
                    "class2",
                    params=("ko", "o0"),
                    units=("kcal/mol", "deg"),             
    ))
    lp.register(lpot("oop",
                    "class2_aa",
                    params=("M1", "M2", "M3", "a1", "a2", "a3"),
                    units=("kcal/mol", "kcal/mol", "kcal/mol", "deg", "deg", "deg"),             
                    is_cross=True,
                    cross_parent="class2"
    ))


    #################################################################################################################################
    #
    # cha    ... these are registered only for yaml output (non bonded are handeled differntly in ff2lammps)

    lp.register(lpot("cha",
                     "gaussian",
                     params=("q", "sigma"),
                     units=("e", "A")
                     ))


   #################################################################################################################################
    #
    # vdw    ... these are registered only for yaml output (non bonded are handeled differntly in ff2lammps)

    lp.register(lpot("vdw",
                     "buck6d",
                     params=("r0", "eps"),
                     units=( "A", "kcal/mol")
                     ))


   #################################################################################################################################
    #
    # vdwpr    ... these are registered only for yaml output (non bonded are handeled differntly in ff2lammps)

    lp.register(lpot("vdwpr",
                     "buck6d",
                     params=("r0", "eps"),
                     units=( "A", "kcal/mol")
                     ))
    
  ####################################################################################################################################
  #
  # reapr
    
    lp.register(lpot("oop", #@KK Initital test hijacks oop instead of creating a new dedicated section
                    "oop_nzn",
                    params=("alpha0", "beta0", "k"        , "d"        , "dx", "a0", "r00"),
                    units= ("rad"   , "rad"  , "kcal/mol" , "kcal/mol" , ""  , ""  , "A" )
                    ))

    lp.register(lpot("reapr",
                    "r_nzn",
                     params=("alpha0", "beta0", "k"        , "d"        , "dx", "a0", "r00"),
                     units= ("rad"   , "rad"  , "kcal/mol" , "kcal/mol" , ""  , ""  , "A" )
                    ))

    lp.register(lpot("reapr",
                    "r_overlap_nzn",
                     params=("dp_0"     , "ds_0"     , "a_p", "a_s", "rp_0", "rs_0"),
                     units= ("kcal/mol" , "kcal/mol" , "1/A"   , "1/A"   , "A"   , "A")
                    ))
    
    lp.register(lpot("reapr",
                    "r_overlap_nzn_dsf",
                     params=("dp_0"     , "ds_0"     , "a_p", "a_s", "rp_0", "rs_0"),
                     units= ("kcal/mol" , "kcal/mol" , "1/A"   , "1/A"   , "A"   , "A")
                    ))
    
    lp.register(lpot("reapr",
                    "r_overlap_nzn_dsf_no_pi",
                     params=("ds_0"     ,"a_s", "rs_0"),
                     units= ("kcal/mol" ,"1/A", "A")
                    ))

    lp.register(lpot("reapr",
                    "r_overlap_nzn_pseudoangle_dsf",
                     params=("dp_0"     , "ds_0"     , "a_p", "a_s", "rp_0", "rs_0"),
                     units= ("kcal/mol" , "kcal/mol" , "1/A"   , "1/A"   , "A"   , "A"),
                     lammps_name="manybody/donor/acceptor/sfg"
                    ))
    
    lp.register(lpot("reapr",
                    "r_overlap_nzn_pseudoangle_dsf_no_pi",
                     params=("ds_0"     , "a_s","rs_0"),
                     units= ("kcal/mol" , "1/A"   , "A")
                    ))
    
    lp.register(lpot("reapr",
                "r_morse",
                    params=("de"     , "a"     , "r0"),
                    units= ("kcal/mol" , "1/A" , "A")
                ))

    lp.register(lpot("reapr",
                    "debug_reapr",
                     params=("dp_0"     , "ds_0"     , "a_p", "a_s", "rp_0", "rs_0"),
                     units= ("kcal/mol" , "kcal/mol" , "1/A"   , "1/A"   , "A"   , "A")
                    ))




#####################################################################################################################################
#                
#            dictionaries with possible pair styles
# 
# ###################################################################################################################################

allowed_pair_styles = {
    "lj/cut" :                      ["cutoff"],
    "lj/mdf" :                      ["cutoff_inner", "cutoff"],
    "coul/long" :                   ["cutoff"],
    "lj/mdf/coul/long" :            ["cutoff_inner", "cutoff", "cutoff_coul"],
    "lj/mdf/coul/dsf" :             ["cutoff_inner", "cutoff", "coul_dampfact", "cutoff_coul"],
    "lj/cut/coul/long" :            ["cutoff", "cutoff"],
    "lj/cut/coul/dsf" :             ["cutoff", "cutoff"],
    "lj/charmmfsw/coul/long" :      ["cut_lj_inner", "cutoff"],   # inner:cuttoff must be cutoff -2.0
    "lj/charmm/coul/gauss/long" :   ["cut_lj_inner", "cut_lj", "coul_smooth", "cut_coul"], 
    "lj/charmm/coul/gauss/dsf" :    ["cut_lj_inner", "cut_lj", "cut_coul"],
    "buck/coul/long" :              ["cutoff"],
    "buck/coul/long/cs" :           ["cutoff"],
    "wangbuck/coul/gauss/long" :    ["vdw_smooth", "coul_smooth", "cutoff"],
    "wangbuck/coul/gauss/long/cs" : ["vdw_smooth", "coul_smooth", "cutoff"],
    "buck6d/coul/gauss/long" :      ["vdw_smooth", "coul_smooth", "cutoff"],
    "buck6d/coul/gauss/dsf" :       ["vdw_smooth", "cutoff", "cutoff_coul"],
    "buck6d/qeq/gauss/dsf" :        ["vdw_smooth", "cutoff", "cutoff_coul"],
    "buck6d/acks2/gauss/dsf" :      ["vdw_smooth", "cutoff", "cutoff_coul"],
    "buck6d/coul/gauss/dsf/cs" :    ["vdw_smooth", "cutoff", "cutoff_coul"],
}
