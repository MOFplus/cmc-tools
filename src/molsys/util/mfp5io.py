"""

                            mfp5io

    implements a hdf5/h5py based class to handle storage of 
    molecular systems (used for restart and trajectory data)

    This is a rewrite of the V1.0 that ships with molsys and is based on a 
    molsys type mol object. It is meant to work with any MM engine that provides
    a molsys mol object and provides a certain API. It can be used with pydlpoly
    (using the new startup mode with molsys) or pylmps currently. Note that in pylmps 
    trajectory info is written in C++ in lammps using dump_mfp5.cpp (which needs to be
    compiled in from the package PYLMPS)

    one part of the data is "fixed"
    - number of atoms
    - elements
    - atom types
    - connectivity
    - boundary conditions
    - forcefield parameters
    in a restart this data is used as a replacement to the usual file input

    the other part is "staged"
    in other words if needed multiple stages can be defined like "equilibration",
    "sample1", "sample2" etc.
    the initial and default stage is called "default" by default ... yes!
    for each stage current restart information (position/velocitites/cell) is
    automatically stored every N steps. restart data is stored in double precision
    in addition, trajectory info is stored every M steps. (both N and M can be changed)
    trajectory data (pos or pos/vel or pos/vel/force) can be stored both in single and
    double precision on request.
    
    Versions (in file attribute "version"
    - no version info == 1.0 : original
    - version 1.1            : nstep of trajectories is stored per dataset and not for the stage
    - version 1.2            : mfp5io version to work with new mol objects (can write par info)
    
This has been renamed in 2021 from pdlpio2 to mfp5io (according with the name change in lammps)
"""

from h5py import File, special_dtype
import numpy as np

import os

from molsys import mpiobject
import molsys


class mfp5io(mpiobject):

    def __init__(self, fname, ffe=None, restart=None, filemode="a", mpi_comm=None, out=None, hpc_mode=False):
        """generates a mfp5 file object

        This object knows three states: conencted to a ffe (force field engine) (with a mol object),
        connected to a ffe but without mol object, which implies a restart (will generate mol from mfp5 file)
        or the analysis mode. In the analysis mode writing to trajectory data is forbidden.
        In the connected mode one can request appending to a stage. 
        Otherwise a new stage is required.

        NOTE: always use self.open() as a first command in a method that will access the file. it will check if the file
              is already open and only open if it is not open. in certain cases (e.g. dumping with lammps) the file needs to be 
              closed before passing to lammps. To make sure the file is connected call open().
        
        Args:
            fname (string): filename (including .mfp5 extension)
            ffe (object, optional): Defaults to None. Force Field engine (must contain a ffe.mol instance of type molsys)
            restart (string, optional): Defaults to None. Name of the stage from which a restart should be read
            hpc_mode (bool, optional): Defaults to False, if True this means in the ffe only the master has a mol object
        """ 
        super(mfp5io, self).__init__(mpi_comm,out)
        self.verbose = 0
        self.filemode = filemode
        # helper object for hdf5 variable length strings
        self.str_dt = special_dtype(vlen=str)
        #
        self.fname = fname
        # check if this file exists or should be generated
        if self.is_master:
            self.fexists = os.path.isfile(self.fname)
        else:
            self.fexists = False
        self.fexists = self.mpi_comm.bcast(self.fexists)
        self.h5file_isopen = False
        # check if we run with an FFE or not
        self.ffe  = ffe
        self.hpc_mode = hpc_mode
        # in hpc mode use function without communication
        if self.hpc_mode:
            self.get_mol_from_system = self.get_mol_from_system_hpc
        if ffe is not None:
            self.ffe_name = type(ffe).__name__
            if hpc_mode:
                setup_failed = False # we need to do all chekcs on the headnode and communicate
                if restart is None:
                    if self.is_master:
                        # a force field engine is connected => check if mol object is available and of the right type
                        try:
                            self.mol = ffe.mol
                        except AttributeError:
                            setup_failed = "MFP5 ERROR: Force Field Engine has no mol object!"
                        try:
                            is_topo = self.mol.is_topo
                        except AttributeError:
                            setup_filed = "MFP5 ERROR: mol object seems not be a molsys object! This is required for mfp5io"
                        # last but not least check that mol is not a topo ... we do not want that
                        if is_topo is True:
                            setup_field = "MFP5 ERROR: mol object is a topo object .. that should not happen!"
                    # now bcast result and respond on all nodes
                    setup_failed = self.mpi_comm.bcast(setup_failed)
                    if setup_failed == False:
                        self.mode = "ffe"
                        # now all is fine .. set defaults for this runmode
                        self.stage = "default" # default stage which can be overwritten (contains no trajectory data, only restarts)
                    else:
                        raise Exception(setup_failed)
                else:
                    # this is a restart, which means the file must exist and also the stage name
                    assert self.fexists, "MFP5 ERROR: For restart the file must exist. Check filename and path!"
                    self.restart_stage = restart
                    self.mode = "restart"
            else:
                # regular non-hpc mode ... keep all as it was
                if restart is None:
                    # a force field engine is connected => check if mol object is available and of the right type
                    try:
                        self.mol = ffe.mol
                    except AttributeError:
                        self.pprint("MFP5 ERROR: Force Field Engine has no mol object!")
                        raise
                    # ok, now let us check if mol is actually a molsys object (require it to have a is_topo attribute)
                    try:
                        is_topo = self.mol.is_topo
                    except AttributeError:
                        self.pprint("MFP5 ERROR: mol object seems not be a molsys object! This is required for mfp5io")
                        raise
                    # last but not least check that mol is not a topo ... we do not want that
                    if is_topo is True:
                        self.pprint("MFP5 ERROR: mol object is a topo object .. that should not happen!")
                        raise Exception("mfp5 error")
                    self.mode = "ffe"
                    # now all is fine .. set defaults for this runmode
                    self.stage = "default" # default stage which can be overwritten (contains no trajectory data, only restarts)
                else:
                    # this is a restart, which means the file must exist and also the stage name
                    assert self.fexists, "MFP5 ERROR: For restart the file must exist. Check filename and path!"
                    self.restart_stage = restart
                    self.mode = "restart"
        else:
            # if ffe is None we are in analysis mode (means the file must exist)
            assert self.fexists, "MFP5 ERROR: For analysis the file must exist!"
            self.mode = "analysis"
            if restart is not None:
                self.restart_stage = restart
            else:
                self.restart_stage = "default" # use default stage to start up (should be always there, right?)
        # now open the file
        self.open()
        #
        self.file_version = 1.2
        if self.fexists is True:
            if self.mode == "ffe" or self.mode == "restart":
                # this is an exisiting file and we are in ffe/restart mode and want to add data => make sure all is matching
                if self.is_master:
                    file_version = self.h5file.attrs["version"]
                else:
                    file_version = None
                file_version = self.mpi_comm.bcast(file_version)  
                assert self.file_version == file_version, "MFP5 ERROR: Exisiting file has a different version! Can not add data"
                if self.mode == "ffe":
                    # check the system if it contains the same molecule (we could be more clever here, but if the list of atomtypes is identical we should be safe)
                    # TBI mor consistent tests like paramters etc .. we could read the complete mol out and compare
                    self.compare_system()
        else:
            # file is new .. set version
            if self.is_master:
                self.h5file.attrs["version"] = self.file_version
            # this must be ffe mode so we can start generating the system group
            self.write_system()
            # new file so we write the initital structure to the restart
            self.add_stage(self.stage)
            self.write_restart()
        return
        
    # def __del__(self):
    #     self.close()
    #     return

    def open(self):
        """ method to make sure that h5file is open """
        if self.h5file_isopen:
            return
        self.h5file_isopen = True
        if self.is_master:
            self.h5file = File(self.fname, self.filemode)
        else:
            self.h5file = None
        return

    def close(self):
        """ method to make sure that the file is closed """
        if not self.h5file_isopen:
            return
        self.h5file_isopen = False
        if self.is_master:
            self.h5file.close()
        return         

    # system IO
    #
    # The system group contains all the time independent information of the system (connectivity, atomtypes, parameter)
    # Note: in the old pdlpio system this data was prepared by the mol (assign_FF) class and just written here
    #       Now in mfpfio we expect a certain interface, namely the mol object from molsys and collect all the 
    #       data in an active way
    #
    # RS 2025  added fixed charges
    #       he legacy method was to generate charges from the FF which is in the system group. It was cheap to
    #       regenerate and so no need to store. 
    #       Now we generate charges by topoqeq (they are differnt for all atoms even if atype is the same)
    #       this takes a while and we prefer to store them
    #       This is different from any fluctuating charge model where q needs to be stored in the trajectory info
    #       TODO do we need charges in restart??? only if charge is a dynamic variable (extended lagrangian methods)

    def write_system(self):
        """ Writes all the info from a mol object to the system group in the mfp5 file
        
        The following info is written:
            - elems       (strings)
            - atypes      (strings)
            - fragtypes   (strings)
            - fragnumbers (int)
            - bcd         (int)
        
        Further info is stored in subgroups for different addons
            - ff addon (parameter and rics)
            - molecules addon (TBI)
            - charges from charge addon IF these are made by topoqeq (including metadata)

        """
        self.open()
        if self.is_master:
            system = self.h5file.require_group("system")
            # elems
            na = self.ffe.mol.get_natoms()
            elems = self.ffe.mol.get_elems()
            mfp5_elems = system.require_dataset("elems", shape=(na,), dtype=self.str_dt)
            mfp5_elems[...] = elems
            # atypes
            atypes = self.ffe.mol.get_atypes()
            mfp5_atypes = system.require_dataset("atypes", shape=(na,), dtype=self.str_dt)
            mfp5_atypes[...] = atypes
            # fragtypes
            fragtypes = self.ffe.mol.get_fragtypes()
            mfp5_fragtypes = system.require_dataset("fragtypes", shape=(na,), dtype=self.str_dt)
            mfp5_fragtypes[...] = fragtypes
            # fragnumbers
            fragnumbers = self.ffe.mol.get_fragnumbers()
            mfp5_fragnumbers = system.require_dataset("fragnumbers", shape=(na,), dtype="i")
            mfp5_fragnumbers[...] = fragnumbers
            # get connectivity table
            cnc_table = self.ffe.mol.get_ctab()
            if len(cnc_table) > 0:
                # catch if there are no bonds at all: h5py does not like zero size selections
                cnc_table = np.array(cnc_table, dtype="i")
                mfp5_cnc_table = system.require_dataset("cnc_table", shape=cnc_table.shape, dtype=cnc_table.dtype)
                mfp5_cnc_table[...] = cnc_table
            system.attrs["bcd"] = self.ffe.mol.get_bcond()
            if "ff" in self.ffe.mol.loaded_addons:
                # a force field is loaded: we assume it is also initialized. TBI: a switch in the ff addon to verify this
                data = self.ffe.mol.ff.pack()
                ff = system.require_group("ff")
                ric = ff.require_group("ric")
                par = ff.require_group("par")
                if data["FF"] is not None:
                    par.attrs["FF"] = data["FF"]
                else:
                    par.attrs["FF"] = ""
                for r in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
                    # write ric arrays to mfp5 file
                    if r in data:
                        d = data[r]
                        fd = ric.require_dataset(r, shape=d.shape, dtype="i")
                        fd[...] = d
                    # write par data to mfp5 file
                    #  order is ptype, names, npars, pars
                    if r+"_par" in data:
                        p = data[r+"_par"]
                        fp = par.require_group(r)
                        n = len(p[0]) # number of paramters for this ric type
                        fptypes = fp.require_dataset("ptypes", shape=(n,), dtype=self.str_dt)
                        fptypes[...] = p[0]
                        fnames = fp.require_dataset("names", shape=(n,), dtype=self.str_dt)
                        fnames[...] = p[1]
                        fnpars = fp.require_dataset("npars", shape=p[2].shape, dtype="i")
                        fnpars[...] = p[2]
                        fpars = fp.require_dataset("pars", shape=p[3].shape, dtype="float64")
                        fpars[...] = p[3]
            # now check if we have a charge addon and if charges are made by topoqeq
            if "charge" in self.ffe.mol.loaded_addons:
                # check if charges are made by topoqeq
                charge = self.ffe.mol.charge
                method = charge.method.split(",")[0]
                if charge.has_charges and (method == "topoqeq" or method == "topoqeq_sparse"):
                    # we need to store charges
                    chgf = system.require_dataset("charge", shape=(na,), dtype="float32")
                    chgf[...] = charge.q
                    # store metadata
                    chgf.attrs["method"] = charge.method
        return

    def compare_system(self):
        """ method to verify that the mol object in the ffe is consistent with the stuff on file 
            NOTE: this can get arbitrary complex .. do a simple test of atypes here """
        self.open()
        OK = None
        if self.is_master:
            system = self.h5file["system"]
            mfp5_atypes = list(np.array(system["atypes"]).astype("str"))
            atypes = self.ffe.mol.get_atypes()
            OK = True
            if len(mfp5_atypes)==len(atypes):
                for a1,a2 in zip(mfp5_atypes, atypes):
                    if a1 != a2:
                        print ("not matching atomtypes %s %s" % (a1, a2))
                        OK = False
            else:
                OK = False
        OK = self.mpi_comm.bcast(OK)
        assert OK, "MFP5 ERROR: The system in the mfp5 file is not equivalent to your actual system. Aborting!"
        return

    def get_mol_from_system(self, vel=False, img=True, restart_ff=True, mol=None):
        """ read mol info from system group and generate a mol object 

        in parallel this is done on the master only and the data is broadcasted to the other nodes
        
        Args:
            - vel (bool, otional): Defaults to False. If True: return velocity array in addtion to mol object
        """
        self.open()
        # read the data from file on the master
        if self.is_master:
            system = self.h5file["system"]
            # for python3: now all strings coming from hdf5 are byte class types .. need to convert to strings
            elems  = list(np.array(system["elems"]).astype("str"))
            atypes = list(np.array(system["atypes"]).astype("str"))
            fragtypes = list(np.array(system["fragtypes"]).astype("str"))
            fragnumbers = list(system["fragnumbers"])
            if "cnc_table" in list(system.keys()):
                cnc_table = list(system["cnc_table"])
            else: 
                cnc_table = []
            bcd = system.attrs["bcd"]
        # broadcast if we run parallel
        if self.mpi_size > 1:
            if self.is_master:
                data = (elems, atypes, fragtypes, fragnumbers, cnc_table, bcd)
            else:
                data = None
            elems, atypes, fragtypes, fragnumbers, cnc_table, bcd = self.mpi_comm.bcast(data)
        # generate mol object
        na = len(elems)
        if mol == None:
            mol = molsys.mol()
        mol.set_natoms(na)
        mol.set_elems(elems)
        mol.set_atypes(atypes)
        mol.set_fragnumbers(fragnumbers)
        mol.set_fragtypes(fragtypes)
        mol.set_ctab(cnc_table, conn_flag=True)
        mol.bcond = bcd
        # now do the charges ... need to check on the master if there are charges and communicate this
        if self.is_master:
            if "charge" in list(system.keys()):
                charge = np.array(system["charge"], dtype="float64")
                has_charge = True
                method = system["charge"].attrs["method"]
            else:
                has_charge, method = False, None
        else:
            has_charge, method = False, None
        has_charge, method = self.mpi_comm.bcast((has_charge, method))
        if has_charge:
            mol.addon("charge")
            if not self.is_master:
                charge = np.empty([na], dtype="float64")
            self.mpi_comm.Bcast(charge)
            mol.charge.q[:] = charge
            mol.charge.method = method
            mol.charge.has_charges = True
        # now read the restart info that needs to be passed to the mol instance (Note: no velocities etc are read here)
        velocities = None
        images = None
        if self.is_master:
            try:
                rstage = self.h5file[self.restart_stage]
            except KeyError:
                self.pprint("MFP5 ERROR: The requested restart stage %s does not exist in the file!" % self.restart_stage)
                raise
            restart = rstage["restart"]
            xyz = np.array(restart["xyz"], dtype="float64")
            cell = np.array(restart["cell"], dtype="float64")
            if vel:
                velocities = np.array(restart["vel"], dtype="float64")
            if img:
                images = np.array(restart["img"], dtype="int32")            
        if self.mpi_size>1:
            if self.is_master:
                self.mpi_comm.Bcast(xyz)
                self.mpi_comm.Bcast(cell)
                if vel:
                    self.mpi_comm.Bcast(velocities)
                if img:
                    self.mpi_comm.Bcast(images)
            else:
                xyz = np.empty([na, 3], dtype="float64")
                cell = np.empty([3, 3], dtype="float64")
                self.mpi_comm.Bcast(xyz)
                self.mpi_comm.Bcast(cell)
                if vel:
                    velocities = np.empty([na, 3], dtype="float64")
                    self.mpi_comm.Bcast(velocities)
                if img:
                    images = np.empty([na, 3], dtype="int32")
                    self.mpi_comm.Bcast(images)
        mol.set_xyz(xyz)
        mol.set_cell(cell)
        # new check if addon data is present in the system group
        # start with ff
        ff_data = None
        if self.is_master:
            if ("ff" in list(system.keys())) and restart_ff:
                # ok, there is force field data in this file lets read it in as a packed directory
                ff_data = {}
                ff = system["ff"]
                ric = ff["ric"]
                par = ff["par"]
                ff_data["FF"] = par.attrs["FF"]
                for r in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
                    # read in ric integer arrays
                    if r in ric:
                        ff_data[r] = np.array(ric[r])
                    # now get the parameter
                    if r in par:
                        rpar = par[r]
                        ff_data[r+"_par"] = (\
                                        list(rpar["ptypes"]),\
                                        list(rpar["names"]),\
                                        np.array(rpar["npars"]),\
                                        np.array(rpar["pars"]))
        if self.mpi_size>1:
            ff_data = self.mpi_comm.bcast(ff_data)
        if ff_data is not None:
            # yes there was ff data .. set up mol addon ff
            mol.addon("ff")
            mol.ff.unpack(ff_data)
        return mol, velocities, images

    def get_mol_from_system_hpc(self, vel=False, img=True, restart_ff=True):
        """ read mol info from system group and generate a mol object only on the head node
        all othre nodes get None object

        TBI: this is clumsy .. it would be much better if we could bcast mol objects, but becasue mpi_comm is an attribute this
        can not be pickled. we need to find out if this can be avoided. for the moment we have two routines .. one that commincates all
        data and one (hpc mode) that does not.

        NOTE: it is a bit hacky but we need to communicate the velocities/images in case of a restart. that is why we bcast them here also in hpc mode.

        Args:
            - vel (bool, otional): Defaults to False. If True: return velocity array in addtion to mol object
        """
        self.open()
        # read the data from file on the master
        if self.is_master:
            system = self.h5file["system"]
            # for python3: now all strings coming from hdf5 are byte class types .. need to convert to strings
            elems  = list(np.array(system["elems"]).astype("str"))
            atypes = list(np.array(system["atypes"]).astype("str"))
            fragtypes = list(np.array(system["fragtypes"]).astype("str"))
            fragnumbers = list(system["fragnumbers"])
            if "cnc_table" in list(system.keys()):
                cnc_table = list(system["cnc_table"])
            else: 
                cnc_table = []
            bcd = system.attrs["bcd"]
            # generate mol object
            na = len(elems)
            mol = molsys.mol()
            mol.set_natoms(na)
            mol.set_elems(elems)
            mol.set_atypes(atypes)
            mol.set_fragnumbers(fragnumbers)
            mol.set_fragtypes(fragtypes)
            mol.set_ctab(cnc_table, conn_flag=True)
            mol.bcond = bcd
            # check for charges present in the system group
            if "charge" in list(system.keys()):
                mol.addon("charge")
                mol.charge.q[:] = np.array(system["charge"], dtype="float64")
                mol.charge.method = system["charge"].attrs["method"]
                mol.charge.has_charges = True
            # now read the restart info that needs to be passed to the mol instance (Note: no velocities etc are read here)
            try:
                rstage = self.h5file[self.restart_stage]
            except KeyError:
                self.pprint("MFP5 ERROR: The requested restart stage %s does not exist in the file!" % self.restart_stage)
                raise
            restart = rstage["restart"]
            xyz = np.array(restart["xyz"], dtype="float64")
            cell = np.array(restart["cell"], dtype="float64")
            mol.set_xyz(xyz)
            mol.set_cell(cell)
            # new check if addon data is present in the system group
            # start with ff
            ff_data = None
            if ("ff" in list(system.keys())) and restart_ff:
                # ok, there is force field data in this file lets read it in as a packed directory
                ff_data = {}
                ff = system["ff"]
                ric = ff["ric"]
                par = ff["par"]
                ff_data["FF"] = par.attrs["FF"]
                for r in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
                    # read in ric integer arrays
                    if r in ric:
                        ff_data[r] = np.array(ric[r])
                    # now get the parameter
                    if r in par:
                        rpar = par[r]
                        ff_data[r+"_par"] = (\
                                        list(rpar["ptypes"]),\
                                        list(rpar["names"]),\
                                        np.array(rpar["npars"]),\
                                        np.array(rpar["pars"]))
            if ff_data is not None:
                # yes there was ff data .. set up mol addon ff
                mol.addon("ff")
                mol.ff.unpack(ff_data)
            if vel:
                velocities = np.array(restart["vel"], dtype="float64")
            if img:
                images = np.array(restart["img"], dtype="int32")
        else:
            mol = None
        # now communicate the restart info
        if not self.is_master:
            na = 0
        na = self.mpi_comm.bcast(na)
        if vel:
            if not self.is_master:
                velocities = np.empty([na, 3], dtype="float64")
            self.mpi_comm.Bcast(velocities)
        else:
            velocities = None
        if img:
            if not self.is_master:
                images = np.empty([na, 3], dtype="int32")
            self.mpi_comm.Bcast(images)
        else:
            images = None
        return (mol, velocities, images) # if vel or images not requested they will be None

    ##### expot stuff ##################################################################

    def expot_write_to_system(self, name, data):
        """write setup info for an expot to system

        we provide a dictionary of fixed size numpy arrays (either int32 or float)
        and these will be written to the file for a retrieval
        we asume this info does NOT exist ... or overwrite it
        use expot_in_system to check

        Args:
            name (string): name of the expot
            data (dict): dict of numpy arrays
        """
        self.open()
        if self.is_master:
            system = self.h5file["system"]
            expot = system.require_group("expot")
            # within the expot group we need a group with the name of the expot
            this_expot = expot.require_group(name)
            for d in data:
                assert type(d) == type("")
                a = data[d]
                mfp5_a = this_expot.require_dataset(d, shape=a.shape, dtype=a.dtype)
                mfp5_a[...] = a
        return
    
    def expot_in_system(self, name):
        self.open()
        FOUND = False
        if self.is_master:
            system = self.h5file["system"]
            if "expot" in system:
                expot = system["expot"]
                if name in expot:
                    FOUND = True
        FOUND = self.mpi_comm.bcast(FOUND)
        return FOUND

    def expot_read_from_system(self, name):
        self.open()
        if self.is_master:
            system = self.h5file["system"]
            expot = system["expot"]
            this_expot = expot[name]
            data = {}
            for k in this_expot:
                data[k] = np.array(this_expot[k])
        else:
            data = None
        data = self.mpi_comm.bcast(data)
        return data

    #######################################################################

    def setup_molecules(self,mol):
        mol.addon('molecules')
        mm = mol.molecules
        molnames = ['xyz']
        for i,wm in enumerate(mm.whichmol):
            if wm != 0:
                if mol.fragtypes[i] not in molnames:
                    molnames.append(mol.fragtypes[i])
        mm.molnames = molnames
        mm.moltypes = [0] + [i+1 for i in mm.moltypes[1:]]
        return mol

    def add_stage(self, stage, simulation_parameters: dict = None):
        """add a new stage 
        
        Args:
            stage (string): stage name to add
            simulation_parameters(dict):A dictionary containing simulation parameters as key -> value pairs
                                        that will be added as attributes to the stage.
                                        An attribute is a key -> value pair, too.
                                        (see attributes in hdf5 documentation on the internet)

        Returns:
            (bool) True if it worked and False if the stage already existed
        """
        if stage == "system":
            self.pprint("MFP5 ERROR: Stage name system is not allowed!")
            raise IOError
        self.open()
        OK = True
        if self.is_master:
            if stage in list(self.h5file.keys()):
                OK = False
            else:
                sgroup = self.h5file.create_group(stage)
                #-----------------------------
                # add simulation parameters as attributes to the stage
                # check that the dictionary is not None and not empty
                if simulation_parameters is not None:
                    if simulation_parameters:
                        for key, value in simulation_parameters.items():
                            sgroup.attrs[key] = value
                    else:
                        print("simulation_parameters is empty")
                else: print("simulation_parameters is None")
                #-----------------------------
                rgroup = sgroup.create_group("restart")
                # generate restart arrays for xyz, cell, vel and img
                na = self.ffe.mol.get_natoms()
                rgroup.require_dataset("xyz", shape=(na,3), dtype="float64")
                rgroup.require_dataset("vel", shape=(na,3), dtype="float64")
                rgroup.require_dataset("cell", shape=(3,3), dtype="float64")
                rgroup.require_dataset("img", shape=(na,3), dtype="int32")
        OK = self.mpi_comm.bcast(OK)
        return OK

    def has_stage(self,stagename):
        return stagename in self.get_stages()
               

    def get_stages(self):
        stagelist = list(self.h5file.keys())
        stagelist.remove("system")
        return stagelist

    def get_traj_from_stage(self, sname):
        self.open()
        if self.is_master:
            if sname not in self.h5file.keys():
                return False
            stage = self.h5file[sname]
            if "traj" in stage.keys():
                traj = stage["traj"]
            else:
                traj = {}
        else:
            traj = None
        return traj
        

    def write_restart(self, stage=None, velocities=False):
        """write restart info to file

        Args:
            stage (string, optional): Defaults to None. Name of the stage to write restart (must exist). Uses current stage by default
            velocities (bool, optional): Defaults to False. When True writes also velocities
        """
        self.open()
        if stage is None:
            stage = self.stage
        OK = True
        # we need to do this in parallel on nodes (lammps gather is a parallel step)
        xyz = self.ffe.get_xyz()
        img = self.ffe.get_image()
        if velocities:
            vel = self.ffe.get_vel()
        if self.is_master:
            if stage not in list(self.h5file.keys()):
                OK = False
            else:
                restart = self.h5file[stage+"/restart"]
                cell = self.ffe.get_cell()
                rest_xyz = restart.require_dataset("xyz",shape=xyz.shape, dtype=xyz.dtype)
                rest_xyz[...] = xyz
                rest_img = restart.require_dataset("img",shape=img.shape, dtype=img.dtype)
                rest_img[...] = img
                if cell is not None:
                    rest_cell = restart.require_dataset("cell", shape=cell.shape, dtype=cell.dtype)
                    rest_cell[...] = cell
                if velocities:
                    rest_vel = restart.require_dataset("vel", shape=vel.shape, dtype=vel.dtype)
                    rest_vel[...] = vel
        OK = self.mpi_comm.bcast(OK)
        if not OK:
            self.pprint("MFP5 ERROR: writing restart to stage %s failed. Stage does not exist" % stage)
            raise IOError
        return

    def prepare_stage(self, stage, traj_data, traj_nstep, data_nstep=1, thermo_values=[], prec="float32", tstep=0.001):
        """prepare a stage for trajectory writing
        
        Args:
            stage (string): name of the stage (must exist)
            traj_data (list of strings): name of the data to be written
            traj_nstep (int): frequency in steps to write
            data_nstep (int or list of ints), optional): Defaults to 1. freq to write each datatype
            thermo_values (list): Defaults to [], list of strings of thermodynamic data ... each code needs to know what is meant here
            prec (str, optional): Defaults to "float64". precision to use in writing traj data
            tstep (float, optional): Defaults to 0.001. Timestep in ps
        """ 
        # RS function rewritten for parallel runs (in lammps some of the data_funcs need to be exectued by all nodes)
        self.open()       
        if type(data_nstep)== type([]):
            assert len(data_nstep)==len(traj_data)
        else:
            data_nstep = len(traj_data)*[data_nstep]
        OK = True
        if self.is_master:
            if stage not in list(self.h5file.keys()):
                OK = False
        OK = self.mpi_comm.bcast(OK)
        if not OK:
            raise IOError("MFP5 ERROR: preparing stge %s failed. Stage does not exist!" % stage)
        if self.is_master:
            traj = self.h5file.require_group(stage+"/traj")
            traj.attrs["nstep"] = traj_nstep
            traj.attrs["tstep"] = tstep
        for dname,dnstep in zip(traj_data, data_nstep):
            assert dname in list(self.ffe.data_funcs.keys())
            data = self.ffe.data_funcs[dname]()
            dshape = list(data.shape)
            dtype = data.dtype.name
            # reduce accuracy of float trajectory data on request.
            if dtype[:5] == "float":
                dtype = prec
            if self.is_master:
                mfp5_data = traj.require_dataset(dname, 
                                shape=tuple([1]+dshape),
                                maxshape=tuple([None]+dshape),
                                chunks=tuple([1]+dshape),
                                dtype=dtype)
                mfp5_data[...] = data
                mfp5_data.attrs["nstep"] = dnstep
        if len(thermo_values)>0:
            # write thermodynamic data .. size of array is length of list
            nthermo = len(thermo_values)
            if self.is_master:
                mfp5_data = traj.require_dataset("thermo",
                                shape=(1, nthermo),
                                maxshape=(None, nthermo),
                                chunks=(1, nthermo),
                                dtype=prec)
                # TBI .. get thermo data in ffe on python level
                mfp5_data[...] = 0.0
                mfp5_data.attrs["labels"] = " ".join(thermo_values)
        self.mpi_comm.barrier()
        return

    def add_bondtab(self, stage, nbondsmax, bond_tab = None, bond_order = None):
        """generates a bondtab/bondorder entry in the trajectory group of the current stage for ReaxFF
        
        Args:
            stage (string): name of the current stage     
            nbondsmax (int): size of the tables
            bond_tab (list) : bond table
            bond_order (list) : bond order
        """
        if self.is_master:
            st = self.h5file[stage]
            traj = st["traj"]
            bondtab = traj.require_dataset("bondtab",
                            shape=(1,nbondsmax,2),
                            maxshape=(None, nbondsmax,2),
                            chunks=(1,nbondsmax,2),
                            dtype = "int32")
            bondord = traj.require_dataset("bondord",
                            shape=(1,nbondsmax),
                            maxshape=(None, nbondsmax),
                            chunks=(1,nbondsmax),
                            dtype = "float32")
            if bond_tab is not None and bond_order is not None:
                bondtab[...] = bond_tab
                bondord[...] = bond_order
        return

        
    def __call__(self, force_wrest=False):
        """ this generic routine is called to save restart info to the hdf5 file
            if force_wrest is True then restart is written in any case """
        if not self.pd: raise IOError("No pydlpoly instance")
        data_written = False
        if (self.counter%self.rest_nstep == 0) or force_wrest:
            # restart is written .. fetch all data 
            for d in self.rest_data:
                if self.verbose:
                    self.pprint("Writing restart data %s to mfp5 file" % d)
                data = self.data_funcs[d]()
                self.rest_datasets[d][...] = data
            data_written = True
        if self.track_data != None:
            for i,d in enumerate(self.track_data):
                if (self.counter%self.traj_nstep[i] == 0):
                    self.traj_frame[i] += 1
                    data = self.data_funcs[d]()
                    tds = self.traj_datasets[d]
                    tds.resize(self.traj_frame[i], axis=0)
                    tds[self.traj_frame[i]-1,...] = data
                    data_written = True
        # now we are done 
        if data_written : self.h5file.flush()
        self.counter += 1
        return
    

        
