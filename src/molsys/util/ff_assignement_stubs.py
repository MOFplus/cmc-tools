
""" DEBUG DEBUG

this is a dummy file with methods taken from ff addon which need to be converted into a script

this is non-functional code .. please fix it

"""

class AssignmentError(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self,*args,**kwargs)

    def to_dict(self):
        rv = {}
        rv["error"]="AssignmentError"
        rv["message"]="Set of parameters is incomplete"
        return rv





    @timer("assign parameter")
    def assign_params_offline(self, ref, key = False, incomplete=False):
        loaded_pots = {'bnd':[],
                'ang':[],
                'dih':[],
                'oop':[],
                'vdw':[],
                'cha':[]}
        for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
            for params in self.par[ic].values():
                pot = params[0]
                if pot not in loaded_pots[ic]: loaded_pots[ic].append(pot)
        with self.timer("find rics"):
            self.ric.find_rics(specials = {'linear':[]})
            self._init_data()
        # check here for assignment based on a ric par file
        # or on an old key file
        with self.timer("make atypes"):
            self.aftypes = []
            for i, a in enumerate(self._mol.get_atypes()):
                if not key:
                    self.aftypes.append(aftype(a, self._mol.fragtypes[i]))
                else:
                    self.aftypes.append(aftype(a, "leg"))
        with self.timer("parameter assignement loop"):
            for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
                for i, r in enumerate(self.ric_type[ic]):
                    if self.parind[ic][i] is None:
                        full_parname_list = []
                        aft_list = self.get_parname_sort(r,ic)
                        parname  = tuple(aft_list)
                        sparname = list(map(str,parname))
                        full_parname_list = []
                        for p in loaded_pots[ic]:
                            #full_parname = p+"->("+string.join(sparname,",")+")|"+ref
                            # MODIFICATION, remove having to know the ref
                            full_parname = p+"->("+",".join(sparname)+")"
                            mod_pars = []
                            for pname in self.par[ic]:
                                semifull_parname = pname.split("|")[0]
                                if full_parname == semifull_parname and pname not in full_parname_list:
                                    full_parname_list.append(pname)
                        if full_parname_list != []:
                            self.parind[ic][i] = full_parname_list
        # check if all rics have potentials
        if not incomplete:
          self.check_consistency()
        # check if there are potentials which are not needed -> remove them
        for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
            # make a set
            badlist = []
            l = []
            #print (self.parind[ic][:])
            #print(ic)
            if not (self.parind[ic] is None):
              for i in self.parind[ic]: 
                if i is not None:
                  l+=i
                else:
                  l+=['']
              s = set(l)
              for j in self.par[ic].keys():
                  if j not in s:
                      badlist.append(j)
              # remove stuff from badlist
              for i in badlist: 
                  del self.par[ic][i]
                  #logger.warning("")


    @timer("assign multi parameters")
    def assign_multi_params(self, FFs, refsysname=None, equivs={}, azone = [], special_atypes = {}, smallring = False, generic = None):
        """
        Method to orchestrate the parameter assignment for the current system using multiple force fields
        defined in FFs by getting the corresponding data from the webAPI.

        Args:
            FFs (list): list of strings containing the names of the FFs which should be assigned. Priority
                is defined by the ordering of the list

        Keyword Args:
            refsysname (string): if set this is a refsystem leading to special
                treatment of nonidentified params; defaults to None
            equivs (dict): dictionary holding information on equivalences,
                needed for parameterization processes in order to "downgrade" atomtpyes, defaults to
                {}
            azone (list): list of indices defining the active zone, defaults to []
            special_atypes (dict): dict of special atypes, if empty special atypes from
                mofplus are requested, defaults to {}

        """
        for i, ff in enumerate(FFs):
            # first element
            if i == 0: 
                self.assign_params(ff, refsysname = refsysname,equivs = equivs, azone = azone, 
                        special_atypes = special_atypes, consecutive = True, ricdetect = True, smallring=smallring)
            # last element
            elif i == len(FFs)-1:
                self.par.FF = ff
                self.assign_params(ff, refsysname = refsysname, equivs = equivs, azone = azone, 
                        special_atypes = special_atypes, consecutive = False, ricdetect = False, smallring=smallring, generic = generic)
            # in between
            else:
                self.par.FF = ff
                self.assign_params(ff, refsysname = refsysname, equivs = equivs, azone = azone, 
                        special_atypes = special_atypes,consecutive = True, ricdetect = False, smallring=smallring)
        return


                
    @timer("assign parameter")
    def assign_params(self, FF, verbose=0, refsysname=None, equivs = {}, azone = [], special_atypes = {}, 
            plot=False, consecutive=False, ricdetect=True, smallring = False, generic = None, poltypes = [],
            dummies = None, cache = None, cross_terms = []):
        """
        Method to orchestrate the parameter assignment for this system using a force field defined with
        FF getting data from the webAPI

        Args:

            - FF        :    [string] name of the force field to be used in the parameter search
            - verbose   :    [integer, optional] print info on assignement process to logger;
            defaults to 0
            - refsysname:    [string, optional] if set this is a refsystem leading to special 
            treatment of nonidentified params; defaults to None
            - equivs    :    [dict, optional] dictionary holding information on equivalences,
            needed for parameterization processes in order to "downgrade" atomtpyes, defaults to
            {}
            - azone     :    [list, optional] list of indices defining the active zone;
            defaults to []
            - special_atypes : [dict, optional] dict of special atypes, if empty special atypes from
            mofplus are requested, defaults to {}
        """
        assert type(equivs) == dict
        assert type(azone) == list
        self.refsysname = refsysname
        self.equivs = equivs
        self.active_zone = azone
        if self.refsysname is None and len(self.equivs.keys()) > 0:
            raise IOError("Equiv feature can only used together with a defined refsysname")
        if self.refsysname is None and len(self.active_zone) > 0:
            raise IOError("Azone feature can only used together with a defined refsysname")
        with self.timer("connect to DB"):
            ### init api
            if self._mol.mpi_rank == 0:
                if cache is None:
                    from mofplus import FF_api
                    api = FF_api()
                    self.cache = api_cache(api)
                else:
                    self.cache = cache
                if len(special_atypes) == 0: special_atypes = self.cache.list_special_atypes()
            else:
                self.cache = None
                special_atypes = None
            if self._mol.mpi_size > 1:
                special_atypes = self._mol.mpi_comm.bcast(special_atypes, root = 0)
            #if self._mol.mpi_rank == 0:
            #    from mofplus import FF_api
            #    self.api = FF_api()
            #    if len(special_atypes) == 0: special_atypes = self.api.list_special_atypes()
            #else:
            #    self.api = None
            #    special_atypes = None
            #if self._mol.mpi_size > 1:
            #    special_atypes = self._mol.mpi_comm.bcast(special_atypes, root = 0)
        if ricdetect==True:
            with self.timer("find rics"):
                self.ric.find_rics(specials = special_atypes, smallring = smallring)
                if dummies is not None:
                    dummies(self)
                if len(poltypes)>0:
                    self._setup_coreshell(poltypes)           
                self._init_data()
                self._init_pardata(FF)
            # as a first step we need to generate the fragment graph
            with self.timer("fragment graph"):
                self._mol.addon("fragments")
                self.fragments = self._mol.fragments
                self.fragments.make_frag_graph() 
                if plot:
                    self.fragments.plot_frag_graph(plot, ptype="png", vsize=20, fsize=20, size=1200)
                # create full atomistic graph
                self._mol.graph.make_graph()
            # now make a private list of atom types including the fragment name
            with self.timer("make atypes"):
                self.aftypes = []
                for i, a in enumerate(self._mol.get_atypes()):
                    self.aftypes.append(aftype(a, self._mol.fragtypes[i]))
            # add molid info to ic["vdw"]
            with self.timer("make atypes"):
                self._mol.graph.get_components()
                for i, at in enumerate(self.ric_type["vdw"]):
                    at.molid = self._mol.graph.molg.vp.molid[i]
                self._mol.molid = self._mol.graph.molg.vp.molid.get_array()
        # detect refsystems
        self.find_refsystems_new(plot=plot)
        with self.timer("parameter assignement loop"):
            for ref in self.scan_ref:
                counter = 0
                logger.info("assigning params for ref system %s" % ref)
                curr_fraglist = self.ref_fraglists[ref]
                curr_atomlist = self.ref_atomlists[ref]
                curr_par = {\
                    "bnd" : self.ref_params[ref]["twobody"]["bnd"],\
                    "bnd5" : self.ref_params[ref]["twobody"]["bnd5"],\
                    "ang" : self.ref_params[ref]["threebody"]["ang"],\
                    "ang5" : self.ref_params[ref]["threebody"]["ang5"],\
                    "dih" : self.ref_params[ref]["fourbody"]["dih"],\
                    "oop" : self.ref_params[ref]["fourbody"]["oop"],
                    "cha" : self.ref_params[ref]["onebody"]["charge"],
                    "vdw" : self.ref_params[ref]["onebody"]["vdw"]
                    }
                curr_equi_par = {}
                for ic in ["bnd", "ang", "dih", "oop", "cha", "vdw"]:
                    if verbose>0: logger.info(" ### Params for %s ###" % ic)
                    if verbose== 3: 
                        logger.info(" ##### available pramas:\n %s \n\n" % str(curr_par[ic]))
                        logger.info(" ##### curr_fraglist \n %s" % str(curr_fraglist))
                        logger.info(" ##### curr_atomlist \n %s" % str(curr_atomlist))
                    for i, r in enumerate(self.ric_type[ic]):
                        if self.parind[ic][i] is None:
                            if ((self.atoms_in_subsys(r, curr_fraglist)) and (self.atoms_in_active(r, curr_atomlist))):
                                # no params yet and in current refsystem => check for params
                                full_parname_list = []
                                aft_list = self.get_parname_equiv(r,ic,ref)
                                #aft_list = map(lambda a: self.aftypes[a], r)
                                 # generate list of permuted tuples according to ic and look up params
                                #parname, par_list = self.pick_params(aft_list, ic, curr_par[ic])
                                parname, par_list = self.pick_params(aft_list, ic, r, curr_par)
                                if par_list != None:
                                    if verbose>1 : logger.info(" found parameter for atoms %20s (types %s) -> %s" % (str(r), aft_list, parname))
                                    for par in par_list:
                                        ### check for equivalences
                                        if par[0] == "equiv":
                                            for j, aft in enumerate(aft_list):
                                                aidx = r[j]
                                                if ((str(aft) == par[1][0]) and (aidx not in curr_equi_par)):
                                                    curr_equi_par[aidx] = par[1][1]
                                                    logger.info("  EQIV: atom %d will be converted from %s to %s" % (aidx, aft, par[1][1]))
                                        else:
                                            sparname = list(map(str, parname))
                                            full_parname = par[0]+"->("+",".join(sparname)+")|"+ref
                                            full_parname_list.append(full_parname)
                                            if not full_parname in self.par[ic]:
                                                logger.info("  added parameter to table: %s" % full_parname)
                                                self.par[ic][full_parname] = par
                                else:
                                    if verbose > 1:
                                        logger.info(" NO parameter for atoms %20s (types %s) " % (str(r), aft_list))
                                if full_parname_list != []:
                                    counter += 1
                                    self.parind[ic][i] = full_parname_list
                                #else:
                                #    print("DEBUG DEBUG DEBUG %s" % ic)
                                #    print(self.get_parname(r))
                                #    print(self.get_parname_sort(r, ic))
                logger.info("%i parameters assigned for ref system %s" % (counter,ref))
                #EQUIVALENCE
                # now all params for this ref have been assigned ... any equivalnce will be renamed now in aftypes
                for i, a in enumerate(copy.copy(self.aftypes)):
                    if i in curr_equi_par.keys():
                        at, ft = curr_equi_par[i].split("@")
                        self.aftypes[i] = aftype(at,ft)
        if consecutive==False:
            if refsysname:
                self.fixup_refsysparams(cross_terms=cross_terms)
            else:
                self.check_consistency(generic = generic)
        self.timer.report()
        return



    @timer("find reference systems")
    def find_refsystems_new(self, plot=None,):
        """
        function to detect the reference systems:
            - self.scan_ref      : list of ref names in the order to be searched
            - self.ref_systems   : dictionary of mol objects
            - self.ref_fraglist  : list of fragment indices belonging to this refsystem
            - self.ref_params    : paramtere dictionaries per refsystem (n-body/type)
        """
        with self.timer("get reference systems"):
            scan_ref  = []
            scan_prio = []
            if self._mol.mpi_rank == 0:
                ref_dic = self.cache.list_FFrefs(self.par.FF)
            else:
                ref_dic = []
            if self._mol.mpi_size > 1:
                ref_dic = self._mol.mpi_comm.bcast(ref_dic, root=0)
            for refname in ref_dic.keys():
                prio, reffrags, active, upgrades, atfix = ref_dic[refname]
                if len(reffrags) > 0 and all(f in self.fragments.get_fragnames() for f in reffrags):
                    scan_ref.append(refname)
                    scan_prio.append(prio)
                # check for upgrades
                elif upgrades and len(reffrags) > 0:
                    oreffrags = copy.deepcopy(reffrags)
                    for d,u in upgrades.items():
                        reffrags = [i.replace(d,u) for i in reffrags]
                        if all(f in self.fragments.get_fragnames() for f in reffrags):
                            scan_ref.append(refname)
                            scan_prio.append(prio)
            # sort to be scanned referecnce systems by their prio
            self.scan_ref = [scan_ref[i] for i in np.argsort(scan_prio)]
            self.scan_ref.reverse()
        # now get the refsystems and make their fraggraphs and atomistic graphs of their active space
        with self.timer("make ref frag graphs"):
            self.ref_systems = {}
            if self._mol.mpi_rank == 0:
                ref_mol_strs  = self.cache.get_FFrefs_graph(self.scan_ref, )
            else:
                ref_mol_strs = {}
            if self._mol.mpi_size > 1:
                ref_mol_strs = self._mol.mpi_comm.bcast(ref_mol_strs, root = 0)
            for ref in self.scan_ref:
    #            if self._mol.mpi_rank == 0:
    #                ref_mol_str = self.api.get_FFref_graph(ref, out="str")
    #            else:
    #                ref_mol_str = None
    #            if self._mol.mpi_size > 1:
    #                ref_mol_str = self._mol.mpi_comm.bcast(ref_mol_str, root=0)
                ref_mol = molsys.mol.from_string(ref_mol_strs[ref])
                ref_mol.addon("fragments")
                ref_mol.fragments.make_frag_graph()
                if plot:
                    ref_mol.fragments.plot_frag_graph(ref, ptype="png", size=600, vsize=20, fsize=20)
                # if active space is defined create atomistic graph of active zone
                active = ref_dic[ref][2]
                if active: ref_mol.graph.make_graph(active)
                self.ref_systems[ref] = ref_mol
        # now search in the fraggraph for the reference systems
        with self.timer("scan for ref systems"):
            logger.info("Searching for reference systems:")
            self.ref_fraglists = {}
            self.ref_atomlists = {}
            for ref in copy.copy(self.scan_ref):
                # TODO: if a ref system has only one fragment we do not need to do a substructure search but
                #       could pick it from self.fragemnts.fraglist
                subs = self._mol.graph.find_subgraph(self.fragments.frag_graph, self.ref_systems[ref].fragments.frag_graph)
                # in the case that an upgrade for a reference system is available, it has also to be searched
                # for the upgraded reference systems
                upgrades = ref_dic[ref][3]
                if upgrades:
                    if len(upgrades) != 1:
                        raise ValueError('Currently, only one upgrade is supported')
                    # if upgrades should be applied, also an active zone has to be present
                    assert ref_dic[ref][2] != None
                    for s,r in upgrades.items():
                        subs_upgrade = []
                        n_upgrade_frags = self.ref_systems[ref].fragments.get_occurence_of_frag(s)
                        for i in range(n_upgrade_frags):
                            self.ref_systems[ref].fragments.upgrade(s, r, rep_n=1)
                            subs_upgrade += self._mol.graph.find_subgraph(self.fragments.frag_graph, self.ref_systems[ref].fragments.frag_graph)
                        subs += subs_upgrade
                logger.info("   -> found %5d occurences of reference system %s" % (len(subs), ref))
                if len(subs) == 0:
                    # this ref system does not appear => discard
                    self.scan_ref.remove(ref)
                    del(self.ref_systems[ref])
                else:
                    # join all fragments
                    subs_flat = list(set(itertools.chain.from_iterable(subs)))
                    self.ref_fraglists[ref] = subs_flat
                    # now we have to search for the active space
                    # first construct the atomistic graph for the sub in the real system if 
                    # an active zone is defined
                    if ref_dic[ref][2] != None and type(ref_dic[ref][2]) != str:
                        idx = self.fragments.frags2atoms(subs_flat)
                        self._mol.graph.filter_graph(idx)
                        asubs = self._mol.graph.find_subgraph(self._mol.graph.molg, self.ref_systems[ref].graph.molg)
                        if len(asubs) == 0:
                            # no substructure found .. let us plot the two graphs
                            print ("DOING GRAPH refsys")
                            self.ref_systems[ref].graph.plot_mol_graph("sub_%s" % ref, self.ref_systems[ref].graph.molg)
                            # print ("DOING GRAPH parent")
                            # self._mol.graph.plot_mol_graph("parent_%s" % ref, self._mol.graph.molg, size=2000)
                        ### check for atfixes and change atype accordingly, the atfix number has to be referred to its index in the azone
                        if ref_dic[ref][4] != None and type(ref_dic[ref][4]) != str:
                            atfix = ref_dic[ref][4]
                            for s in asubs:
                                for idx, at in atfix.items():
                                    azone = ref_dic[ref][2]
                                    self.aftypes[s[azone.index(int(idx))]].atype = at
                        self._mol.graph.molg.clear_filters()
                        asubs_flat = itertools.chain.from_iterable(asubs)
                        self.ref_atomlists[ref] = list(set(asubs_flat))
                    else:
                        self.ref_atomlists[ref] = None
        # get the parameters
        with self.timer("get ref parmeter sets"):
            if self._mol.mpi_rank == 0:
                self.ref_params = self.cache.get_ref_params(self.scan_ref, self.par.FF)
            else:
                self.ref_params = None
            if self._mol.mpi_size > 1:
                self.ref_params = self._mol.mpi_comm.bcast(self.ref_params, root=0)

            #self.ref_params = {}
            #for ref in self.scan_ref:
                #logger.info("Getting params for %s" % ref)
                #if self._mol.mpi_rank == 0:
                #    ref_par = self.api.get_params_from_ref(self.par.FF, ref)
                #else:
                #    ref_par = None
                #if self._mol.mpi_size > 1:
                #    ref_par = self._mol.mpi_comm.bcast(ref_par, root=0)                
                #self.ref_params[ref] = ref_par


                #print(("DEBUG DEBUG Ref system %s" % ref))
                #print((self.ref_params[ref]))
        return


    @timer("find reference systems")
    def find_refsystems(self, plot=None):
        """
        function to detect the reference systems:
            - self.scan_ref      : list of ref names in the order to be searched
            - self.ref_systems   : dictionary of mol objects
            - self.ref_fraglist  : list of fragment indices belonging to this refsystem
            - self.ref_params    : paramtere dictionaries per refsystem (n-body/type)
        """
        with self.timer("get reference systems"):
            scan_ref  = []
            scan_prio = []
            if self._mol.mpi_rank == 0:
                ref_dic = self.api.list_FFrefs(self.par.FF)
            else:
                ref_dic = []
            if self._mol.mpi_size > 1:
                ref_dic = self._mol.mpi_comm.bcast(ref_dic, root=0)
            for refname in ref_dic.keys():
                prio, reffrags, active, upgrades, atfix = ref_dic[refname]
                if len(reffrags) > 0 and all(f in self.fragments.get_fragnames() for f in reffrags):
                    scan_ref.append(refname)
                    scan_prio.append(prio)
                # check for upgrade   
            #    elif upgrades and len(reffrags) > 0:        
            #        print (refname)
            #        oreffrags = copy.deepcopy(reffrags)
            #        for d,u in upgrades.items():
            #            reffrags = [i.replace(d,u) for i in reffrags]
            #            if all(f in self.fragments.get_fragnames() for f in reffrags):
            #                scan_ref.append(refname)
            #                scan_prio.append(prio)
            # sort to be scanned referecnce systems by their prio
            self.scan_ref = [scan_ref[i] for i in np.argsort(scan_prio)]
            self.scan_ref.reverse()
        # now get the refsystems and make their fraggraphs and atomistic graphs of their active space
        with self.timer("make ref frag graphs"):
            self.ref_systems = {}
            if self._mol.mpi_rank == 0:
                ref_mol_strs  = self.api.get_FFrefs_graph(self.scan_ref, out ="str")
            else:
                ref_mol_strs = {}
            if self._mol.mpi_size > 1:
                ref_mol_strs = self._mol.mpi_comm.bcast(ref_mol_strs, root = 0)
            for ref in self.scan_ref:
    #            if self._mol.mpi_rank == 0:
    #                ref_mol_str = self.api.get_FFref_graph(ref, out="str")
    #            else:
    #                ref_mol_str = None
    #            if self._mol.mpi_size > 1:
    #                ref_mol_str = self._mol.mpi_comm.bcast(ref_mol_str, root=0)
                ref_mol = molsys.mol.from_string(ref_mol_strs[ref])
                ref_mol.addon("fragments")
                ref_mol.fragments.make_frag_graph()
                if plot:
                    ref_mol.fragments.plot_frag_graph(ref, ptype="png", size=600, vsize=20, fsize=20)
                # if active space is defined create atomistic graph of active zone
                active = ref_dic[ref][2]
                if active: ref_mol.graph.make_graph(active)
                self.ref_systems[ref] = ref_mol
        # now search in the fraggraph for the reference systems
        with self.timer("scan for ref systems"):
            logger.info("Searching for reference systems:")
            self.ref_fraglists = {}
            self.ref_atomlists = {}
            allowed_downgrades = ["Zn4O_benz"]
            for ref in copy.copy(self.scan_ref):
                # TODO: if a ref system has only one fragment we do not need to do a substructure search but
                #       could pick it from self.fragemnts.fraglist
                subs = self._mol.graph.find_subgraph(self.fragments.frag_graph, self.ref_systems[ref].fragments.frag_graph)
                # in the case that an upgrade for a reference system is available, it has also to be searched
                # for the upgraded reference systems
                #upgrades = ref_dic[ref][3]
                #if upgrades:
                #    # if upgrades should be applied, also an active zone has to be present
                #    assert ref_dic[ref][2] != None
                #    for s,r in upgrades.items():
                #        self.ref_systems[ref].fragments.upgrade(s, r)
                #        subs += self._mol.graph.find_subgraph(self.fragments.frag_graph, self.ref_systems[ref].fragments.frag_graph)
                # check again for vtypes2 fragments (substituted phenyl like fragments)
                if (len(subs) == 0) and (ref in allowed_downgrades):
                    self._mol.graph.plot_graph("ref", g = self.ref_systems[ref].fragments.frag_graph)
                    self._mol.graph.plot_graph("host", g = self.fragments.frag_graph, vertex_text=self.fragments.frag_graph.vp.types2)
                    subs += self._mol.graph.find_subgraph(self.fragments.frag_graph, self.ref_systems[ref].fragments.frag_graph, 
                        graph_property = self.fragments.frag_graph.vp.types2)
                logger.info("   -> found %5d occurences of reference system %s" % (len(subs), ref))
                if len(subs) == 0:
                    # this ref system does not appear => discard
                    self.scan_ref.remove(ref)
                    del(self.ref_systems[ref])
                else:
                    # join all fragments
                    subs_flat = list(set(itertools.chain.from_iterable(subs)))
                    self.ref_fraglists[ref] = subs_flat
                    # now we have to search for the active space
                    # first construct the atomistic graph for the sub in the real system if 
                    # an active zone is defined
                    if ref_dic[ref][2] != None and type(ref_dic[ref][2]) != str:
                        idx = self.fragments.frags2atoms(subs_flat)
                        self._mol.graph.filter_graph(idx)
                        asubs = self._mol.graph.find_subgraph(self._mol.graph.molg, self.ref_systems[ref].graph.molg)
                        ### check for atfixes and change atype accordingly, the atfix number has to be referred to its index in the azone
                        if ref_dic[ref][4] != None and type(ref_dic[ref][4]) != str:
                            atfix = ref_dic[ref][4]
                            for s in asubs:
                                for idx, at in atfix.items():
                                    azone = ref_dic[ref][2]
                                    self.aftypes[s[azone.index(int(idx))]].atype = at
                        self._mol.graph.molg.clear_filters()
                        asubs_flat = itertools.chain.from_iterable(asubs)
                        self.ref_atomlists[ref] = list(set(asubs_flat))
                    else:
                        self.ref_atomlists[ref] = None
        # get the parameters
        with self.timer("get ref parmeter sets"):
            self.ref_params = {}
            for ref in self.scan_ref:
                logger.info("Getting params for %s" % ref)
                if self._mol.mpi_rank == 0:
                    ref_par = self.api.get_params_from_ref(self.par.FF, ref)
                else:
                    ref_par = None
                if self._mol.mpi_size > 1:
                    ref_par = self._mol.mpi_comm.bcast(ref_par, root=0)                
                self.ref_params[ref] = ref_par
                #print(("DEBUG DEBUG Ref system %s" % ref))
                #print((self.ref_params[ref]))
        return
