"""Reaction Data Base

This is RDB which implements a pyDAL database wrapper for a reaction database

First version .. most likely the database strcutre is going to change
currently we use a sqlite database and store files in the storage folder

"""

from pydal import DAL, Field
import io
import os
from collections import OrderedDict

import molsys

import copy

# DB typekeys
typekeys = {
    "s" : "string",
    "t" : "text",
    "i" : "integer",
    "b" : "boolean",
    "u" : "upload",
    "r" : "reference",
    "d" : "double",
    "li": "list:integer",
    "ls": "list:string"
}

class RDB:

    def __init__(self, db_path, mode="a", do_commit=True):
        """generate a RDB access object
        
        Args:
            db_path (string): path to the sqlite database and the storage folder
            mode (str, optional): mode for opening database: a is append (must exist) n is new (must not exist). Defaults to "a".
        """
        db_path = os.path.abspath(db_path)
        # check access mode
        if mode == "a":
            assert os.path.isdir(db_path)
            assert os.path.exists(db_path+"/RDB.db")
        elif mode == "n":
            assert not os.path.exists(db_path)
            os.mkdir(db_path)
        self.db = DAL("sqlite://RDB.db", folder=db_path)
        self.db_path = db_path
        self.do_commit = do_commit
        ###########################################################################################
        # define databse structure
        # 
        #           core of RDB -> here the tabel layout of the sql database is defined
        #

        # GS: Note maybe obvious, but as reminder for references the order in dbstruc matters
        dbstruc = OrderedDict()
        #
        # reaction space 
        # 
        dbstruc["reactions"] = [
            "b:uni",          # true if unimolecular
            "b:change",       # change in graph for educt and products (used for screening)
            "s:source",       # information on where this reactions comes from
            "r:reactions:origin",    # 
        ]

        dbstruc["species"] = [
            "s:sumform",      # sum formula
            "u:compare_data", # holds the molecular graph 
            "s:compare_type", # specifies the comparison type e.g. molgraph
        ]

        dbstruc["reac2spec"] = [
            "r:reactions",   # reference to the reactions table
            "r:species",     # reference to educt in species table
            "i:label",       # type (-1 educt, 0 TS, 1 product)            
        ]

        dbstruc["lot"] = [
            "s:name"          # level of theory
        ]
        dbstruc["opt_species"] = [
            "r:species",      # reference to species table 
            "r:lot",          # ref to lot
            "d:energy",       # energy (in kcal/mol or Hartree)
            "u:xyz",          # upload xyz file
            "u:mfpx",         # upload mfpx file        
            "u:png",          # thumbnail
            "s:path",         # path to input files for this job
            "s:info",         # additional information on the optimization you want to store
            "b:molgchange",   # indicates change in molgraph w.r.t. species
            "li:rbonds",      # reactive bonds (list with 2*nbonds atom ids of the TS)
        ]
        

        #
        # MD space 
        #
 
        dbstruc["md"] = [       
            "s:path",         # filename of the mfp5 file
            "s:stage",        # name of the stage
            "i:nframes",      # number of frames
            "d:timestep",     # time in fs between two frames (MD timestep times frame rate)
            "d:temp"          # temperature in Kelvin
        ]         # TBI !!! add more info here ... this is is just for testing
                  #         username, datetime, method, .... more

        dbstruc["revent"] = [
            "r:reactions",    # ref to table unique_revent
            "b:reversed",     # is it the back reaction in unique_revent?
            "r:md",           # ref to table md
            "b:uni",          # true if unimolecular
            "i:frame",        # frame number
            "li:ed",          # educt species (sorted)
            "li:ts",          # transition state species
            "li:pr",          # product species
            "i:tr_ed",        # number of traced educt species
            "i:tr_pr",        # number of traced product species
            "li:rbonds",      # reactive bonds (list with 2*nbonds atom ids of the TS)
        ]
        dbstruc["md_species"] = [
            "r:revent",       # ref to revent
            "r:species",      # ref to species
            "t:smiles",       # smiles
            "s:sumform",      # sum formula
            "i:spec",         # species ID in the frame
            "i:foffset",      # frame offset (-1 educt, 0 TS, 1 product)
            "u:mfpx",         # upload mfpx file
            "b:tracked",      # is tracked?
            "u:png",          # thumbnail
            "b:react_compl",  # is this a reactive complex
            "li:atomids",     # list of atom indeces in the system
        ]
        dbstruc["rgraph"] = [
            "r:revent:from_rev",        # source reaction event ref
            "r:revent:to_rev",          # target reaction event ref
            "r:md_species:from_spec",   # source species ref
            "r:md_species:to_spec",     # target species ref
        ]
        #############################################################################################
        self._define_tables(dbstruc)
        # define soem defaults
        self.current_md = None
        return

    def _define_tables(self, dbstruc, dryrun=False):
        """helper function to set up a database from a dictionary

        keys are teble names, vlaues are lists corresponding to fields with names split by :
        """ 
        # define some defaults here
        db=self.db
        for t in dbstruc:
            fields = []
            for f in dbstruc[t]:
                kwargs = {}
                d = f.split(":")
                assert d[0] in typekeys
                # catch special field types
                if d[0] == "r":
                    # make sure that ref is in tables
                    assert d[1] in dbstruc
                    if len(d)>2:
                        # if a specific field name is requestd
                        fieldname = d[2]
                    else:
                        # else use tablenameID
                        fieldname = d[1] + "ID"
                    fieldtype = "reference %s" % d[1]
                elif d[0] == "u":
                    # generate an upload field .. by default the folder is in /storage/<fieldname>
                    fieldname = d[1]
                    fieldtype = typekeys[d[0]]
                    kwargs["uploadfolder"] = "%s/storage/%s" % (self.db_path, fieldname)
                else:
                    fieldname = d[1]
                    fieldtype = typekeys[d[0]]
                # add field
                field = Field(fieldname, type=fieldtype, **kwargs)
                if dryrun:
                    print (str(field))
                else:
                    fields.append(field) 
            # now all fields are defined ... generate the table
            if dryrun:
                print (" define table %s with the above fields" % t)
            else:
                db.define_table(t, fields)
        db.commit()
        return

    ######### API methods #####################################################################

    def commit(self):
        self.db.commit()
        return

    def set_md_run(self, mfp5_fname, stage, **kwargs):
        # get the absolute pathname to have a reliable identifier
        mfp5_fname = os.path.abspath(mfp5_fname)
        # find out if mfp5_fname is in database
        rows = self.db((self.db.md.path == mfp5_fname) & (self.db.md.stage == stage)).select()
        assert len(rows) < 2, "The mfp5 file %s is twice in the database" % mfp5_fname
        if len(rows) == 0:
            # this is a new entry
            for k in ["nframes", "timestep", "temp"]:
                assert k in kwargs, "for a new md entry %s must be specified" % k
            # make sure that the mfp5 file really exists
            assert os.path.exists(mfp5_fname)
            mdID = self.db.md.insert(
                path     = mfp5_fname,
                stage    = stage,
                nframes  = kwargs["nframes"],
                timestep = kwargs["timestep"],
                temp     = kwargs["temp"]
            )
            self.db.commit()
        else:
            mdID = rows[0].id
        self.current_md = mdID
        return mdID

    def get_lot(self, lot):
        row = self.db(self.db.lot.name==lot).select().first()
        if row is None:
            id = self.db.lot.insert(name=lot)
        else:
            id = row.id
        return id
    
    def register_reaction(self, uni=False, change=True, source="fromMD", origin=None):

        if origin is None:
            reventID = self.db.reactions.insert(
                uni        = uni,
                change     = change,
                source     = source,
            )
        else:
            reventID = self.db.reactions.insert(
                uni        = uni,
                change     = change,
                source     = source,
                origin     = origin.id
            )

        if self.do_commit:
            self.db.commit()
        return reventID

    def register_revent(self, frame, ed, ts, pr, tr_ed, tr_pr, rbonds, uni=False):

        reventID = self.db.revent.insert(
            mdID       = self.current_md,
            uni        = uni,
            frame      = frame,
            ed         = ed,
            ts         = ts,
            pr         = pr,
            tr_ed      = tr_ed,
            tr_pr      = tr_pr,
            rbonds     = rbonds,
        )

        if self.do_commit:
            self.db.commit()
        return reventID

    def get_revent(self, frame):
        event = self.db( (self.db.revent.mdID == self.current_md) & (self.db.revent.frame == frame) ).select().first()
        return event

    def add_md_species(self, reventID, mol, spec, foff, aids : list, tracked=True, react_compl=False):
        # generate smiles
        mol.addon("obabel")
        smiles = mol.obabel.cansmiles
        sumform = mol.get_sumformula()
        # generate the file stream
        mfpxf = io.BytesIO(bytes(mol.to_string(), "utf-8"))
        # generate a filename
        fname = "%s_%d.mfpx" % (("ed", "ts", "pr")[foff+1], spec)
        aids.sort()
        # register in the database
        specID = self.db.md_species.insert(
            reventID     = reventID.id,
            smiles      = smiles,
            sumform     = sumform,
            spec        = spec,
            foffset     = foff,
            tracked     = tracked,
            mfpx        = self.db.md_species.mfpx.store(mfpxf, fname),
            react_compl = react_compl,
            atomids     = aids
        )
        if self.do_commit:
            self.db.commit()
        return specID

    def add_species(self, mol, check_if_included=False, compare_type = "molg_from_mol"):
        is_new = True
        assert compare_type in ["molg_from_mol"], "Unknown comparison type"
        
        mfpxf = io.BytesIO(bytes(mol.to_string(), "utf-8"))
        sumform = mol.get_sumformula()
        if  not hasattr(mol, "graph"):
           mol.addon("graph")
        mol.addon("obabel")
        mol.graph.make_graph()
        molg = copy.deepcopy(mol.graph.molg)
        is_chiral, centers = mol.obabel.check_chirality()
        smiles = mol.obabel.cansmiles
        if check_if_included:
           # get all species with same sumform
           specs = self.db((self.db.species.sumform == sumform)).select()
           for sp in specs:
              cmp_type = sp["compare_type"]
              if cmp_type == compare_type:
                 # only compare species of the same type
                 fname, mfpxf1 = self.db.species.compare_data.retrieve(sp.compare_data)
                 mfpxs = mfpxf1.read().decode('utf-8')
                 mfpxf1.close()
                 moldb = molsys.mol.from_string(mfpxs)
                 moldb.addon("graph")
                 moldb.graph.make_graph()
                 moldbg = moldb.graph.molg
                 is_equal, error_code = molsys.addon.graph.is_equal(moldbg, molg, use_fast_check=False)
                 if is_equal:
                    if is_chiral:
                        # We have a chiral center -> perform additionally a geo check.
                        moldb.addon("obabel")
                        smilesdb = moldb.obabel.cansmiles
                        if smiles != smilesdb:
                            continue
                    is_new = False
                    specID = sp.id
                    return specID, is_new
                    break
              else:
                 is_new = True
                 continue 
        if is_new:
           specID = self.db.species.insert(
               sumform       = sumform ,
               compare_data  = self.db.species.compare_data.store(mfpxf, "mol4molg.mfpx") ,
               compare_type  = compare_type
           )
        mfpxf.close()
        return specID, is_new

    def add_reac2spec(self, reactID, specID, itype):
        # register in the database
        reac2specID = self.db.reac2spec.insert(
            reactionsID = reactID,
            speciesID   = specID,
            label       = itype
        )
        if self.do_commit:
            self.db.commit()
        return reac2specID

    # TBI .. this is really stupid because we have to get revent for each species .. for DEBUG ok
    #        but merge these methods and make it more clever
    def get_revent_species(self,frame):
        # get requested revent of current md
        revent = self.db((self.db.revent.mdID == self.current_md) & (self.db.revent.frame == frame)).select().first()
        assert revent is not None, "No reaction event for frame %d" % frame
        return (revent.ed, revent.ts[0], revent.pr)


    def get_md_species(self, frame, spec, foff):
        # get requested revent of current md
        revent = self.db((self.db.revent.mdID == self.current_md) & (self.db.revent.frame == frame)).select().first()
        assert revent is not None, "No reaction event for frame %d" % frame
        # now get the species
        mdspec = self.db(
            (self.db.md_species.reventID == revent.id) &
            (self.db.md_species.foffset == foff) & 
            (self.db.md_species.spec == spec) &
            (self.db.md_species.react_compl == False)
        ).select().first()
        assert mdspec is not None, "No species %d for this reaction event" % spec
        # DEBUG DEBUG
        # print ("Frame %d Species %d frame offset %d" % (frame, spec, foff))
        # print (mdspec.smiles)
        return self.get_md_species_mol(mdspec)

    def get_md_species_mol(self, mdspec):
        """

        Args:
            - mdspec : a md_species database row
        """
        # get the file and convert to a mol object
        fname, mfpxf = self.db.md_species.mfpx.retrieve(mdspec.mfpx)
        mfpxs = mfpxf.read().decode('utf-8')
        mfpxf.close()
        mol = molsys.mol.from_string(mfpxs)
        return mol, fname

    def set_rgraph(self, from_fid, to_fid, from_spec, to_spec):
        """connect reaction events -> make an edge in the reaction graph
        
        Args:
            from_fid (int): frame id of intial Product frame
            to_fid (int): frame id of final Educt frame
            from_spec (int): spec id of initial species (products)
            to_spec (int): spec id of final species  (educts)
        """
        from_ev = self.get_revent(from_fid-1)
        to_ev   = self.get_revent(to_fid+1)
        from_smd = self.db(
            (self.db.md_species.reventID == from_ev.id) &
            (self.db.md_species.foffset == 1) &
            (self.db.md_species.spec == from_spec) &
            (self.db.md_species.react_compl == False)
        ).select().first()
        # assert from_smd is not None, "no species %d in frame %d to connect" % (from_spec, from_fid)
        to_smd = self.db(
            (self.db.md_species.reventID == to_ev.id) &
            (self.db.md_species.foffset == -1) &
            (self.db.md_species.spec == to_spec) &
            (self.db.md_species.react_compl == False)
        ).select().first()
        # assert to_smd is not None, "no species %d in frame %d to connect" % (to_spec, to_fid)
        # now we can add a new edge into the reaction graph
        if (from_smd is not None) and (to_smd is not None):
            rgraphID = self.db.rgraph.insert(
                from_rev  = from_ev.id,
                to_rev    = to_ev.id,
                from_spec = from_smd.id,
                to_spec   = to_smd.id 
            )
            if self.do_commit:
                self.db.commit()
        return

    def add_opt_species(self, mol, lot, energy, specID, path, change_molg=False, rbonds=None, info=""):
        """add an optimized structure to the DB
        
        Args:
            mol (mol object): structure to be stored
            lot (string or int): name or id of the level of theory
            energy (float): energy of the system (unit is defined by lot)
            mdspecID (int): reference id of the md_species entry
        """
        if type(lot) == type(""):
            lot = self.get_lot(lot)
        xyzf  = io.BytesIO(bytes(mol.to_string(ftype="xyz"), "utf-8"))
        mfpxf = io.BytesIO(bytes(mol.to_string(), "utf-8"))
        optID = self.db.opt_species.insert(
            speciesID = specID,
            lotID        = lot,
            energy       = energy,
            xyz          = self.db.opt_species.xyz.store(xyzf, "opt.xyz"),
            mfpx         = self.db.opt_species.mfpx.store(mfpxf, "opt.mfpx"),
            path         = path,
            info         = info,
            molgchange   = change_molg,
            rbonds       = rbonds
        )
        if self.do_commit:
            self.db.commit()
        return optID
        

################################################################################################

# reaction graph generation


    def view_reaction_graph(self, start=None, stop=None, browser="firefox", only_unique_reactions=False, plot2d=False, rlist = None):
        """ generate a reaction graph

        we use the current md (must be called before)

        if you use svg the database must be locally available (only links to the figures are stored)
        png can get very big

        Args:
            name (string, optional): name of the output file, default = rgraph
            format (string, optional): format of the output (either png or svg), default = png
            start (int, optional) : first frmae to consider
            staop (int, optional) : last frame to consider
            browser (string, optional) : browser for visualization
            only_unique_reactions (bool, optional) : show only unqiue reactions
            plot2d (bool, optional) : plot molecules as 2d structure
        """
        import pydot
        import tempfile
        import webbrowser
        import warnings
        # set the path to the images
        img_path = self.db_path + "/storage/png/"
        # get all relevant revents
        if start is None:
            start = -1

        already_visited = {}

        revents = self.db((self.db.revent.mdID == self.current_md) & \
                          (self.db.revent.frame >= start)).select(orderby=self.db.revent.frame)


        rgraph = pydot.Dot(graph_type="digraph")
        rgnodes = {} # store all generated nodes by their md_speciesID
        # start up with products of first event
        cur_revent = revents[0]

        mds = self.db((self.db.md_species.reventID == cur_revent) & \
                      (self.db.md_species.react_compl == False)   & \
                      (self.db.md_species.foffset == 1)).select()
           
        for m in mds:
            frame = cur_revent.frame
            if plot2d:
                fname, mfpxf = self.db.md_species.mfpx.retrieve(m.mfpx)
                mfpxs = mfpxf.read().decode('utf-8')
                mfpxf.close()
                mol = molsys.mol.from_string(mfpxs)
                mol.addon("obabel")
                fname = os.path.splitext(img_path+m.png)[0]
                fimg = fname+".svg"
                mol.obabel.plot_svg(fimg)  
            else:
                fimg = img_path+m.png
            new_node = pydot.Node("%d_pr_%d" % (frame, m.spec),
                                       image = fimg,
                                       label = "",
                                       height = 2.5,
                                       width  = 2.5,
                                       imagescale=False,
                                       shape = "box")
            rgraph.add_node(new_node)
            rgnodes[m.id] = new_node

        num_revents = 0

        # now loop over revents
        for (i, cur_revent) in enumerate(revents[1:]):
            if (stop is not None) and (cur_revent.frame > stop):
                break

            if rlist is not None:
                if not cur_revent.id in rlist:
                    continue

            # make the nodes of the revent
            educ = []
            prod = []
            reactID = cur_revent["reactionsID"]
            if only_unique_reactions:
 
                if reactID in already_visited:
                   continue
                else:
                   already_visited[reactID] = True

                reactions = self.db((self.db.reactions.id == reactID)).select().first()

                if not reactions["change"]:
                    continue

            mds = self.db((self.db.md_species.reventID == cur_revent) & (self.db.md_species.react_compl == False)  ).select()

            for m in mds:

                if plot2d:
                    fname, mfpxf = self.db.md_species.mfpx.retrieve(m.mfpx)
                    mfpxs = mfpxf.read().decode('utf-8')
                    mfpxf.close()
                    mol = molsys.mol.from_string(mfpxs)
                    mol.addon("obabel")
                    fname = os.path.splitext(img_path+m.png)[0]
                    fimg = fname+".svg"
                    mol.obabel.plot_svg(fimg)  
                else:
                    smiles = "\n"
                    fimg = img_path+m.png

                if m.foffset == -1:
                    new_node = pydot.Node("%d_ed_%d_react_%d" % (cur_revent.frame, m.spec, reactID),\
                                       image = fimg,\
                                       label = "reaction %s \n %s" % (reactID, m.sumform),\
                                       labelloc = "t", \
                                       height = 2.5, \
                                       width  = 2.5, \
                                       imagescale=False, \
                                       shape = "box")
                    educ.append(new_node)
                elif m.foffset == 1:
                    new_node = pydot.Node("%d_pr_%d_react_%d" % (cur_revent.frame, m.spec, reactID),\
                                       image = fimg,\
                                       label = "%s" % m.sumform,\
                                       labelloc = "t", \
                                       height = 2.5, \
                                       width  = 2.5, \
                                       imagescale=False, \
                                       shape = "box")
                    prod.append(new_node)
                else:
                    if cur_revent.uni:
                        label = "%10d (unimol)" % cur_revent.frame 
                    else:
                        label = "%10d" % cur_revent.frame
                    new_node = pydot.Node("%d_ts_%d_react_%d" % (cur_revent.frame, m.spec, reactID),\
                                       image = fimg,\
                                       label = label,\
                                       labelloc = "t",\
                                       shape = "box",\
                                       height = 2.5, \
                                       width  = 2.5, \
                                       imagescale=False, \
                                       style = "rounded")
                    ts = new_node
                rgraph.add_node(new_node)
                rgnodes[m.id] = new_node
            # now add edges
            for e in educ:
                rgraph.add_edge(pydot.Edge(e, ts))
            for p in prod:
                rgraph.add_edge(pydot.Edge(ts, p))
            # now connect from the previous events
            if only_unique_reactions:
                warnings.warn("Using only the unique reactions I can not generate a fully connected reaction graph!")
            else:
                concts = self.db(self.db.rgraph.to_rev == cur_revent).select()
                for c in concts:
                    rgraph.add_edge(pydot.Edge(rgnodes[c.from_spec], rgnodes[c.to_spec], color="blue"))
            num_revents += 1
        # done
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.curdir
            #os.chdir(tmpdir)
            rgraph.write_svg("rgraph_md%d.svg" %self.current_md)
            rgraph.write_png("rgraph_md%d.png" %self.current_md)
            webbrowser.get(browser).open_new("rgraph_md%d.svg" %self.current_md)
            #os.chdir(cwd)
 

        print("Number of plotted events: " + str(num_revents))
        
    



            

        





if __name__ == "__main__":
    rdb = RDB("./test", mode="n")


