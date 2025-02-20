# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:29:19 2016

@author: rochus

          Fragmentizer class

          depends on graph addon (this means graph_tool must be installed)
          
          MPI-safe version .. Warning: use prnt function which is overloaded


        RS (May 2020, corona times)
        This is a complete revision of the fragmentizer. we allow only fragments with more than 1 atom.
        All single atom fragments like halides Me groups, ether Os or thio SHs are assigned "by hand" in a hardcoded way
        Note that some names of fragments are thus defined here and NOT in the MOFplus database.

        - Option for local fragments is removed!!
        - fragments are cached if you do multiple fragmentizations

        RS (Nov 2024)
        Revision for the new assignement including use of the new API avoiding xmlrpc
        Experimental: add option to annotate atomtypes at the stage of fragmentization: special atom types that are not precisely defiend by the atype get additional information
        We do not need the catalog anymore since only relevant frags are downloaded
        


"""
from __future__ import print_function
import numpy
import logging
import molsys

import mofplus
import os
import sys

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
except ImportError as e:
    mpi_comm = None
    mpi_size = 1
    mpi_rank = 0
    mpi_err = e

# overload print function in parallel case
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
def print(*args, **kwargs):
    if mpi_rank == 0:
        return __builtin__.print(*args, **kwargs)
    else:
        return


import logging

logger = logging.getLogger("molsys.fragmentizer")

if mpi_comm is None:
    logger.error("MPI NOT IMPORTED DUE TO ImportError")
    logger.error(mpi_err)

############## single atom fragment rules #################################
# sarules_1 first attempt rules with full atomtype
# sarules_2 second attempt for atoms with only a root atomtype
sarules_1 = {
    "f1_c1"   : "f",         # we define the halogens explicit ... if they are not at C but at what else? N-Cl ?? 
    "cl1_c1"  : "cl",        #                                     --> should be a bigger fragment then
    "br1_c1"  : "br",        #
    "i1_c1"   : "i",         #
    "o2_c2"   : "eth",        # this is an ether and NOT an OH
    "o2_c1h1" : "oh",         # alcohol .. NOT a carboxylic acid .. found as co2-one anyway
    "s2_c1h1" : "sh",         # thiol
    "s2_c2"   : "thio",       # thioether
    "n3_c3"   : "tamin",      # tertiary amine
    "n3_c1h2" : "nh2",        # primary amine 
    "zn4_n4"  : "zn4",
    "cl0_"    : "cl-"         # chloride anion
}

sarules_2 = {
    "c4"      : "me",         # this is a methyl group .. sp3 carbon 
}

class fragmentizer:

    def __init__(self):
        """
        fragmentizer gets a catalog of fragments
        it will use the API to download from MOF+
        """
        # default
        self.fragments = {}          # this is a cache and can be reused if multiple calls are done
        self.frag_vtypes = {}        # do we need this?
        self.frag_prio = {}
        # API calls are done on the master only
        if mpi_rank == 0:
            self.api = mofplus.api()
            status = self.api.connect(os.environ["MFPUSER"], os.environ["MFPPW"])
            if not status:
                print ("Could not connect to MOF+")
                sys.exit(1)
        else:
            self.api = None
        return

    def read_frags_from_API(self, vtypes, verbose=False):
        """
        API call on master only ... broadcsted to other nodes 

        New (2024):
        - using napi 
        - only download relevant fragments
        - check if in cache already and download mol object only if new
        - need to broadcast the string and braodcast before converting to mol on all nodes (beause mol objects can not be pickled)
        """
        if mpi_rank == 0:
            frags = self.api.get_FFfrags(vtypes)
        else:
            frags = None
        if mpi_size > 1:
            frags = mpi_comm.bcast(frags, root=0)        
        # now check which fragments are not in the cache and download the mol files
        for f, entry in frags.items():
            if f not in self.fragments:
                self.fragments[f] = entry # add the entry to the cache
                # if there is an annotation convert the keys from string to int
                if entry["annotate"] is not None:
                    ann = entry["annotate"]
                    ann_int = {}
                    for k, v in ann.items():
                        ann_int[int(k)] = v
                    self.fragments[f]["annotate"] = ann_int
                if verbose:
                    print ("Fragment %s added to cache" % f)
                if mpi_rank == 0:
                    mfpxstr = self.api.get_mol_from_db(entry["file"], as_string=True)
                else:
                    mfpxstr = None
                if mpi_size > 1:
                    mfpxstr = mpi_comm.bcast(mfpxstr, root=0)
                m = molsys.mol.from_string(mfpxstr)
                m.addon("graph")
                m.graph.make_graph(hashes=False, omit_x=True)
                self.fragments[f]["mol"] = m
        return list(frags.keys()) # return a list of new fragments             

    def __call__(self, mol, plot=False, get_missing=False, verbose=True):
        """
        tries to assign all fragmnets in the catalog to the mol object

        :Parameters:

            - mol  (mol object): mol object to be fragmentized
            - plot (bool, opt): write fragment graphs as png file (defualts to False)
            - get_missing (bool, opt): if True analyze in case of failure and propose missing fragments 
            - verbose (bool, opt): talk to the user (defaults to True) 

        """
        if verbose:
            print ("This is Fragmentizer")
            print ("====================")
        # set all fragment info to none
        mol.set_nofrags()
        #
        mol.addon("graph")
        mol.graph.make_graph(hashes=False)
        if plot:
            mol.graph.plot_graph(plot, ptype="png", vsize=20, fsize=20)
        # get list of atypes
        atypes = mol.get_atypelist()
        vtypes = [e.split("_")[0] for e in atypes if (e[0] != "x") and (e[0] != "h")]
        vtypes = list(set(vtypes)) # remove duplicates
        if verbose:
            print ("The system contains the following root atomtypes for which we test fragments:")
            print (vtypes)
        # scan for relevant fragments
        frags = self.read_frags_from_API(vtypes, verbose=verbose) # this will get all fragments that are not in the cache (and broadcast to all nodes)
        # now sort according to prio
        frags.sort(key=lambda f: self.fragments[f]["priority"], reverse=True)
                        # now run over the system and test the fragments
        atypes = mol.get_atypes()
        fragnames = []        # this a list of all existing fragment names
        frag_atoms = []       # a list of the fragments with their atoms
        fi = 0
        if verbose:
            print ("Scanning for the following fragments:")
        for f in frags:
            if verbose:
                print (f"  - {self.fragments[f]['priority']:2d} {f:20s} : {self.fragments[f]['comment']}")
            fidx = mol.graph.find_fragment(self.fragments[f]["mol"], add_hydrogen=True)
            if verbose:
                print (f"    Found {len(fidx)} fragments")
            ann = self.fragments[f]["annotate"]
            for alist in fidx:
                # if any of the atoms in alist is already in a fragment we can skip
                assigned_already = any(mol.fragnumbers[i] >= 0 for i in alist)
                if not assigned_already:
                    if plot:
                        self.fragments[f]["mol"].graph.plot_graph(f, ptype="png", vsize=20, fsize=20, size=400)
                    for i in alist:
                        mol.fragtypes[i]   = f
                        mol.fragnumbers[i] = fi
                    fragnames.append(f)
                    frag_atoms.append(alist)
                    fi += 1
                    # if this fragment contains annotations then apply them for this alist
                    if ann is not None:
                        for i,v in ann.items():
                            # i is the index in ilist mapping to the global atom index
                            ii = alist[i]
                            orig_at = mol.atypes[ii]
                            if v.split("%")[0] != orig_at:
                                print (f"    WARNING!! Annotating atom {ii} with {v} (was {orig_at})")
                            mol.atypes[ii] = v
        # now all retrieved frags are tested .. now try to assign the remaining atoms with fragments if possible
        for i in range(mol.natoms):
            if mol.fragtypes[i] == '-1':
                at  = atypes[i] # atomtype
                rat = at.split("_")[0] # root atomtype
                # this is unassigned .. is it hydrogen?
                if rat != "h1":
                    fname = None
                    # first test explicit sarules_1
                    if at in sarules_1:
                        fname = sarules_1[at]
                    else:
                        if rat in sarules_2:
                            fname = sarules_2[rat]
                    # did we find something that applies?
                    if fname is not None:
                        # all hydogens connected to this atom are also part of the fragment (check if already assigned .. should not be)
                        alist = [i]
                        for j in mol.conn[i]:
                            if atypes[j].split("_")[0] == "h1":
                                alist.append(j)
                                assert mol.fragtypes[j] == "-1"
                        for i in alist:
                            mol.fragtypes[i] = fname
                            mol.fragnumbers[i] = fi
                        fragnames.append(fname)
                        frag_atoms.append(alist)
                        fi += 1
        nfrags =  max(mol.fragnumbers)+1
        # now analyse if fragmentation was successful and propse missing fragments if get_missing is True
        if verbose:
            print ("The following fragments have been found in the system")
            print (set(fragnames))
            nuassigned = mol.fragtypes.count('-1')
            if nuassigned > 0:
                print ("!!!! Unassigned atoms: %d    mol %s" % (nuassigned, mol.name))
        if get_missing:
            # try to identify groups of atoms
            pass

        return

    # this method should go evetually and is left here only for legacy reasons (who needs it as a static method?)
    @staticmethod
    def pure_check(mol):
        """
        check if all atoms are in a fragment and all is consistent
        """
        fragnames = []        # this a list of all existing fragment names
        frag_atoms = []       # a list of the fragments with their atoms
        nfrags =  max(mol.fragnumbers)+1
        fraglist  = [None]*(nfrags) # additional list to hold the fragments with their name
        for i in range(nfrags):
            frag_atoms.append([])
        for i in range(mol.natoms):
            ft = mol.fragtypes[i]
            fn = mol.fragnumbers[i]
            if ft == "0":
                return False
            else:
                if fraglist[fn] is None:
                    # set name of fragment
                    fraglist[fn] = ft
                else:
                    # check if this is the same name
                    if fraglist[fn] != ft: return False
                if ft not in fragnames:
                    fragnames.append(ft)
                frag_atoms[mol.fragnumbers[i]].append(i)
        # in the end make sure that all fragments have been named
        if None in fraglist: return False
        return True

    
