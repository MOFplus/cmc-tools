"""

           findR

           find reactions in a ReaxFF trajectory

           this code is inspired by chemtrayzer but focuses on heterogeneous gas phase reactions
           2019 RUB Rochus Schmid



"""

import numpy as np
import pickle
#import copy
import os

import molsys
from molsys.util import mfp5io 

from molsys.util.timer import timer, Timer
from molsys.util.print_progress import print_progress
from molsys import mpiobject
from molsys.util import RDB
# import external classes 
from molsys.util.findR_classes import frame, species, fcompare, revent

import sys

class findR(mpiobject):

    def __init__(self, mfp5filename,  stage, rdb_path, mpi_comm = None, out = None):
        super(findR,self).__init__(mpi_comm, out)
        # To be tested: open read only on all nodes in parallel .. does it work?
        self.mfp5 = mfp5io.mfp5io(mfp5filename, filemode="r")
        self.mol = self.mfp5.get_mol_from_system()
        assert stage in self.mfp5.get_stages(), "Stage %s not in stages in file"
        self.traj = self.mfp5.h5file["/%s/traj" % stage]
        self.rest = self.mfp5.h5file["/%s/restart" % stage]
        data = list(self.traj.keys())
        # make sure that xyz and bondord/bondterm is available
        assert "xyz" in data
        assert "bondord" in data
        assert "bondtab" in data
        self.f_xyz = self.traj["xyz"]
        self.f_bondord = self.traj["bondord"]
        self.f_bondtab = self.traj["bondtab"]
        self.cell = np.array(self.rest["cell"])
        # get some basic info from the mol object
        self.natoms = self.mol.get_natoms()
        self.elems  = self.mol.get_elems()
        self.nframes = self.f_xyz.shape[0]
        # defaults and data structures
        self.frames = self.nframes*[None]
        self.timer = Timer("findR")
        self.bondord_cutoff = 0.5  # below this bond roder a bond is registered but swithced off
        self.min_atom = 6          # below this number of atoms a species is considered as gasphase
        self.min_elems = {"c" : 2} # below this number of atoms of the given element in the species it is not tracked
        # for debugging
        self.nunconnected = 0
        self.unconnected = []
        # for the new connector 
        self.unconn_species = [] # contains a list of unconnected species of prior reaction events (tracked products only)
        # for the search
        self.sstep ={
            "forward" : 200,
            "back"    : -10,
            "fine"    : 1,
        }
        self.skip_recross = 2 # number of frames to check if a reaction event was reverted
        # now open the RDB    TBI: how to deal with a parallel run?
        self.rdb = RDB.RDB(rdb_path, do_commit=False) # we open in no-commit mode -> have to call commit ourselves
        # TBI  get temp and timestep from mfp5 file
        self.rdb.set_md_run(mfp5filename, stage, nframes=self.nframes, temp=2000.0, timestep=10.0)
        return

    @timer("process_frame")
    def process_frame(self, fid):
        """process a single frame

        Args:
            fid (int): number of frame to process
            
        Returns:
            frame: frame object
        """
        if self.frames[fid] is not None:
            return self.frames[fid]
        # get the essential data from the open mfp5 file for the current frame
        bondord = np.array(self.f_bondord[fid])
        bondtab = np.array(self.f_bondtab[fid])
        xyz = np.array(self.f_xyz[fid])
        # make the frame object (pass settings to track species)
        f = frame(fid, xyz, self.mol, bondtab, bondord,\
                            cutoff=self.bondord_cutoff,\
                            min_atom=self.min_atom,\
                            min_elems=self.min_elems)
        # store the frame
        self.frames[fid] = f
        return f

    def get_comparer(self, f1, f2):
        if not self.frames[f1]:
            self.process_frame(f1)
        if not self.frames[f2]:
            self.process_frame(f2)
        return fcompare(self.frames[f1], self.frames[f2])
    
    @timer("do_frames")
    def do_frames(self, start=0, stop=None, stride=1, progress=True, plot=False):
        """generate all frames (molg, species) representations
        
        Args:
            start (integer, optional): starting frame. Defaults to None.
            stop (integer, optional): final frame (not inlcuded). Defaults to None.
            stride (integer, optional): stride. Defaults to None.
        """ 
        if not stop:
            stop = self.nframes
        if progress:
            self.pprint ("Processing frames from %d to %d (stride=%d)" % (start, stop, stride))
        for i in range(start, stop, stride):
            if progress:
                ci = (i-start)/stride
                print_progress(ci, (stop-start)/stride, prefix="Processing frames:", suffix=" done!")
            f = self.process_frame(i)
            if plot:
                f.plot()
        return

    @timer("search_frames")
    def search_frames(self, verbose=False):
        """search the frames for reactions 

        This is the core routine of findR
        """
        mode = "forward"     # search mode
        currentf = 0          # current frame
        #last_event = currentf # last reaction event => product frame of a reaction (initial frame is also an event)
        open_events = []      # list of reaction events located in a forward segment (located by backtracking)
        self.process_frame(currentf)
        self.store_initial_event()
        nextf = currentf+self.sstep[mode]
        # start mainloop (just search on until we are at the end of the file)
        # init some variables
        delta_segment_end = 0
        segment_events = []
        first_event = None # this variable stores the maximum fid of a reaction event at the beginning of a segment being a recrossing
        last_event  = None # this stores the last revent object of a segment (None in the first segment) 
        stop = False
        while nextf < self.nframes and nextf >= 0 and not stop:
            if not verbose:
                print_progress(nextf/self.sstep["forward"], self.nframes/self.sstep["forward"], suffix="Scanning Frames")
            self.process_frame(nextf)
            # do comparison and report
            comparer = self.get_comparer(currentf, nextf)
            flag = comparer.check()
            if flag>0:
                # a difference was found between current and next
                if mode == "forward":
                    segment_end = nextf
                    segment_start = currentf
                    mode = "back"
                elif mode == "back":
                    subseg_end = currentf  # NOTE for the subsegment we go backwards and the role of nextf
                    subseg_start = nextf   #      and currentf changes
                    mode = "fine"
                elif mode == "fine":
                    revt_lst = []
                    if flag == 1:
                        # we found a bimolecular reaction
                        comparer.analyse_aids()
                        # search critical bond
                        comparer.find_react_bond()   
                        # now we are ready to make a reaction event and store it
                        #revt = revent(comparer, self)
                        #TS_fid = revt.TS_fid

                        for ireac in range(comparer.nreacs):
                            revt_lst.append(revent(comparer, self, ireac=ireac))

                        TS_fid = revt_lst[0].TS_fid

                        if verbose:
                            print ("###########  Event at %d  #############" % TS_fid)
                    elif flag == 2:
                        # unimolecular reaction
                        comparer.analyse_bonds() # this method already finds the reactive bonds
                        # make reaction event and store
                        #revt = revent(comparer, self, unimol=True)
                        #TS_fid = revt.TS_fid

                        for ireac in range(comparer.nreacs):
                            revt_lst.append(revent(comparer, self, unimol=True, ireac=ireac))

                        TS_fid = revt_lst[0].TS_fid
                        if verbose:
                            print ("###########  Unimol Event at %d  #############" % TS_fid)
                    # now add the event to the list
                    #segment_events.append(revt)
                    segment_events.extend(revt_lst)

                    # first we need to make sure that there is no other reaction in the subsegment
                    #   test if the product (TS_fid+1) is equal to the end of the subsegment (subseg_end)
                    #   if this is not the case then we have to go forward in fine mode further on
                    if TS_fid+1 >= self.nframes:
                        print("Warning! We run out of frames")
                        break
                    comparer_subseg = self.get_comparer(TS_fid+1, subseg_end)
                    flag = comparer_subseg.check()
                    if flag > 0:
                        if verbose:
                            print ("The product frame %d and the end of the subsegment at %d are not equal" % (TS_fid+1, subseg_end))
                            print ("should continue forward in fine mode ... ")
                        # need not to do anything (mode stay fine)
                    else:
                        # next we need to make sure that the start of the subseg species
                        #   are really idential to where we started the segment (segment_start)
                        comparer_segment = self.get_comparer(segment_start, subseg_start)
                        flag = comparer_segment.check()
                        if flag > 0:
                            if verbose:
                                print ("The subseg start %d and the segment start %d are not equal" % (subseg_start, segment_start))
                                print ("continue in back mode")
                            nextf = subseg_start
                            mode = "back"
                        else:
                            # At this point we can be sure that all events in the segment are now in the list
                            # now we should get rid of recrossings, add to the DB and connect
                            # first sort our events (because of backtracking they might not be sorted)
                            event_TSfids = [r.TS_fid for r in segment_events]
                            event_index = np.argsort(event_TSfids)
                            segment_events[:] = [segment_events[i] for i in event_index]
                            event_TSfids = [r.TS_fid for r in segment_events]
                            if verbose:
                                print ("events in segment: %s" % str(event_TSfids))
                            ### handle recrossings ################################################
                            if self.skip_recross > 0:
                                # find recrossings if skip_recross is larger than 1 -> add event indices in list recross and remove
                                recross = []
                                start_recross = 0  # if the first event is part of a recrossing over the segment boundary then set this to 1
                                # is there a possible early event?
                                if first_event is not None:
                                    # we should have a reaction event before the frame given in first event
                                    if event_TSfids[0] <= first_event:
                                        if verbose:
                                            print ("  found an early event for recrossing at %d" % event_TSfids[0])
                                        first_event = None
                                        recross = [0]
                                        start_recross = 1
                                # check the events for any recross
                                if (len(event_TSfids)+start_recross) > 1: 
                                    for i in range(start_recross, len(event_TSfids)-1):
                                        e1 = event_TSfids[i]
                                        e2 = event_TSfids[i+1]
                                        if (e2-e1) <= self.skip_recross:
                                            # two events are in recrossing distance 
                                            if verbose:
                                                print ("possible recross between %d and %d" % (e1, e2))
                                            compare_recross = self.get_comparer(e1-1, e2+1) # compare ED of event1 and PR of event2
                                            flag = compare_recross.check()
                                            if flag == 0:
                                                # this is a recrossing
                                                recross += [i, i+1]
                                                if verbose:
                                                    print ("recrossing found")
                                            else:
                                                if verbose:
                                                    print ("no recrossing")
                                # check if the last event is close enough to the segment bound 
                                e = event_TSfids[-1]
                                if segment_end-e <= self.skip_recross:
                                    compare_recross = self.get_comparer(e-1, e+self.skip_recross+1)
                                    flag = compare_recross.check(verbose=True)
                                    if flag == 0:
                                        # this is a recrossing over the segment bound
                                        if verbose:
                                            print ("recrossing over segment boundary from %d to %d" % (e-1, e+self.skip_recross+1))
                                            # DEBUG DEBUG ... analysie what happens over the segment boundary
                                            for ie in range(e-3, e+self.skip_recross+4):
                                                if self.frames[ie] is None:
                                                    self.process_frame(ie)
                                                print ("Sumform at frame %5d : %s" % (ie, self.frames[ie].get_main_species_formula()))
                                            # DEBUG DEBUG END
                                        recross += [len(event_TSfids)-1]
                                        first_event = e+self.skip_recross
                                        # set segment_end to the end of the recrossing .. if we move forward we want to compare to the end of the recrossing event
                                        print ("segment end was: %d" % segment_end)
                                        segment_end_new = e+self.skip_recross+1
                                        delta_segment_end = segment_end_new - segment_end
                                        print ("delta is %d" % delta_segment_end)
                                        segment_end = segment_end_new
                                if verbose:
                                    if len(recross) > 0:
                                        print ("The following event indices are marked to be removed because of recrossing")
                                        print (str(recross))
                                # now recrossing events are complete and we can remove them
                                if len(recross) > 0:
                                    segment_events = [segment_events[e] for e in range(len(segment_events)) if e not in recross]
                                    event_TSfids = [r.TS_fid for r in segment_events]
                                    if verbose:
                                        print ("remaining segments to be stored: %s" % str(event_TSfids))
                            ###### end of handle recrossings ####################################
                            # store events in DB now
                            for revt in segment_events:
                                if revt.unimol:
                                    self.store_uni_event(revt)
                                else:
                                    self.store_bim_event(revt)
                            # connect events (start with connecting last_event with the first in the segment)
                            if len(segment_events) > 0:
                                for revt in segment_events:
                                    self.connect_events(last_event, revt, verbose=verbose)
                                    last_event = revt
                            # now move forward again stating from segment_end
                            mode = "forward"
                            nextf = segment_end
                            # clear segment_events
                            segment_events = []
                            if verbose:
                                print ("#########################################################")
                                print ("all events in segment stored and connected .. moving forward again"   )


            currentf = nextf
            nextf = currentf+self.sstep[mode]-delta_segment_end
            delta_segment_end = 0
            # capture problems -- this should not happen
            if mode == "back":
                if nextf < segment_start:
                    back_delt = segment_start - nextf
                    if back_delt > self.skip_recross+1:
                        print ("segement_start is %d" % segment_start)
                        print ("nextf is %d" % nextf)
                        print ("backtracking went wrong -> abort")
                        raise
                    else:
                        if verbose:
                            print ("backtracking to a shifted segment bound (due to recross)")
                            print ("We shift nextf from %d to %d" % (nextf, segment_start))
                        nextf = segment_start
            if mode == "fine":
                if nextf > subseg_end:
                    print ("dinetracking went wrong -> abort")
                    raise 
            if verbose:
                sumform = self.frames[currentf].get_main_species_formula()
                print ("&& current %10d  next %10d  mode %8s   (current frame sumformula %s" % (currentf, nextf, mode, sumform))
            # check if the currentf has zero tracked species -> then stop searching
            if len(self.frames[currentf].specs) == 0:
                print ("Current frame %d has zero species to track ... stop searching")
                stop = True
        # end of mainloop

        self.rdb.commit()
        # DEBUG
        if self.nunconnected > 0:
            print ("NUMBER OF UNCONNECTED EVENTS: %d" % self.nunconnected)
            for r in self.unconnected:
                print ("%5d --> %5d" % r)
        return

    def store_initial_event(self):
        """for consistency we need an event for frame 0 just registering the initial tracked species as products

        We also need to add these to the self.unconn_species list for connections

        Note: we do not store any ED or TS species. 
        """
        f = self.frames[0]
        spec = list(f.specs.keys()) # get all tracked species
        # store reaction event
        revID = self.rdb.register_revent(
            -1,          # frame ID of TS is -1 (because 0 is the PR!)
            [],          # no ED
            [],          # no TS
            spec,        # spec IDs of "products"
            0,           # no educts
            len(spec),   # tracked species (all)
            []           # no rbonds
        )
        # make mol objects and store
        xyz = np.array(self.f_xyz[0])
        for s_id in spec:
            s = f.specs[s_id]
            m = s.make_mol(xyz[list(s.aids)], self.mol)
            self.rdb.add_md_species(
                revID,
                m,
                s_id,
                1,            # PR
                list(s.aids), 
                tracked=True
            )
            # add to unconnected list
            self.unconn_species.append(s)
        return

    @timer("store_bim_event")
    def store_bim_event(self, revt):
        """store species and further info in database

        Args:
            revt (revent object): reaction event object
        """
        # now we should have all data
        # generate mol objectes with coordinates
        xyz_ed = np.array(self.f_xyz[revt.ED.fid])
        xyz_ts = np.array(self.f_xyz[revt.TS.fid])
        xyz_pr = np.array(self.f_xyz[revt.PR.fid])
        ED_mol = []
        ED_aids = []
        ED_spec_tracked = []
        # ED/PR_spec are dicitionaries of species .. we need a sorted list of integers
        ED_spec_id = list(revt.ED_spec.keys())
        ED_spec_id.sort()
        for s in ED_spec_id:
            aids = list(revt.ED_spec[s].aids)
            ED_aids.append(aids)
            m = revt.ED_spec[s].make_mol(xyz_ed[aids], self.mol)
            ED_mol.append(m)
            if revt.ED_spec[s].tracked:
                ED_spec_tracked.append(s)
        PR_mol = []
        PR_aids = []
        PR_spec_tracked = []
        PR_spec_id = list(revt.PR_spec.keys())
        PR_spec_id.sort()
        for s in PR_spec_id:
            aids = list(revt.PR_spec[s].aids)
            PR_aids.append(aids)
            m = revt.PR_spec[s].make_mol(xyz_pr[aids], self.mol)
            PR_mol.append(m)
            if revt.PR_spec[s].tracked:
                PR_spec_tracked.append(s)
        # only one TS (all species) .. use make_mol of frame
        TS_spec_id = list(revt.TS_spec.keys())
        TS_spec_id.sort()
        TS_mol, TS_aids = revt.TS.make_mol(revt.TS_spec)
        # experimental make  reactive complex
        ED_react_compl, ED_react_compl_aids = revt.ED.make_mol(revt.ED_spec)
        PR_react_compl, PR_react_compl_aids = revt.PR.make_mol(revt.PR_spec)
        # map the broken/formed bonds to atom ids of the TS subsystem
        rbonds_global = revt.formed_bonds+revt.broken_bonds
        rbonds = []
        for b in rbonds_global:
            rbonds.append(TS_aids.index(b[0]))
            rbonds.append(TS_aids.index(b[1]))
        # now let us register the reaction event in the database
        revID = self.rdb.register_revent(
            revt.TS.fid,
            ED_spec_id,
            TS_spec_id,
            PR_spec_id,
            len(ED_spec_tracked),
            len(PR_spec_tracked),
            rbonds
        )
        # add the md species (mol objects etc) to this entry
        for i,s in enumerate(ED_spec_id):
            self.rdb.add_md_species(
                revID,
                ED_mol[i],
                s,
                -1,           # ED
                ED_aids[i],
                tracked= s in ED_spec_tracked
            )
        for i,s in enumerate(PR_spec_id):
            self.rdb.add_md_species(
                revID,
                PR_mol[i],
                s,
                1,            # PR
                PR_aids[i],
                tracked= s in PR_spec_tracked
            )
        self.rdb.add_md_species(
            revID,
            TS_mol,
            TS_spec_id[0],    # species of TS are stored in revent, here only first
            0,
            TS_aids
        )
        self.rdb.add_md_species(
            revID,
            ED_react_compl,
            ED_spec_id[0],    
            -1,
            ED_react_compl_aids,
            react_compl = True
        )
        self.rdb.add_md_species(
            revID,
            PR_react_compl,
            PR_spec_id[0],    
            1,
            PR_react_compl_aids,
            react_compl = True
        )
        return

    @timer("store_uni_event")
    def store_uni_event(self, revt):
        """store species and further info in database for an unimol reaction event
        (one ED/TS and PR species only)

        Args:
            revt (revent object): reaction event object
        """
        assert revt.unimol==True, "store_uni_event called for a bimolecular event!"
        # now we should have all data
        # generate mol objectes with coordinates
        xyz_ed = np.array(self.f_xyz[revt.ED.fid])
        xyz_ts = np.array(self.f_xyz[revt.TS.fid])
        xyz_pr = np.array(self.f_xyz[revt.PR.fid])
        # there must be only one species in ED, PR and TS
        # ED
        assert len(revt.ED_spec) == 1, "Found %s species, but only 1 expected" % (len(revt.ED_spec))
        ED_spec_id = next(iter(revt.ED_spec))
        ED_spec = revt.ED_spec[ED_spec_id]
        ED_mol = ED_spec.make_mol(xyz_ed[list(ED_spec.aids)], self.mol)
        # TS
        assert len(revt.TS_spec) == 1, "Found %s species, but only 1 expected" % (len(revt.TS_spec))
        TS_spec_id = next(iter(revt.TS_spec))
        TS_spec = revt.TS_spec[TS_spec_id]
        TS_mol = TS_spec.make_mol(xyz_ts[list(TS_spec.aids)], self.mol)
        # PR
        assert len(revt.PR_spec) == 1, "Found %s species, but only 1 expected" % (len(revt.PR_spec))
        PR_spec_id = next(iter(revt.PR_spec))
        PR_spec = revt.PR_spec[PR_spec_id]
        PR_mol = PR_spec.make_mol(xyz_pr[list(PR_spec.aids)], self.mol)
        # map the broken/formed bonds to atom ids of the TS subsystem
        rbonds_global = revt.formed_bonds+revt.broken_bonds
        rbonds = []
        TS_aids = list(TS_spec.aids)
        for b in rbonds_global:
            if (b[0] not in TS_aids) or (b[1] not in TS_aids):
                import pdb; pdb.set_trace() 
            rbonds.append(TS_aids.index(b[0]))
            rbonds.append(TS_aids.index(b[1]))
        # now let us register the unimol reaction event in the database
        revID = self.rdb.register_revent(
            revt.TS.fid,
            [ED_spec_id],
            [TS_spec_id],
            [PR_spec_id],
            1, # number of tracked ed species .. for uni always 1
            1, # number of tracked pr species .. for uni always 1
            rbonds,
            uni = True
        )
        # add the md species (mol objects etc) to this entry
        self.rdb.add_md_species(
            revID,
            ED_mol,
            ED_spec_id,
            -1,   # ED
            list(ED_spec.aids)
        )
        self.rdb.add_md_species(
            revID,
            PR_mol,
            PR_spec_id,
            1,            # PR
            list(PR_spec.aids)
        )
        self.rdb.add_md_species(
            revID,
            TS_mol,
            TS_spec_id,  
            0,
            list(TS_spec.aids)
        )
        return

    @timer("connect_events")
    def connect_events(self, revt_prev, revt, verbose=False):
        """new connector

        revt_prev is passed just for compatibility with the old connector and will not be used
        TBI: remove at some point and clean up search code.
        revt is the current event to be connected with the new algorithm
        TBI: connect already when storing events

        Args:
            revt_prev ([type]): previous reaction event
            revt ([type]): current event
            verbose (bool, optional): [description]. Defaults to False.
        """
        # loop over educt species in the current event
        # print ("DEBUG in unconnlist: %s" % self.unconn_species)
        for edsid in revt.ED_spec:
            eds = revt.ED_spec[edsid]
            if len(revt.ED_spec) != len(set(revt.ED_spec)):
                print("Educt list is not unique")
            if eds.tracked:
                # match with existing species in the unconnected list
                noccur = self.unconn_species.count(eds)
                if noccur != 1:
                    # GS: Now just print a Warning
                    print("WARNING: Possible problem in connect_events")
                    print(noccur)
                #    import pdb; pdb.set_trace()

                else:
                    indprs = self.unconn_species.index(eds)
                    prs = self.unconn_species[indprs]
                    # remove it from the unconn list
                    self.unconn_species.pop(indprs)
                    # now connect in the database from product to educt
                    # print ("connect %s with %s " % (prs, eds))
                    self.rdb.set_rgraph(prs.fid, eds.fid, prs.mid, eds.mid)
        # ok, now we have to add all the new tracked product species from this event to the unconn list
        for prsid in revt.PR_spec:
            prs = revt.PR_spec[prsid]
            if prs.tracked:
                self.unconn_species.append(prs)
        return

    @timer("connect_events_old")
    def connect_events_old(self, revt1, revt2, verbose=False):
        """connecting two reaction events 
        
        Args:
            revt1 (revent object): first reaction event (can be None to indicate initial frame)
            revt2 (revent obejct): second reaction event
        """
        if revt1 is None:
            fid1 = 0
        else:
            fid1 = revt1.TS_fid+1
        fid2 = revt2.TS_fid-1
        comparer = self.get_comparer(fid1, fid2)
        flag = comparer.check()
        # if the two events are side by side (revt2.TS_fid = revt1._TSfid+1) then fid1 > fid2
        #  => this will always lead to a fail ... this should never happen.
        # in this case we can not connect from PR1 to ED2. instead ED2 is TS1!
        if (fid1 >= fid2) or (flag != 0):
            if verbose:
                print ("##########################################################")
                print ("connecting between %d and %d" % (fid1, fid2))
                print ("products of revent 1: %s" % str(revt1.PR_spec))
                print ("educts of revent   2: %s" % str(revt2.ED_spec))
                print ("sumforms frame 1 %s" % str(self.frames[fid1].get_main_species_formula()))
                print ("sumforms frame 2 %s" % str(self.frames[fid2].get_main_species_formula()))
                print ("##########################################################")
        # assert flag == 0, "ED of revt2 not equal PR revt1 .. this should never happen"
        if flag != 0:
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print ("This should never happen!! Could not Connect!!")
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            import pdb; pdb.set_trace()
            self.nunconnected += 1
            self.unconnected.append((fid1, fid2))
            return # DEBUG DEBUG -- just ignore connection in this case
        if len(comparer.bond_match) == 0:
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print ("This is unexpected!!")
            print ("the comparer did not find any matching species")
            import pdb; pdb.set_trace()
            self.nunconnected += 1
            self.unconnected.append((fid1, fid2))
        for match in comparer.bond_match:
            print ("DEBUG: check this match between frame %5d (species %2d) and frame %5d (species%2d)" % (fid1, match[0], fid2, match[1]))
            if not self.frames[fid1].specs[match[0]] == self.frames[fid2].specs[match[1]]:
                import pdb; pdb.set_trace()
            self.rdb.set_rgraph(fid1, fid2, match[0], match[1])
        return

    ##########################  DEBUG stuff #################################################################

    def deb_check_every(self, every):
        """run over the frames and check for changes between every n frame
        
        Args:
            every (int): stride to check frames
        """
        self.process_frame(0)
        for i in range(every, self.nframes, every):
            self.process_frame(i)
            comp = self.get_comparer(i-every, i)
            flag = comp.check()
            taetae = ""
            if flag > 0:
                taetae = "!!!!!!!!"
            print("flag between %4d and %4d : %1d  %s" % (i-every, i, flag, taetae))
        return


    ########################## in case of restart ###########################################################

    @timer("store_frames")
    def store_frames(self, fname):
        f = open(fname, "wb")
        pickle.dump(self.frames, f)
        f.close()
        return

    @timer("load_frames")
    def load_frames(self, fname):
        f = open(fname, "rb")
        self.frames = pickle.load(f)
        f.close()
        return

    def report(self):
        self.timer.report()










