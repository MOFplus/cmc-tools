#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

import logging
logger = logging.getLogger('molsys.ff')

class api_cache(object):

    def __init__(self, api = None):
        self._api = api
        self._ref_dic = []
        self._ref_mol_strs = {}
        self._ref_params = {}
        self._special_atypes = {}
        return

    def list_FFrefs(self,ffname):
        if len(self._ref_dic)==0:
            self._ref_dic = self._api.list_FFrefs(ffname)
        return self._ref_dic

    def get_FFrefs_graph(self,scan_ref):
        if len(self._ref_mol_strs.keys()) == 0:
            self._ref_mol_strs  = self._api.get_FFrefs_graph(scan_ref, out ="str")
        return self._ref_mol_strs

    def get_ref_params(self, scan_ref, ffname):
        if len(self._ref_params.keys()) == 0:
            for ref in scan_ref:
                ref_par = self._api.get_params_from_ref(ffname, ref)
                self._ref_params[ref] = ref_par
        return self._ref_params
    
    def list_special_atypes(self):
        if len(self._ref_params.keys()) == 0:
            self._special_atypes = self._api.list_special_atypes()
        return self._special_atypes

class potentials(dict):
    """
    Class to store the parameter values, multiple ff objects can use the same par instance
    in order to perform multistruc fits
    """
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return

    def attach_variables(self):
        if hasattr(self, 'variables') == False:
            self.variables = varpars()

class varpar(object):

    """
    Class to hold information of parameters marked as variable in order to be fitted.
    """

    def __init__(self, par, name, val = 1.0, range = [0.0,2.0], bounds = ["z","i"]):
        assert len(bounds) == 2
        assert bounds[0] in ["h", "i", "z"] and bounds[1] in ["h", "i"]
        self._par     = par
        self.name    = name
        self._val     = val
        self.range   = range
        self.pos     = []
        self.bounds   = bounds

    def __repr__(self):
        """
        Returns objects name
        """
        return self.name

    def __call__(self, val = None):
        """
        Method to set a new value to the varpar oject and write this 
        into the ff.par dictionary.
        :Parameters:
            - val(float): new value of the varpar object
        """
        if val != None: self.val = val
        for i,p in enumerate(self.pos):
            ic, pottype, parindex  = p
            self._par[ic][pottype][1][parindex] = self.val
        return

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self,val):
        assert (type(val) == float) or (type(val) == np.float64) or (type(val) == type(""))
        if type(val) == type(""):     # NOTE this was val[0] == "$" but we want to allow general strings in case of yaml par files
            self._val = val
        else:
            self._val = float(val)    # make sure it is a pure float in case it is a number
        return

class varpars(dict):

    """
    Class inherited from dict, that holds all varpar objects
    """

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return

    def __setitem__(self,k,v):
        assert type(v) == varpar
        # loop over all items and check if the variable is already in there 
        if k in self.keys():
            k += 'd'
            v.name = k
            return self.__setitem__(k,v)
        else:
            super(varpars,self).__setitem__(k,v)
            return k

    def finditem(self, ic, pot, pos):
        """
        Method to find a varpar object by its position in the ff.par dictionary
        :Parameters:
            - ic(str):  ric, e.g. "bnd", "ang", "dih", "oop"
            - pot(str): name of the potential (eg. "mm3->(c3_c1o2@co2,o2_c1cu1@co2)|CuPW")
            - pos(int): position of the parameter 
        :Returns:   
            - (str, int): name of the varpar object and the index of the entry in the pos list (first is the original ref and the follwoing are references in cross terms)
        """
        for k,v in self.items():
            for j, p in enumerate(v.pos):
                if p[0] == ic and p[1] == pot and p[2] == pos:
                    return k, j
        return False, -1

    @property
    def ranges(self):
        ranges = []
        for k,v in self.items():
            # ranges.append(float(v.range)) # seems wrong .. range is a list
            ranges.append(v.range)
        return np.array(ranges)

    @ranges.setter
    def ranges(self, ranges):
        assert len(ranges) == len(self.keys())
        for i,v in enumerate(self.values()):
            v.range = ranges[i]

    @property
    def vals(self):
        vals   = []
        for k,v in self.items():
            vals.append(float(v.val))
        return vals

    def cleanup(self):
        """
        Method to delete all unsused varpar objects
        """
        rem = []
        for k,v in self.items():
            if len(v.pos) == 0:
                logger.warning("varpar %s is not used --> will be deleted!" % k)
                rem.append(k)
        for k in rem: del[self[k]]
        return

    @property
    def varpots(self):
        varpots = []
        for k,v in self.items():
            for i in range(len(v.pos)):
                varpot = (v.pos[i][0], v.pos[i][1])
                if varpot not in varpots: varpots.append(varpot)
        return varpots

    def varpotnums(self,ff):
        """
        Property which gives a dictionary telling in how much terms a
        varpot is involved
        """
#        ff = self.values()[0]._par
        varpots = self.varpots
        varpotnums = {}
        for i in varpots: varpotnums[i] = 0
        ics = ["bnd", "ang", "dih", "oop"]
        for ic in ics:
            for pi in ff.parind[ic]:
                for p in pi:
                    if (ic,p) in varpots: varpotnums[(ic,p)]+=1
        return varpotnums


    def __call__(self, vals = None):
        """
        Method to write new values to the varpar objects and in the 
        ff.par dictionary
        :Parameters:
            -vals(list of floats): list holding the new parameters 
        """
        if type(vals) == type(None): vals = len(self)*[None]
        assert len(vals) == len(self)
        for i,v in enumerate(self.values()): v(vals[i])
        return

    def report(self):
        """
        Method to print a report of the varpars
        """
        for k,v in self.items():
            print (f"Var: {k:20s} : val = {v.val:10.3f} range = {v.range[0]:10.3f} - {v.range[1]:10.3f} bounds = {v.bounds[0]} - {v.bounds[1]}")
            for p in v.pos:
                print (f"  {p[0]:3s} {p[1]:30s} {p[2]:3d}")
        return
