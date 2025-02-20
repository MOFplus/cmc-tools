# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:09:01 2017

@author: rochus

          aftype

          a class for an aftype (atomtype and fragmenttype)

RS 2024 revision:
    appart from "full" aftypes we have truncated forms (only coord number or pure element) as well as wildcards
    for both atype and ftype

RS 2025 added feature:
    in addition to c3_c2h1 (full), c3 (truncated), c (pure) and wildcard * we add another level above truncated:
    c3_c2x1 (unspecified) with x as a placeholder for the unspecified neigbor element
    limitation: only one of the two aftype is allowed to have an unspecified element (typically this is the DB entry, whereas the real molecule has a full aftype)

    aftypes can be annotated with a % sign like c3_c3%b indicating the bridge carbon in naphtalene as an example 

    New: for afdict we need a hierarchy of reduced keys because if a complete match is not found, we need
    to check first if an unspecified entry is there, then we check for a truncated and then for a pure match.
    Always the most reduced entry in a tuple defines the level of the comparison.
    This is done in afdict by keeping several list which are searched in order.

    Comparison with __eq__ gives a True also if one side (or both) are truncated or wildcarded
    so there will be a match

    Note:
        af1 = aftype("c3_c2h1", "ph") af2 = aftype("c3", "ph")
        af1 == af2 gives True
        af1 is af2 gives False becasue the hash is different (need special dict to make that work) 

                         
    aftuple:
        is a tuple of aftypes including the ric type. aftuples can be compared (__eq__) according to truncated and wildcarded
        aftuples (see above)

        Note that for incomplete aftypes the canonical sorting for each ric type can not be used any more
        However, with permutations() aftuples have a method to return all possible permutations of the aftuple

    afdict:
        is a dict for aftuples. It additionally uses a list of permuted keys to lookup values. If a key is not found in the dict 
        and is incomplete (truncated or wildcarded aftypes) it will be
        checked against the list of keys. If it is found there the corresponding value will be returned. This is useful
        for truncated and wildcarded aftypes. 

    TODO: remove DEBUG print statements (currently commented out) if this stabilizes
         
"""

# generates the missing rich comparison methods
from functools import total_ordering
import itertools    
# import regular expressions module
import re

@total_ordering
class aftype(object):

    def __init__(self, atype, fragtype):
        self.is_aftype = True
        self.atype = atype
        self.fragtype = fragtype
        return

    @property
    def atype(self):
        return self._atype

    @atype.setter
    def atype(self, at):
        self._atype = at
        self._wild_at = False
        self._pure = False
        self._truncated = False
        self._unspecified = False
        # separate annotation
        self._annotated = False
        if "%" in at:
            self._annotated = True
            at, self._annotation = at.split("%")   
        # if this is a wildcard type we are done
        if at == "*":
            assert not self._annotated
            self._wild_at = True
        elif not "_" in at:
            assert not self._annotated
            m = re.search("[1-9]", at)
            # check for pure elem type like c
            if not m:
                # we have a pure elem type
                self._pure = True
                self._atype_pure = at
                self._atype_trunc = at
            else:
                self._truncated = True
                self._atype_trunc = at
                self._atype_pure = at[0:m.start()]
        else:
            # regular full (or unspecified) atype
            at, neighb = at.split("_")
            coord = int(at[-1])          
            m = re.findall("[1-9]", neighb)
            mi = [int(x) for x in m]
            assert sum(mi) == coord
            # check for unspecified type like c3_c2x1
            if "x" in neighb:
                assert not self._annotated
                self._unspecified = True
                self._atype_trunc = at
                self._atype_pure = at[0:-1]
            else:
                self._atype_trunc = at
                self._atype_pure = at[0:-1]
        return

    @property
    def fragtype(self):
        return self._fragtype

    @fragtype.setter
    def fragtype(self, ft):
        self._fragtype = ft
        self._wild_ft = False
        if ft == "*":
            self._wild_ft = True

    def __repr__(self):
        return "%s@%s" % (self._atype, self._fragtype)
  
    def __eq__(self, other):
        """aftype comparison

        RS 2024 revised comparison without "level" 
        we need to choose if aftypes are equal based on the level we have

        Args:
            other (aftype): other aftype

        Returns:
            bool: aftypes are equal
        """
        assert type(other) is aftype
        # compare frags .. if one of them is wild they always match
        if self._wild_ft or other._wild_ft:
            # one or both are *
            ft_equal = True
        else:
            ft_equal = (self._fragtype == other._fragtype)
        # compare atypes .. if one is * they always match
        if self._wild_at or other._wild_at:
            at_equal = True
        else:
            # if one (or both) are pure elements we compare on this level
            if self._pure or other._pure:
                at_equal = (self._atype_pure == other._atype_pure)
            else:
                # if one (or both) are truncated we compare on this level
                if self._truncated or other._truncated:
                    at_equal = (self._atype_trunc == other._atype_trunc)
                else:
                    if self._unspecified or other._unspecified:
                        if self._unspecified and other._unspecified:
                            at_equal = (self._atype == other._atype) # if both contain unspecified elemts compare the full atype
                        else:
                            # this is the most complex situation. we need to compare the truncated part and then check the elements
                            at_equal = (self._atype_trunc == other._atype_trunc)
                            # now we know that the coord number is equal .. just check if the specified elements match
                            if self._unspecified:
                                us_at, s_at = self._atype, other._atype
                            else:
                                us_at, s_at = other._atype, self._atype
                            us_at = us_at.split("_")[1]
                            s_at = s_at.split("_")[1]
                            pat = re.compile("[1-9]")
                            us_at = {k : int(v) for (k, v) in zip(pat.split(us_at)[:-1], pat.findall(us_at))}
                            s_at = {k : int(v) for (k, v) in zip(pat.split(s_at)[:-1], pat.findall(s_at))}
                            nx = us_at.pop("x")                      
                            for k in us_at.keys():                            
                                if k not in s_at:
                                    # all elements in us_at must be in s_at
                                    at_equal = False
                                    continue    
                                if us_at[k] != s_at[k]:
                                    # if the coord number is different we can check if it can be compensated by the unspecified element
                                    diff = s_at[k] - us_at[k]
                                    if diff <= nx:
                                        nx -= diff
                                    else:
                                        at_equal = False
                    else:
                        # both are full atypes NOTE: this includes any annotation
                        at_equal = (self._atype == other._atype)
        # the two are equal if both atypes and frags are equal
        return ft_equal and at_equal

    def __lt__(self, other):
        assert type(other) is aftype
        return ("%s@%s" % (self._atype, self._fragtype)) < ("%s@%s" % (other._atype, other._fragtype))

    def __gt__(self, other):
        assert type(other) is aftype
        return ("%s@%s" % (self._atype, self._fragtype)) > ("%s@%s" % (other._atype, other._fragtype))

    def __hash__(self):
        return hash("%s@%s" % (self._atype, self._fragtype))

class aftuple():

    """special tuple for aftypes
    
    tuple to store aftypes including ric type to properly sort them
    it also knows if all aftypes are complete

    use this for the param dict from the database (not the corresponding aftypes of the real molecules)
    """

    nbody = {"bnd": 2, "ang": 3, "dih": 4, "oop": 4, "cha": 1, "vdw": 1}

    def __init__(self, t, ric=None):
        assert ric in aftuple.nbody.keys()
        assert type(t) is tuple or type(t) is list and len(t) == aftuple.nbody[ric]
        self._ric = ric
        for a in t:
            assert type(a) is aftype
        self.t = t
        return
    
    def permutations(self):
        """return all permutations of the aftuple
        
        Note: we need to permute the aftuple if it is not complete
        """
        if self.all_complete:
            return [aftuple(self.t, ric=self.ric)]
        else:
            perms = [aftuple(self.t, ric=self.ric)]
            # we need to permute the aftuple
            if self._ric == "bnd":
                perms.append(aftuple([self.t[1], self.t[0]], ric="bnd"))
            elif self._ric == "ang":
                perms.append(aftuple([self.t[2], self.t[1], self.t[0]], ric="ang"))
            elif self._ric == "dih":
                perms.append(aftuple([self.t[3], self.t[2], self.t[1], self.t[0]], ric="dih"))
            elif self._ric == "oop":
                centa = self.t[0]
                perms += [aftuple([centa] + list(bcd), ric="oop") for bcd in list(itertools.permutations(self.t[1:]))[1:]]   
            return perms

    @property
    def ric(self):
        return self._ric
    
    @property
    def all_complete(self):
        for a in self.t:
            if (a._pure or a._truncated or a._wild_at or a._wild_ft or a._unspecified):
                return False
        return True
    
    def has_red_level(self, l):
        if l == "wild":
            for a in self.t:
                if (a._wild_at or a._wild_ft):
                    return True
            return False
        elif l == "pure":
            for a in self.t:
                if a._pure:
                    return True
            return False
        elif l == "trunc":
            for a in self.t:
                if a._truncated:
                    return True
            return False
        elif l == "unspec":
            for a in self.t:
                if a._unspecified:
                    return True
            return False
        else:
            raise ValueError("unknown level")
            
    # check this post ... https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr
    # RS (2024) I will implement both __repr__ and __str__ for aftuple
    # repr will get the ric prepended (depending on the context this info is redundant)
    # str will implement the "old" behaviour looking like a tuple
    def __repr__(self):
        ats = ",".join(f"{a}" for a in self.t)
        return f"{self._ric}({ats})"
    
    def __str__(self):
        ats = ",".join(f"{a}" for a in self.t)
        return f"({ats})"

    def __eq__(self, other):
        assert type(other) is aftuple
        if not self._ric == other._ric:
            return False
        for a, b in zip(self.t, other.t):
            if not a == b:
                return False
        return True
    
    def __hash__(self):
        return hash(str(self))

class afdict(dict):

    """special dict for aftypes

    this dict uses a list of the keys to lookup values, first a match on the hash basis is used but if the 
    key is found in the list it will also be pulled from the dict

    Note: we want to match also truncated and wildcarded aftypes in this dict, which works by using a list from the keys
          here the __eq__ method of the aftype class is used to compare the keys
          Status flags will be used to keep track if a truncated name was used or if the order of the types has been 
          inverted

    Problem: for the scrambling of the order we need to know what type of ric we are working on
             for oop and dih we have 4 atoms but a different way to sort.

             In case of setting new items we need to ignore "expansion" rules for truncated aftypes
             So __contains__ needs to check literally (if k in dict)
             In case of getting/reading from the dict these rules have to apply
             ==> we add a status flag that switches between the two modes for __contains__

    """

    levels = ["unspec", "trunc", "pure", "wild"]

    def __init__(self, *args, ric=None):
        assert ric in ["bnd", "ang", "dih", "oop", "cha", "vdw"]
        self._ric = ric
        self._keylist = {}
        for l in afdict.levels:
            self._keylist[l] = {}
        super().__init__(*args)
        # status switches
        self.retrieve_mode = False
        self.found_full = True  # this will be set to False for a truncated/wildcarded match (until the next __getitem__ call)
        self.reversed = False   # this will be set to True if the order of the aftuple was reverted (until the next __getitem__ call)
        self.replaced_key = None # this will be set to the key that was replaced by a permutation (until the next __getitem__ call)
        self.level = None # this will be set to the level of the key that was found (until the next __getitem__ call)
        return
    
    def set_retrieve_mode(self):
        self.retrieve_mode = True
        return  
    
    def unset_retrieve_mode(self):
        self.retrieve_mode = False
        return

    def __setitem__(self, k, v):
        assert self.retrieve_mode == False
        assert type(k) is aftuple
        assert k.ric == self._ric
        # print (f"DEBUG afdict setting {k} -> {v}")
        if not k.all_complete:
            # check at which level this key is
            sanity = False
            for l in reversed(afdict.levels):
                if k.has_red_level(l):
                    sanity = True
                    break
            assert sanity
            klist = self._keylist[l]                
            # we need to add permutations of the aftuple to the keylist
            if k not in klist:
                for pk in k.permutations():
                    klist[pk] = k
        super().__setitem__(k, v)
        return

    def __getitem__(self, k):
        # only for DEBUG .. remove again
        # print (f"DEBUG afdict searching for {k} : what we have is {list(super().keys())}") 
        self.found_full = True  # this will be set to False for a truncated/wildcarded match (until the next __getitem__ call)
        self.reversed = False   # this will be set to True if the order of the aftuple was reverted (until the next __getitem__ call)
        return super().__getitem__(k)

    def __missing__(self, k):
        # print ("DEBUG this is __missing__ in afdict")
        # print ("DEBUG we search for", k)
        self.found_full = False
        for l in afdict.levels:
            klist = self._keylist[l]
            # print ("DEBUG keys are", klist.keys())
            for kr in list(klist.keys()):
                # yes we have it but as a permutation
                if k == kr:
                    if self._ric in ["ang", "dih"]:
                        if hash(kr) != hash(klist[kr]): # Note: != does not work because how __eq__ is impl for aftuples (but this will always fail)
                            self.reversed = True
                    # print (f"DEBUG found as permutation {kr} reversed={self.reversed}")
                    # print (f"DEBUG afdict replace {k} with {klist[kr]} on level {l}")
                    self.replaced_key = klist[kr]
                    self.level = l                
                    return super().__getitem__(klist[kr])
        else:
            raise KeyError
        
    def __contains__(self, k):
        # print (f"DEBUG in afdict contains")
        if k in self.keys():
            # print ("DEBUG  .. found as such")
            return True
        elif self.retrieve_mode:
            # test if it is in the keylist for permuted truncated or wildcarded aftypes
            for l in afdict.levels:
                klist = self._keylist[l]
                # print (f"DEBUG searching for {k} in {list(klist.keys())} on level {l}")
                if k in list(klist.keys()):
                    # print ("DEBUG .. found in keylist")
                    return True 
            return False
        else:
            return False        


def aftype_sort(afl, ic):
    """
    helper function to sort a list of aftype objects according to the type (ic)

    Note: afl must be a list and not a tuple
    """
    if ic == "bnd":
        afl.sort()
    elif ic == "ang":
        if afl[0] > afl[2]: afl.reverse()
    elif ic == "dih":
        if afl[1] > afl[2]:
            afl.reverse()
        elif afl[1] == afl[2]:
            if afl[0] > afl[3]: afl.reverse()
    elif ic == "oop":
        plane = afl[1:]
        plane.sort()
        afl[1:] = plane
    elif ic == "vdwpr":
        return aftype_sort(afl, "bnd")
    return afl

