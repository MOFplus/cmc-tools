# -*- coding: utf-8 -*-
### overload print in parallel case (needs to be the first line) [RS] ###
from __future__ import print_function

import numpy as np
import copy

#see also: https://stackoverflow.com/questions/23545404/should-i-use-a-descriptor-class-or-a-property-factory


def argsorted(seq, cmp=None, key=None, reverse=False, sort_flag=False):
    if key is None:
        argsorted = sorted(
            range(len(seq)), cmp=cmp, reverse=reverse,
            key=seq.__getitem__)
    else:
        argsorted = sorted(
            range(len(seq)), cmp=cmp, reverse=reverse,
            key=lambda x: key(seq.__getitem__(x)) )
    if sort_flag:
        seq.sort(cmp=cmp, key=key, reverse=reverse)
    return argsorted

class Property(list):
    def __init__(self, name, plen, ptype, default=-1):
        self.name = name
        self.plen = plen
        self.ptype = ptype
        self.default = default
        self.extend(update=False)
        return

    ### TBI [RA] ###
    #def __add__(self):
    #    return
    #def __delitem__(self):
    #    return
    #def __delslice__(self):
    #    return
    #def __getitem__(self):
    #    return
    #def __getslice__(self):
    #    return
    #def __setitem__(self):
    #    return
    #def __setslice__(self):
    #    return

    def __delitem__(self, index, update=True):
        if update:
            self.plen -= 1
        list.__delitem__(self, index)
        return

    def __delslice__(self, iindex, jindex, update=True):
        if update:
            self.plen -= jindex - iindex
        list.__delslice__(self, iindex, jindex)
        return

    def __add__(self, other):
        new = copy.deepcopy(self)
        new.extend(list(other))
        return new

    def __mul__(self, mul):
        new = copy.deepcopy(self)
        new.extend(list(self)*(mul-1))
        return new

    def __iadd__(self, other):
        self.extend(list(other))
        return self

    def __imul__(self, mul):
        self.extend(list(self)*(mul-1))
        return self

    def count(self, value=None):
        if value is None:
            # append with default
            value = self.default
        return list.count(self, value)

    def append(self, value=None, update=True):
        if value is None:
            # append with default
            value = self.default
        list.append(self, value)
        if update:
            self.plen += 1
        return

    def extend(self, values=None, update=True):
        if values is None:
            # fill missing values with default
            mlen = self.plen - len(self)
            values = mlen*[self.default]
        else:
            mlen = len(values)
        list.extend(self, values)
        if update:
            self.plen += mlen
        return
        
    def index(self, value=None):
        if value is None:
            # return first index of default
            value = self.default
        return list.index(self, value)

    def pop(self, index=None, update=True):
        if update:
            self.plen -= 1
        return list.pop(self, index=index)

    def remove(self, value, update=True):
        if update:
            self.plen -= 1
        list.remove(self, value)
        return

    def insert(self, index, value, update=True):
        if update:
            self.plen += 1
        list.insert(self, index, value)
        return

    def reverse(self, get_asort=True):
        if get_asort:
            asort = range(self.plen)
            asort.reverse()
        else:
            asort = None
        list.reverse(self)
        return asort

    def sort(self, get_asort=True, **kwargs):
        if get_asort:
            asort = argsorted(self, **kwargs)
        else:
            asort = None
        list.sort(self, **kwargs)
        return asort
