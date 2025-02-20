#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
basic mpiobject

@author: johannes
"""
### overload print in parallel case (needs to be the first line) [RS] ###
from __future__ import print_function

import sys
import _io

    
# load a global comm and general info on MPI for logger and other things
# Note that all classes are derived from mpiobject and use theri own communicator
#      which can be different from the world comm
try:
    from mpi4py import MPI
    wcomm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    err = None
except ImportError as e:
    wcomm = None
    size = 1
    rank = 0
    err = e

# overload print function in parallel case
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__
def print(*args, **kwargs):
    if rank == 0:
        return __builtin__.print(*args, **kwargs)
    else:
        return



class mpiobject(object):
    """Basic class to handle parallel attributes
    
    :Parameters:
    - mpi_comm(obj): Intracomm object for parallelization
    - out(str): output file name. If None: stdout
    """
    def __init__(self, mpi_comm = None, out = None):
        if mpi_comm is None:
            try:
                self.mpi_comm = MPI.COMM_WORLD
            except NameError:
                self.mpi_comm = None
        else:
            self.mpi_comm = mpi_comm
        try:
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()
        except AttributeError:
            self.mpi_rank = 0
            self.mpi_size = 1
        if out is None:
            self.out = sys.stdout
        elif type(out) == _io.TextIOWrapper:
            assert out.mode == 'w'
            self.out = out
        else:
            self.out = open(out, "w")

    def __delitem__(self, name):
        self.__delattr__(name)
        return 

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)
        return

    def pprint(self, *args, **kwargs):
        """Parallel print function"""
        if self.is_master:
            __builtin__.print(*args, file=self.out, **kwargs)
            self.out.flush()

    @property
    def is_master(self):
        """
        The mpi process with global rank 0 is always the master rank.
        This methods returns True if the current process has rank 0, else
        False
        """
        return self.mpi_rank == 0


    def __getstate__(self):
        """Get state for pickle and pickle-based method (e.g. copy.deepcopy)
        Meant for python3 forward-compatibility.
        Files (and standard output/error as well) are _io.TextIOWrapper
        in python3, and thus they are not pickle-able. A dedicated method for
        files is needed.
        An example: https://docs.python.org/2.0/lib/pickle-example.html

        N.B.: experimental for "out" != sys.stdout
        """
        newone = type(self)()
        newdict = newone.__dict__
        newdict.update(self.__dict__)
        newdict["out.name"] = newdict["out"].name
        newdict["out.mode"] = newdict["out"].mode
        newdict["out.encoding"] = newdict["out"].encoding
        del newdict["out"]
        return newdict

    def __setstate__(self, stored_dict):
        """Set state for pickle and pickle-based method (e.g. copy.deepcopy)
        For python3 forward-compatibility
        Whatever comes out of getstate, goes int setstate.
        https://stackoverflow.com/a/41754104

        N.B.: experimental for "out" != sys.stdout
        """
        if stored_dict["out.name"] == '<stdout>':
            stored_dict["out"] = sys.stdout
        self.__dict__ = stored_dict
