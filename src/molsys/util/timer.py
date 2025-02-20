#########################################################################
#
# @file
#
# Created:            09-08-2021
#
# Author:             Daniel Oelert (add mail)
#
# Short Description:  Timer class to measure timings.
#
# Last modified:
#
# Copyright:
#
# CMC group, Ruhr-University Bochum 
# The code may only be used and/or copied with the written permission
# of the author or in accordance with the terms and conditions under
# which the program was supplied.  The code is provided "as is"
# without any expressed or implied warranty.
#
#########################################################################

from __future__ import print_function

import sys, time
import functools

from mpi4py import MPI
is_master = (MPI.COMM_WORLD.Get_rank() == 0)

NOT_STARTED = 0
RUNNING = 1
DONE = 2

class Timer(object):
    """Timer object.

    Use like this::

        timer = Timer("Name")
        timer.start()
        # do something
        timer.stop()

    or::

        with timer as mt:
            # do something

    To get a summary call::

        timer.report()

    """

    def __init__(self, desc : str):
        self.desc = desc
        self.status = NOT_STARTED
        self.children = {}
        self.time_accum = 0 
        self.count = 0
        self.t1 = 0
        self.t2 = 0

    @property
    def elapsed(self):
        if self.status == RUNNING:
            return self.time_accum + (time.time() - self.t1)
        return self.time_accum + (self.t2-self.t1)

    def start(self):
        #if self.status == RUNNING:
        #    raise RuntimeError("Timer is already running.")
        self.status = RUNNING
        self.t1 = time.time()
        self.count += 1
    
    def stop(self):
        self.t2 = time.time()
        if self.status == RUNNING:
           self.time_accum += time.time() - self.t1
           self.t1 = 0
           self.t2 = 0 
        self.status = DONE
    
    def reset(self):
        self.time_accum = 0
        self.t1 = 0
        self.t2 = 0

    def __call__(self, name):
        """Context manager for timing a block of code.

        Example (tim is a timer object)::

            with tim('Monitor block of code'):
                x = 2 + 2
        """
        #if not self.status == NOT_STARTED:
        #    self.start()
        #    # HACK: If the timer is started implicitly, it is not stopped on its own.
        #    #       This leads to overly long timings.
        new_timer = self.fork(name)
        # new_timer.start()
        return new_timer

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type:
            raise
        

    def report(self, indent="  ", out=sys.stdout):
        """report timings

        Args:
            indent (str, optional): [description]. Defaults to "  ".
            out ([type], optional): [description]. Defaults to sys.stdout.

        Returns:
            [type]: [description]
        """
        if not is_master: return
        rep = self._report()
        def rep2str(rep,elapsed,level=0):
            replist = []
            repstr = "| " + level*indent
            repstr += ("{:<"+str(39-level*len(indent))+"}").format(rep["desc"])
            repstr += " |{:<10}|".format("-"*int(rep["elapsed"]*10/self.elapsed))
            repstr += " {:>7}s  {:5.1f}%  {:5.1f}% |".format(
                "{:.6g}".format(rep["elapsed"])[:7],rep["elapsed"]/elapsed*100,rep["elapsed"]/self.elapsed*100
                )
            # HACK: There is a small error happening when truncating the elapsed time to a width of 7 characters without rounding here
            # HACK: No indication yet for running timers
            # if rep["status"] == RUNNING:
            #     repstr += " running ..."
            replist.append(repstr)
            for r in rep["children"]:
                replist += rep2str(r,rep["elapsed"],level=level+1)
            return replist
        #
        out.write("|"+"-"*79+"|\n")
        out.write("| Timer report"+39*" "+"| elapsed   rel.%   tot.%  |\n|"+"-"*79+"|\n")
        for i in rep2str(rep,self.elapsed):
            out.write(i+"\n")
        out.write("|"+"-"*79+"|\n")
        return
    
    def _report(self):
        reps = []
        if self.children:
            for key,ch in self.children.items():
                reps.append(ch._report())
        return {"status":self.status,"elapsed":self.elapsed,"desc":self.desc,"children":reps}

    def fork(self, desc : str, start=True):
        if self.status != RUNNING:
            #raise RuntimeError("Unable to fork timer that is not running.")
            self.start() 
        if not desc in self.children.keys():
            new_timer = Timer(desc)
            self.children[desc] = new_timer
        else:
            new_timer = self.children[desc]
        if start:
                new_timer.start() 
        return new_timer 


class timer:
    """Decorator for timing a method call.

    Example::

        from molsys.util.timer import timer, Timer

        class X:
            def __init__(self):
                self.timer = Timer('name')

            @timer('Add function')
            def add(self, a, b):
                return a + b

        """
    def __init__(self, name):
        self.name = name

    def __call__(self, method):
        @functools.wraps(method)
        def new_method(slf, *args, **kwargs):
            newtimer = slf.timer.fork(self.name)
            newtimer.start()
            x = method(slf, *args, **kwargs)
            try:
                newtimer.stop()
            except IndexError:
                pass
            return x
        return new_method

