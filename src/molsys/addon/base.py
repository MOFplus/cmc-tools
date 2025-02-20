#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

class base(object):

    def __init__(self, mol):
        self._mol = mol
        self.mpi_comm = self._mol.mpi_comm
        self.mpi_rank = self._mol.mpi_rank 
        self.mpi_size = self._mol.mpi_size
        return

    def pprint(self,*args,**kwargs):
        return self._mol.pprint(*args,**kwargs)
