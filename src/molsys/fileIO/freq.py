#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: julian  
Read (and write?) molden .freq files
"""

import numpy
import string
from molsys.util.constants import angstrom, kcalmol

def read(mol, f):
    # missing docstring!
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    xyz,elems,eigval,eigvec = [],[],[],[]
    stage,nm_idx = 'Null', -1
    for i,c in enumerate(f):
        if c == '': continue
        if c.count('Atoms') != 0:
            stage = 'Atoms';continue
        elif c.count('FREQ') != 0:
            stage ='Freqs';continue
        elif c.count('FR-COORD') != 0:
            stage = 'Setup_nmarray';continue
        elif c.count('vibration') != 0:
            stage = [i for i in c.split() if i != ''][-1];continue
            ### stage contains then the normal mode index
        else:
            pass

        ###
        if stage == 'Null': continue
        ###atoms
        if stage == 'Atoms':
            sline = [i for i in c.split() if i != '']
            elems.append(sline[0])
            xyz.append([float(i) for i in sline[3:]])
        
        elif stage == 'Freqs':
            eigval.append(float(c.replace(' ','')))
        
        elif stage == 'Setup_nmarray':
            eigvec = numpy.zeros((len(elems),len(eigval),3),dtype='float')
        else:  # in any case this is now an eigenvector!
            if nm_idx != int(stage) -1: 
                nm_idx = int(stage)-1
                nmcount = 0
            nm_idx = int(stage)-1
            eigvec[nmcount,nm_idx,:] = numpy.array([float(i) for i in c.split() if c != ''])
            nmcount += 1
    mol.natoms = len(elems)
    mol.xyz = numpy.array(xyz)/angstrom
    mol.elems = elems
    mol.atypes = elems
    mol.frequencies = eigval
    mol.normalmodes = eigvec    
    mol.set_empty_conn()
    mol.set_nofrags()
    return
