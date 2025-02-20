#!/usr/bin/env python3 -i
# -*- coding: utf-8 -*-

###########################################################################
#
#  Script to batch convert multiple lammps .dump trajectory files to hdf5
#  and store all of them in the same mfp5 file (must exist beforehand)
#
############################################################################

import numpy 
import os
import shutil
import subprocess
import sys
import molsys


backup_mfp5=True

dumps = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'dump']
mfpxs = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'mfpx']
mfp5s = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'mfp5']

if len(mfp5s) != 1:
    print('multiple or no mfp5(s) found, exiting... ')
    os._exit(0)

mfp5 = mfp5s[0]
if len(mfpxs) == 0:
    print('no mfpx found! exiting')
    os._exit(0)

if len(mfpxs) != 1:
    print('multiple mfpxs detected, select the one to be used')
    for i,mfpx in enumerate(mfpxs):
        print('%3d %s' % (i,mfpx))
    inp = input('select mfpx [int]')
    try: 
        i = int(inp)
    except:
        print('invalid input. give an integer!')
        os._exit(0)
    mfpx = mfpxs[i]
else:
    mfpx = mfpxs[0]

if backup_mfp5 is True:
    # do not backup if there is one already!
    if len([x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'mfp5_keep'])   == 0:
        os.system('cp %s %s_keep' % (mfp5,mfp5))
    
for dump in dumps:  
    name = mfpx.rsplit('.',1)[0]
    stagename = dump.rsplit('.',1)[0]
    s = 'add_dump_to_mfp5traj.py %s %s %s %s' % (name,dump,mfp5,stagename)
    os.system(s)
    print(s)
