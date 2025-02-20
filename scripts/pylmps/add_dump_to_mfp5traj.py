#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

###########################################################################
#
#  Script to convert (old) BB (former SBU) files into the new format
#
############################################################################

import numpy 
import os
import shutil
import subprocess
import sys
import molsys
import h5py
from tqdm import tqdm

def write_cell_to_traj(traj,cell):
    tshape = traj['cell'].shape[0]
    traj['cell'].resize(tshape+1,axis=0)
    traj['cell'][-1,...] = cell
    mfp5.flush()
    return

def write_xyz_to_traj(traj,xyz):
    tshape = traj['xyz'].shape[0]
    traj['xyz'].resize(tshape+1,axis=0)
    traj['xyz'][-1,...] = xyz
    #traj['xyz'][-1,...] = m.get_xyz_from_frac(xyz)
    mfp5.flush()
    return


print('usage: [name] [dump] [mfp5] [stagename]')

if len(sys.argv) < 5:
    raise ValueError('Provide proper arguments!')



write_cell=True

name = sys.argv[1]
dumpf = sys.argv[2]
mfp5f = sys.argv[3]
stagename = sys.argv[4]

# /stage/restart/cell
# /stage/restart/vel
# /stage/restart/xyz

# /stage/traj/nstep
# /stage/traj/tstep
# /stage/traj/xyz

mb = molsys.mol()
mb.read(name+'.mfpx')

df = open(dumpf,'r')
natoms = mb.natoms 
xyz = numpy.zeros((natoms,3),dtype=numpy.float32)


mfp5 = h5py.File(mfp5f)
stage= mfp5.require_group(stagename)
traj = stage.require_group("traj")

traj.create_dataset('xyz', dtype='float32',shape=(0,natoms,3),maxshape=(None,natoms,3),chunks=(1,natoms,3))
traj.create_dataset('cell', dtype='float32',shape=(0,3,3),maxshape=(None,3,3),chunks=(1,3,3))


nsteps = 0

done=False
pbar = tqdm()
while not done:
    fl = df.readline()
    if fl == '':
        done=True
    if fl.count('ITEM: TIMESTEP') != 0:
        m=molsys.mol.from_fileobject(df,ftype='lammpstrj',triclinic=True)
        mb.set_xyz(m.get_xyz())
        mb.set_cell(m.get_cell())
        Vt = mb.get_volume()
        write_xyz_to_traj(traj,mb.get_xyz())
        write_cell_to_traj(traj,mb.get_cell())
        nsteps += 1
        pbar.update(1)


