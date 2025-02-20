#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 18:17:13 2017

@author: johannes
"""

import numpy
import string
from molsys.util.constants import angstrom, kcalmol

def read(mol, f, gradient = False, trajectory=False, cycle = -1):
    ### read check ###
    try:
        f.readline ### do nothing
    except AttributeError:
        raise IOError("%s is not readable" % f)
    ### read func ###
    if gradient:
        return read_gradfile(mol,f,cycle)
    if trajectory:
        return read_trajfile(mol,f)
    coord = False
    xyz = []
    elems = []
    for line in f:
        sline = line.split()
        if sline[0][0] == "$":
            if sline[0] == "$coord": 
                coord = True
                continue
            else:
                coord = False
                continue
        if coord:
            xyz.append([float(i) for i in sline[:3]])
            elems.append(sline[3])
    f.close()
    mol.natoms = len(elems)
    mol.xyz = numpy.array(xyz)/angstrom
    mol.elems = elems
    mol.atypes = elems
    mol.set_empty_conn()
    mol.set_nofrags()
    return

def read_trajfile_v(mol,f):
    """Read a turbomole MD log and get a mol instance including the trajectory
    
    (JK)
    The MD log file has to be generated via 'log2egy -g > foo.bar', which is a plain xyz file including the gradients as column 4-7
    IMPORTANT NOTE: Beware, development here has stopped due to the 'gradient' as they claim, actually is the velocity.
                    We need to grasp the gradient from the raw mdlog files!
    Args:
        mol ([molsys.mol]): [mol instance to write the data to]
        f ([type]): [file handle]
    """
    timestep = []
    elems    = []
    energy   = []
    xyz      = []
    grad     = []
    counter  = 0
    found    = False
    ### get how many cylces are in the file
    for i,l in enumerate(f.xreadlines()):
        if i==0:natoms=int(l); continue
        ls = [x for x in l.split() if x != '']
        if len(ls) == 1:
            counter += 1
            continue
        if ls.count('PE') != 0:
            energy.append(float(ls[1]))
            timestep.append(float(ls[4])/femtosecond)
            xyz.append([])
            grad.append([])
        else:
            xyz[counter].append([float(x) for x in ls[1:4]])
            grad[counter].append([float(x) for x in ls[5:]])
            if counter == 0: lems.append(ls[0])
    f.close()
    mol.natoms = natoms
    mol.xyz = numpy.array(xyz[0])/angstrom
    mol.elems = elems
    mol.atypes = elems
    mol.set_empty_conn()
    mol.set_nofrags()
    mol.gradient = numpy.array(grad)*(angstrom/kcalmol)
    mol.energy = energy/kcalmol

def read_trajfile(mol,f):
    """Read a raw turbomole MD log and get a mol instance including the trajectory
    
    (JK)
    
    Args:
        mol ([molsys.mol]): [mol instance to write the data to]
        f ([string]): [basename of the trajectory file]
    """
    import os
    trajfiles = [i for i in os.listdir('.') if (i.count(f) != 0 and i.count('py') == 0)]
    timestep = []
    elems    = []
    energy   = []
    xyz      = []
    vel      = []
    grad     = []
    counter  = 0
    found    = False
    ### get how many cylces are in the file
    for i_t, traj in enumerate(trajfiles):
        #print 'reading %s (%d/%d)' % (traj,i_t+1,len(trajfiles))
        f = open(traj,'r')
        f_text = f.read().replace('\n$end','').rsplit('$current',1)[0]
        frames = f_text.split('t=')[1:]
        frames[-1] = frames[-1].replace('\n$end','')
        f.close()
        for i_f,frame in enumerate(frames):
            natoms, elems, _xyz, _vel, _grad,_E = _read_single_traj_entry(frame)
            grad.append(_grad)
            xyz.append(_xyz)
            vel.append(_vel)
            energy.append(_E)
    xyz,vel,grad,energy = numpy.array(xyz), numpy.array(vel), numpy.array(grad), numpy.array(energy)
    mol.natoms = natoms
    mol.xyz = xyz[0]
    mol.trajectory  = xyz / angstrom
    mol.gradients  = grad / (kcalmol / angstrom)
    mol.velocities = vel / (1.0/2187.69)  ## v_au to angstrom/fs
    mol.energies = energy/kcalmol
    #print 'gradients  :    mol.gradients'
    #print 'trajectory :    mol.trajectory'
    #print 'velocities :    mol.velocities'
    #print 'energies   :    mol.energies'
    mol.elems = elems
    mol.atypes = elems
    mol.set_empty_conn()
    mol.set_nofrags()
    mol.gradient = numpy.array(grad)*(angstrom/kcalmol)
    return

def _read_single_traj_entry(trajtext):
    ts = trajtext.split('\n')[0:-1] ## chop off last element ( == '')
    natoms = (len(ts)-3) / 3
    elems = []
    t = float(ts[0])
    XYZ = ts[1:natoms+1]
    xyz = numpy.array([map(float,line.split()[0:-1]) for line in XYZ])
    elems = [line.split()[-1] for line in XYZ]
    VEL = ts[natoms+2:2*natoms+2]
    vel = numpy.array([map(float,line.split()[1:]) for line in VEL])
    E= float(ts[2*natoms+2].split()[1])
    GRAD= ts[2*natoms+3:]
    grad= numpy.array([map(float,line.split()) for line in GRAD])
    return natoms, elems, xyz, vel, grad,E


def read_gradfile(mol, f, cycle):
    elems    = []
    xyz      = []
    grad     = []
    ncycle   = 0
    found    = False
    ### get how many cylces are in the file
    for line in f:
        sline = line.split()
        if sline[0] == "cycle": ncycle += 1
    f.seek(0)
    scycle = range(ncycle)[cycle]
    for line in f:
        sline = line.split()
        if sline[0] == "cycle" and int(sline[2])-1 == scycle:
            ### found read in 
            energy = float(sline[6])
            found  = True
        elif sline[0] == "cycle" and int(sline[2])-1 != scycle:
            found = False
        if found:
            if len(sline) == 4:
                ### coord info
                xyz.append([float(i) for i in sline[:3]])
                elems.append(sline[3])
            elif len(sline) == 3:
                ### grad info
                #grad.append([lambda a: float(a.replace("D","E"))(i) for i in sline[:3]])
                grad.append([float(i.replace("D","E")) for i in sline[:3]])
    f.close()
    mol.natoms = len(elems)
    mol.xyz = numpy.array(xyz)/angstrom
    mol.elems = elems
    mol.atypes = elems
    mol.set_empty_conn()
    mol.set_nofrags()
    mol.gradient = numpy.array(grad)*(angstrom/kcalmol)
    mol.energy = energy/kcalmol
    return


def write(mol, f, gradient = None, energy = 0.0, cycle = 0):
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    if gradient is not None:
        write_gradient(mol, f, gradient, energy, cycle)
        return
    ### write func ###
    f.write("$coord\n")
    c = mol.xyz*angstrom
    for i in range(mol.natoms):
        f.write("  %19.14f %19.14f %19.14f   %-2s\n" % 
                (c[i,0],c[i,1], c[i,2], mol.elems[i]))
    f.write("$end\n")
    return

def write_gradient(mol, f, gradient, energy, cycle):
    ## TBI: make append mode (no cycle needed) : read until $end .. overwrite and add $end and circle +=1
    gradient *= (kcalmol/angstrom)
    energy   *= kcalmol
    gradnorm = numpy.linalg.norm(gradient)
    f.write("  cycle = %3d   SCF energy = %12.6f   |dE/dxyz| = %12.6f\n" % (cycle, energy, gradnorm))
    c = mol.xyz*angstrom
    for i in range(mol.natoms):
        f.write("  %19.14f %19.14f %19.14f   %-2s\n" % 
                (c[i,0],c[i,1], c[i,2], mol.elems[i]))
    for i in range(mol.natoms):
        line = "  %25.15G %25.15G %25.15G \n" % tuple(gradient[i])
        f.write(line.replace("e", "D"))
    return    


