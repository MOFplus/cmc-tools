# -*- coding: utf-8 -*-

import numpy as np

# helper function
def rotate_cell(cell):
    if np.linalg.norm(cell[0]) != cell[0,0]:
        # system needs to be rotated
        A = cell[0]
        B = cell[1]
        C = cell[2]
        AcB = np.cross(A,B)
        uAcB = AcB/np.linalg.norm(AcB)
        lA = np.linalg.norm(A)
        uA = A/lA
        lx = lA
        xy = np.dot(B,uA)
        ly = np.linalg.norm(np.cross(uA,B))
        xz = np.dot(C,uA)
        yz = np.dot(C,np.cross(uAcB,uA))
        lz = np.dot(C,uAcB)
        cell = np.array([
                [lx,0,0],
                [xy,ly,0.0],
                [xz,yz,lz]])
    return cell

def get_header_line_count(fname):
	""" returns the number of blurb lines in a lammps log file before the acutal thermo data 	
	
	Args:
		fname (str): filename of the logfile
	
	Returns:
		list of str: column names
		int        : number of lines before the data
		int        : number of lines after  the data
	"""
	f = open(fname,'r')
	txt  = f.read().split('\n')
	n = 0
	header = []
	for i,l in enumerate(txt):
		n += 1
		if l.count("Step E_vdwl E_coul") != 0:
			header = l.split()
			break
	nb = 0
	for i,l in enumerate(txt[::-1]):
		nb += 1
		if l.count("WARNING") != 0:
			break
		elif l.count("Loop time of") != 0:
			break
	f.close()
	return header,n,nb