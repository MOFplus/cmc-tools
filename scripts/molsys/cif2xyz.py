#!/usr/bin/env python3
################################################################################
### cif2xyz file converter based on openbabel, script by RS ####################
################################################################################
import sys
import os
import subprocess

def float_w_braces(num):
    num = num.split('(')[0]
    return float(num)

name = sys.argv[1]
if len(sys.argv)>2:
    outname = sys.argv[2]
else:
    outname = name[:-4]+".txyz"
tempname = "temp.xyz"
    

try:
    subprocess.call(["obabel", '-icif', name, '-oxyz', '-O'+tempname, '-UCfill', 'keepconnect'])
except OSError:
    raise OSError("install openbabel - Chemical toolbox utilities (cli)")

# read cif to get cell params
f = open(name, "r")
line = f.readline()
cell = {}
while len(line)>0:
    sline = line.split()
    if len(sline)>1:
        keyw = sline[0] 
        value = sline[1]
        if keyw[:12] == "_cell_length":
            cell[keyw[13]] = float_w_braces(value)
        if keyw[:11] == "_cell_angle":
            cell[keyw[12:]] = float_w_braces(value)
    line = f.readline()
f.close()

cellparams = "%10.5f %10.5f %10.5f     %10.5f %10.5f %10.5f" % (cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])

try:
    subprocess.call(["x2x", "-i", tempname ,"-o", outname, "-c", cellparams])
except OSError:
    raise OSError("get x2x from molsys repo")
finally:
    os.remove(tempname)
