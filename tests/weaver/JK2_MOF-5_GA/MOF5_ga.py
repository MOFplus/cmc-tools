import pytest
import os
import weaver
import mpi4py

from weaver.RTA.genetic import Environment
from weaver import RTA as RTA
from molsys.molsys_mpi import mpiobject

mpi = mpiobject()

def score(chrome, dummy):
    return sum(chrome)

f = weaver.framework('MOF-5')
f.read_topo('pcu.mfpx')
f.net.make_supercell([2,2,2])
f.add_linker_vertex('1')
f.assign_bb('0','Zn4O.mfpx')
f.assign_bb('1','bdc.mfpx',linker=True)
f.autoscale_net()
if os.listdir('.').count('MOF5.orients') != 0: # check if orients file has already been generated
    f.read_orientations('MOF5.orients')        # if so, reuse it
else:
    f.scan_orientations(20)
    f.write_orientations('MOF5.orients')


rta = RTA.RTA(f) # initialize everything to run the genetic algorithm
rta.run(popsize=32,maxgen=1000) # run max 1000 generations of a world with 32 individuals



