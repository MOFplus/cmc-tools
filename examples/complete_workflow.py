"""
        complete workflow

        in this example we perform a complete workflow for an MD of HKUST-1 with the
        following steps:

        1.) download structure from MOF+ via api
        2.) atomtype and fragmentize the system
        3.) assign force field parameters
        4.) optimize the structure with pylmps/lammps
        5.) run a short MD simulation recording a trajectory in an mfp5 file

        Note: Some steps can easier be performed on the commandline with scripts provided.
        her we perform all the steps in one single python code file for illustration

        RS 2021

"""

# step 1 : download mfpx file from MOF+ ... HKUST1 has structureID 2 (in the database)

import mofplus

api = mofplus.user_api()
m = api.get_mof_structure_by_id(2, out="mol")   # without out="mol" it writes an mfpx file 
m.write("hkust1_download.mfpx")

# step 2 : atomtype and fragmentize (fragmentize needs access to MOF+ again)
#          -> use the fragmentize script to do that for a mfpx file on the command line

import molsys

at = molsys.util.atomtyper(m)
at()
frag = molsys.util.fragmentizer()
frag(m)
m.write("hkust1.mfpx")

# step 3 : assign MOF-FF paramters and save as ric/par
#          -> use the query_parameters script in mofplus to do it on the command line

m.addon("ff")
m.ff.assign_params("MOF-FF")
m.ff.write("hkust1")

# step 4 : optimize with pylmps

import pylmps

pl = pylmps.pylmps("hkust1")
pl.setup(mol=m, kspace=True, use_mfp5=True)
pl.MIN(0.1)
pl.write("hkust1_opt.mfpx")

# step 5 : simple MD simulation in NVT

pl.MD_init("test", startup=True, T=300.0, ensemble="nvt", thermo="ber", relax=(0.1,), log=False, dump=False)
pl.MD_run(10000)

pl.end()

# to visualize the trajectory compile the VMD-tools in molsys and use vmd -mfp5 hkust1.mfp5 to see it.
# Also the thermo data can be found in the mfp5 file




