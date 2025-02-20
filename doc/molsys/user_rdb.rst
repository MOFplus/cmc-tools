..

How to create a reaction database using 
========================================

The steps to create a database and analyse a trajectory of HDF5 format (.mfp5)
------------------------------------------------------------------------------
1) Initialize the database for a database named "rdb".

    init_RDB rdb

2) Find the reaction events and fill the database with the reactions from the MD simulations.

    python3 runRDB.py | tee runRDB.out

The runRDB.py for a trajectory named "reax.mfp5", with stage "sample", and database named "rdb" should look like this:

.. code-block:: python

    from molsys import findR
    fr = findR.findR("reax.mfp5","sample","rdb")
    fr.search_frames(verbose=True)
    fr.report()

3) Find the unique reaction events.

    find_unique_reactions rdb reax.mfp5 sample

4) Optimize the MD species, where the reference structures are those from the MD simulation and add them to the database.

opt_md_species rdb reax.mfp5 sample fromMD cmd.inp

where the cmd.inp file should look like this:

{ "calculator" : "reaxff"
, "lot"        : "ReaxFF"
}

5) Run the density functional theory calculations (under progress https://github.com/oyonder/molsys) taking the reference structures from e.g., ReaxFF level of theory.

    opt_dft_species rdb reax.mfp5 sample ReaxFF cmd-DFT.inp

where the cmd-DFT.inp file should look like this:

{ "calculator"                : "turbomole"
, "lot"                       : "ri-utpss/SVP"
, "data_storage_path"         : "/home/username/DATA/"
, "maxmem"                    : 3000
, "submit"                    : True
, "turbodir"                  : "/opt/software/user-soft/turbomole/75"
, "ntasks"                    : 16
, "maxnodes"                  : 4
, "partition"                 : "normal"
}
