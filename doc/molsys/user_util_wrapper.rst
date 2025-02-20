.. molsys documentation master file, created by
   sphinx-quickstart on Mon Aug 21 14:29:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MM Program Interfaces
#####################

Introduction
============
As already explained it is possible to assign force field parameters to a molecular system of
interest by using the FF addon. In addition MOLSYS features interfaces to several molecular
mechanis programms which makes it easy to run molecular dynamics simulations of the system
at the basis of the assigned FF. In this chapter it is explained how to use these interfaces.
Up to know interfaces to PYDLPOLY and PYLMPS are implemented. Setting up an instance of PYDLPOLY
or PYLMPS based on MOLSYS is quite easy. Two ways are possible. 

In the first way one creates a mol object with a FF addon attached, which is then used to initialize
PYDLPOLY/PYLMPS. For PYLPOLY this has to be done as follows:

.. code-block:: python
    :linenos:

    >> import pydlpoly
    >> import molsys
    >> m = molsys.mol.fromFile("HKUST-1.mfpx")
    >> m.addon("ff")
    >> m.ff.read("HKUST-1")
    >> pd = pydlpoly.pydlpoly("HKUST-1")
    >> pd.setup(mol = m)

For PYLMPS it works totally the same:

.. code-block:: python
    :linenos:

    >> import pylmps
    >> import molsys
    >> m = molsys.mol.fromFile("HKUST-1.mfpx")
    >> m.addon("ff")
    >> m.ff.read("HKUST-1")
    >> pd = pylmps.pylmps("HKUST-1")
    >> pd.setup(mol = m)

It is also possbile to startup PYDLPOLY/PYLMPS directly from the .fpar and .ric files.
For this purpose please follow the example below:

.. code-block:: python
    :linenos:

    >> import pylmps
    >> pd = pylmps.pylmps("HKUST-1")
    >> pd.setup(xyz="HKUST-1.mfpx", par="HKUST-1")

In addition it is possible to start PYLMPS/PYDLPOLY from the MOFplus database:

.. code-block:: python
    :linenos:

    >> import pylmps
    >> pd = pylmps.pylmps("HKUST-1")
    >> pd.setup(xyz="HKUST-1.mfpx", ff="MOF-FF")


LAMMPS Interface
================
The LAMMPS interface is implemented in the ff2lammps module in the PYLMPS package. It can be used
to create easily a PYLMPS object for your system of interest as explained above. In addition it can
be used to generate usual LAMMPS input files in oder to start LAMMPS from the command line. 
This is shown in the example below:

.. code-block:: python
    :linenos:

    >> import molsys
    >> from pylmps import ff2lammps
    >> m = molsys.mol.fromFile("HKUST-1.mfpx")
    >> m.addon("ff")
    >> m.ff.read("HKUST-1")
    >> ff2lmp = ff2lammps.ff2lammps(m)
    >> ff2lmp.write_data('hkust1.data')
    >> ff2lmp.write_input('hkust1.in')

