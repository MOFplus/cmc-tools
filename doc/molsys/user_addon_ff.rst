.. molsys documentation master file, created by
   sphinx-quickstart on Mon Aug 21 14:29:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


The FF addon
############

Introduction
============

The FF addon of molsys makes it easy to assign force field (FF) parameters to a molecular system of interest.
For this purpose the FF addon is closely interlocked to the `MOFplus webpage <https://www.mofplus.org>`_ and its 
internal FF parameter database. The FF addon communicates with the MOFplus server via the MOFplus API 
:py:class:`mofplus.ff.FF_api`

The FF addon features a novel FF parameter assignment procedure which has been developed especially for the use of *ab initio*
parameterized FFs.

Assignment Scheme
=================

General Idea
------------

In building block based *ab initio* parameterized FFs, specific paramters for individual systems are generated in a
systematic manner with controlled accuracy, using reference data computed of first principles level of theory
for non-periodic model systems. For the description of periodic structures the building block specific parameters have
to be combined in the correct manner. This so called assignment problem discourages non-expert users from using more
accurate first principles parameterized FFs. The FF addon implements an algortithm which solves the assignment
problem without any user input. For this purpose graph-theory is employed to analyze the molecular system of interest and
to find the best matching parameters. The search for the best matching parameters is based on the idea that the quality
of a parameter set depends on the size of the underlying model system. This ensure that always the most accurate 
parameters are assigned.

Requirement for the assignment procedure is a valid connectivity of the molecular system of interest. On the basis of the
connectivity the internal coordinates (bonds, angles, dihedrals, out-of-planes) are detected. Target of the assignment
process is, that for every ic found a parameter set consisting of a potential type and the corresponding parameters is
assigned.

During the assignment process the molecular system of interest is analyzed by three different specifiers, based on the
connectivity of the system. The first one is the atomtype, it describes the local bonding situation of an 
atom and consists of three different informations. The first one is the element of the atom, the second one is the
coordination number and the third one are the elements of thee connected atoms. For example the sp2 Carbon atom in
benzene gets the type c3_c2h1 and the Hydrogen atom gets h1_c1.

The second one is the so called fragmenttype. In comparison to the atomtype the fragmenttype specifier allows a
coarser view on the structure. Fragment types are assigned on the basis of a predefined fragment catalogue which is
constantly extended. In addition to the corresponding fragmenttype specifier, every atom gets a thrid specifier, specifying
the index of the corresponding fragment in the molecular structure of interest. In a biphenyl molecule all atoms would
get as fragmenttype "ph" ("ph for phenyl) but their fragmentnumbers would differ, atoms belonging to the first phenyl would
get the 1 and atoms belonging to the second one would get the two. By the help of the fragment information it is possible
to setup a fragment connectivity.

In the next step the molelcular structure of interest is analyzed in repect to the available reference systems for a
given forcefield. It is not required that an atom belongs only to one reference system, it can belong to several ones.
For the found reference systems the avaialable parameter set are downloaded directly from the MOFplus server. These are then
assigned to the molecular system of interest following the general assumtion that the quality of a parameter set increases
with the site of the underlying reference system. For this purpose it is looped over the found reference systems, starting
in decreasing order. The size of a reference system is defined by the number fragments it consists of. In an inner loop
it is iterated over all ics of the system of interest. If the ic defining atoms belong to the current reference system a lookup
for the a parameter set of the ic in the current reference system is performed. If a parameter set is found it will be assigned
to the ic. At the end of this procedure a parameter set has hopefully been assigned to every ic, else the requested FF is
not valid for the molecular system of interest. 

This procedure is illustrated at the example of biphenyl.
In principle two different parameterization strategies for this molecule are possible. The first one is the classical building
block approach, there the parameters for biphenyl are obtained in a two step procedure. First a FF for benzene is parameterized
and afterwards the missing parameters for Biphenyl are received. Consequently parametersets belonging to two reference systems
are needed. The second parameterization strategy would be the brute-force alternative. All parameters are directly obtained in
respect to the biphenyl reference system. Both parameterization stragegies have their qualification, the first one features a 
higher transferability whereas a more accurate description of biphenyl is expected from the second ansatz. 

The implemented assignment procedure can handle both situations. It would detect both reference systems in our molecular system
of interest (biphenyl). The assignment loop starts from the larger refererence system i.e. Biphenyl and distributes its parameters.
If the parameterization is done in the brute-force way, the parameters distributed form the biphenyl reference system
will already be sufficient. If the parameterization is done following the building block appproach the parameters from the biphenyl
reference system will not be sufficient, consequently the missing parameters will come from the Benzene reference system.
In this way always in a dynamic fashion always the most accurate parameters are assigned.

Further Feautures
-----------------
In order to be able to use the oberall power of the algorithm it is necessary to understand the following more detailed
concepts.

For some reference systems so called active zones (**azone**) are defined. An **azone** comprises those atoms of a reference
systems which belongs to at least one ic which has been parameterized on the basis of this reference system. Parameters on the basis
of this reference system are then only distributed to those ics which incule at least one atom of the **azone** of the 
reference system. This mechanism is required in order to prevent an erroneousy parameter distribution. (beispiel fehlt noch)

A trick implemented in order to make the algorithm more flexible is to make use of so called equivalences (**equivs**).
This feature is again illustrated employing Biphenyl as example system. As already mentioned all parameters are stored in respect
to atom and fragmentype of the in the ic ivolved atoms. In Biphenyl the Carbons forming the phenyl-phenyl bond have the atomtype
c3_c3. Since this atomtype is not present in the Benzene reference system all parameters for ics including atomtype c3_c3 has
to be obtained from the biphenyl reference system. In order to increase the transferability of the parameters it is possible to define
connectivity dependent atomtype equivalences, which are always applied after the assignment loop has distributed all parameters from the
current reference system. So it could be defined that all c3_c3@ph atom in the bond c3_c3@ph:c3_c3@ph are altered to the atomtype 
c3_c2h1@ph.This could of course also be performed on the basis of any other ic. As consequence more parameters from the 
Benzene referece system could be applied.

A further valuable possibility is making use of the **atfix** feature ...

By the help of the **upgrade** feature reference systems can be upgraded in order to increase their transferability ...


Technical Details
=================
Lorem ipsum

File IO
-------
The assigned FF can also be written to harddisk by the help of two different filetypes, which should be explained in the section. For this
purpose we take in the first place Benzene as example. The first file type is the so called ric file. It defines all ic of the system and links
them to the corresponding parameter sets. A ric file has 6 sections:

    1. **bnd**: In this section all bonds of the system are defined by two atom indices per bond.
    2. **ang**: In this section all angles of the system are defined by three atom indices per angle.
    3. **dih**: In this section all dihedrals of the system are defined by four atom indices per dihedral.
    4. **oop**: In this section all out-of plane bendings of the system are defined by four atom indices per oop.
    5. **cha**: In tis section all charges of the system are defined by one atom index per charge.
    6. **vdw**: In this section all vdW ics of the system are defined by one atom index per vdW. 

A typical section is composed as follows. The first line comprises of the section's name and the number of
corresponding ics. For the **bnd** section of benzene this looks as follows:

.. literalinclude:: _static/benzene.ric
    :language: txt
    :lines: 1

Then as many lines as ics has to follow. The first number in each line is an index running from 1 to the number
of specified ics. The next number is a pointer to a potential defined in the par file, and the last numbers define the 
ic. The whole **bnd** blocks for benzene looks as follows. The highlited line defines a bond between the atoms 1 and 2,
and assigns the potential with index 1 to it.

.. literalinclude:: _static/benzene.ric
    :language: txt
    :emphasize-lines: 2
    :lines: 1-12

The actual potentials are defined in the par file. The first line states always the name of the FF. In our example
this is MOF-FF.

.. literalinclude:: _static/benzene.par
    :language: txt
    :lines: 1

This header line is followed by at least six sections:

    1. **bnd_type**: In this section all stretch potentials are defined 
    2. **ang_type**: In this section all bending potentials are defined
    3. **dih_type**: In this section all torsion potentials are defined
    4. **oop_type**: In this section all out-of plane bending potentials are defined
    5. **cha_type**: In tis section all Coulomb potentials are defined
    6. **vdw_type**: In this section all vdW potentials are defined

A typical section is composed as follows. The first line comprises of the sections's name and the number
of defined potentials. Then as many lines as defined potentials follow. In the benzene model only two diffrent
stretch potentials are defined. In these lines the first number is the index of the potential, the second number in
a line defining an ic in the ric file refers to this index. This index is followed by a string defining the type 
of the potential, and then the actual parameters come. At last a string comes which defines the name and the source
of the parameter set.

.. literalinclude:: _static/benzene.par
    :language: txt
    :emphasize-lines: 2
    :lines: 13-15

If the par file should be used to set up a parameterization run with the FFgenerator code, the file containing the 
parameter sets is not longer called par file, instead it got the ending fpar. In order to mark a parameter as variable
the corresponding floating point number is replaced by a string starting with **$**. In addition the initial value together
with boundary information has to be specified. How this is done is illustrated on the example of biphenyl. An excerpt
of its fpar file is shown.

.. literalinclude:: _static/ph-ph.fpar
    :language: txt
    :linenos:
    :lines: 1-2, 13-15, 81-86,98-99

In line 5 two parameters of the bond potential with index 2 are marked as variable. These variables are then
defined in the added **variables** section, starting at line 11. The **variables** keyword is followed by a number
indicating the amount of variables which should be defined. In the following every line defines one variable.
The string defines the name of the variable, and is followed by its initial value. Then the upper and lower
bound are specified. The last two strings define the nature of the bounds, an **h** stands for a hard margin, which means
that it is not allowed that the variable gets beyond this tressholt, an **i** indicates a weak margin.

In addition two lines differing from the standard par file occur. Line 7 defines the active zone of the system,
whereas line 9 specify the name of the new reference system. Both lines are required.


Examples
========
The files and scripts needed to exploit the following examples can be found in the examples subdirectory of your molsys
distribution.

Assignment
----------
The first example shows how to use the assignment scheme to setup a FF for a molecular system. The MOF HKUST-1 is chosen
as molecular system of interest. At first a mol object of HKUST-1 has to be generated together with an ff addon instance
alongside.

.. code-block:: python
    
    >> import molsys
    >> m = molsys.mol.fromFile("HKUST-1.mfpx")
    >> m.addon("ff")

To assign the parameters for the desired FF (here MOF-FF), the following code has to be executed:

.. code-block:: python

    >> m.ff.assign_params("MOF-FF")

If no AssignmentError is raised, the assignment process has been successful and the mol object can be used for simulations
using LAMMPS or PYDLPOLY.

In order to skip the assignment loop and to read the FF from a file use the following command. It is assumed that a file
HKUST-1.par and HKUST-1.ric is available.

.. code-block:: python

    >> m.ff.read("HKUST-1")

To write the FF to a file use the write procedure:

.. code-block:: python

    >> m.ff.write("HKUST-1")

This create the files HKUST-1.ric and HKUST-1.par.


Preparation of a parameterization run
--------------------------------------

As already mentioned, the FF addon can also used for the preparation of ric and fpar file for a parameterization run. The big
advantage is, that the already existing parameters are automatically assigned. We will demonstrate this feature using Biphenyl as
example. It should be parameterized in a building block fashion, on the basis of already obtained parameters for Benzene.

At first again a mol object together with a ff addon instance has to be instanstiated.

.. code-block:: python
    
    >> import molsys
    >> m = molsys.mol.fromFile("ph-ph.mfpx")
    >> m.addon("ff")

In order to prepare a parameterization run one has to call the ``assign_params`` method in the following way:

.. code-block:: python
    
    >> m.ff.assign_params("MOF-FF", refsysname="ph-ph")

The refsysname argument tells the method to do not raise an AssignmentError if not parameters for all ics could be found.
Instead the internal data strucutures are prepared for an parameterization run. 

In addition one can also specify an **azone** which forces the algorithm to do not distribute parameters to an ic which has atoms 
included in the active zone. The azone is specified as followed:

.. code-block:: python
    
    >> m.ff.assign_params("MOF-FF", refsysname="ph-ph", azone = [0,11])

Note that here like in the whole molsys API the python numbering scheme starting at zero is employed. In addition it is possible to
specify equivalences. For this purpose a dictionary of dictionary to be passed. The keys of the outer dictionary are the names
of the reference systems for which the equivalence should be applied, in this case it is benzene, its value is again a dictionary
in which the ic for which the equivalences should be applied are stated as keys. Argument is the inner dictionary in which the 
indices of the atoms which should be altered are stated as keys. Value is an aftype object into which the origianl aftype should be 
altered.

.. code-block:: python
    
    >> from molsys.util.aftypes import aftpye
    >> equivs = {"benzene": {"dih": {
        0:aftype("c3_c2h1","ph"),
        11:aftype("c3_c2h1","ph")},
        "oop": {
        0:aftype("c3_c2h1","ph"),
        11:aftype("c3_c2h1","ph"),
        }}}
    >> m.ff.assign_params("MOF-FF", refsysname="ph-ph", azone = [0,11])

To write the gathered data to a file, again the write method is invoked

.. code-block:: python

    >> m.ff.write("ph-ph")

This creates the file ph-ph.fpar and ph-ph.ric. The read routine has to invoked with the fpar=True flag.

.. code-block:: python

    >> m.ff.read("ph-ph", fpar = True)


