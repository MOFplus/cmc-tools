.. molsys user documentation (RS March 2021)

Quickstart
==========

This section gives a few examples on how to use the *molsys* *mol* object to give you a flavor of what to do with it. 
This docu is by far not complete and refers to some other topics like file formats, which are discussed elsewhere.

Basics
********

We start working with a molecular system. Let us assume you have made a xyz coordinate file benz.xyz for benzene using molden. It looks like this (you should know how
to make benzene with molden, but in order to proceed feel free to copy&paste this file):

.. literalinclude:: _static/benz.xyz

Now let us generate a molsys object and read the xyz file in.

.. code-block:: python

    >>> import molsys
    >>> m = molsys.mol()
    >>> m.read("benz.xyz")

Note that we can shortcut this with ``m = molsys.mol.from_file("benz.xyz")``. The file IO detects the file type by the extension.
Now we can get the basic info for the mol object **m** :

.. code-block:: python

    >>> m.get_natoms()
    12
    >>> m.get_elems()
    ['c', 'c', 'c', 'c', 'c', 'c', 'h', 'h', 'h', 'h', 'h', 'h']
    >>> m.get_xyz()
    array([[ 0.     ,  0.     ,  0.     ],
        [ 0.     ,  0.     ,  1.4    ],
        [ 1.21244,  0.     ,  2.1    ],
        [ 2.42487,  0.     ,  1.4    ],
        [ 2.42487,  0.     ,  0.     ],
        [ 1.21244,  0.     , -0.7    ],
        [-0.9431 ,  0.     ,  1.9445 ],
        [ 1.21244,  0.     ,  3.189  ],
        [ 3.36797,  0.     ,  1.9445 ],
        [ 3.36797,  0.     , -0.5445 ],
        [ 1.21244,  0.     , -1.789  ],
        [-0.9431 ,  0.     , -0.5445 ]])
    >>> m.get_bcond()
    0
    >>> m.get_atypes()
    ['c', 'c', 'c', 'c', 'c', 'c', 'h', 'h', 'h', 'h', 'h', 'h']

Note that
    - bcond is an integer and defines the PBC
        - 0 non-periodic
        - 1 cubic
        - 2 orthorhombic
        - 3 triclinic
    - the atomtypes (get_atypes) are not set (map to elements)

Since the system was read from pure coordinates also no connectivity info is available. We use a simple distance base connectivity detection in
``detect_conn()``. Better check the results. Here is works just fine.

.. code-block:: python

    >>> m.detect_conn()
    >>> m.get_conn()
    [[1, 5, 11], [0, 2, 6], [1, 3, 7], [2, 4, 8], [3, 5, 9], [0, 4, 10], [1], [2], [3], [4], [5], [0]]

Atom indices start with 0!
Now let us write to a file that stores all connectivity information. We use a file format called mfpx which is just an extended tinker (txyz) format.
In addition to tinker txyz files it stores also fragment info, unit cell data and further meta-data. See the section on file formats :ref:`File Formats`. 

.. code-block:: python

    >>> m.write("benz.mfpx")

Now we can generate atomtype information. *molsys* generates *MOF-FF* atomtypes which have the general form <element><coordination>_[seq:<neighbor elem><count>].
As an example c3_c3h1 means a carbon atom with 3 neighbors, 2x carbon and 1x hydrogen. There is a convenience script to generate atomtypes in a mfpx file 
(file will be overwritten).

.. code-block:: console

    $ atype benz.mfpx

after that the mfpx file looks like this

.. literalinclude:: _static/benz_nofrag.mfpx

The script for this operation is very simple and tells you how to use the atomtyper within python

.. literalinclude:: ../scripts/atype
    :language: python

As a last step we want to assign fragments (or fragment names) to this system, which are used in MOF-FF to assign force fields. For this to work, a couple of
things are needed. Potential fragments are downloaded from `MOF+ <https://www.mofplus.org>`_ and you need to register there as a user. You also need to provide 
your credentials for this to work. The most efficient way is to set the environment variables **MFPUSER** with your username (email used for registration) and 
**MFPPW** with your password (best in your .bashrc, make it readable only by you).
Either use a script (this also does the atomtyping)

.. code-block:: console

    $ fragmentize benz.mfpx

or do it in python by instantiating a fragmentizer and passing the mol object to it.

.. code-block:: python

    >>> import molsys
    >>> m = molsys.mol.from_file("benz.mfpx")
    >>> frag = molsys.util.fragmentizer()
    >>> frag(m)
    >>> m.write("benz.mfpx")

This is the final result

.. literalinclude:: _static/benz.mfpx

All atoms belong the the fragment **ph** for phenyl and to fragment number 0

Periodic Systems
****************

Periodic systems are not that easy to generate as the benzene molecule shown above. Therefore we use a mfpx file of a MOF that can be downloaded
from `MOF+ <https://www.mofplus.org>`_. Under **Frameworks** search for the MOF name "hkust" and check out the MOF with the *tbo* net.
Or you just follow this `link <https://www.mofplus.org/mofs/mof/1>`_ and then dopwnload the coordinates as mfpx (the other option is cif). 
The file you get looks like this (at least the first 20 lines)
NOTE: This is one of the first frameworks we generated on MOF+ and it contains the wrong fragment names. In order to fix this please run
*fragmentize hkust1.mfpx* on the downloaded file. We will fix this at some point. 

.. literalinclude:: _static/hkust1.mfpx
    :lines: 1-20

As you can see in mfpx the cell parameter are stored as metadata (lines with *#* at the top). The system is automatically detected as cubic.

.. code-block:: python

    >> import molsys
    >>> m = molsys.mol.from_file("hkust1.mfpx")
    >>> m.get_bcond()
    1
    >>> m.get_cell()
    array([[26.4327,  0.    ,  0.    ],
        [ 0.    , 26.4327,  0.    ],
        [ 0.    ,  0.    , 26.4327]])
    >>> m.get_cellparams()
    [26.4327, 26.4327, 26.4327, 90.0, 90.0, 90.0]

One of the powerful tools of the native *mol* class (without using any addons) is to generate supercells of periodic systems with an automatic 
generation of the connectivity of the supercell. You can use the method :py:class:`molsys.mol.make_supercell` for that.

.. code-block:: python

    >>> m.make_supercell([2,2,1])
    >>> m.write("hkust1_221.mfpx")

This enhances the system in *x* and *y* direction to span 2 unitcells. The number of atoms gets 4 times as big. To visualize your system you can either use
*moldenx* (You need to have molden installed and in your path. moldenx is a wrapper script in *molsys* that converts the mfpx on the fly into a tinker *txyz* file
and shows it. The better option is to use the vmd plugin that comes with molsys. You need to have vmd installed and compile and install the plugin (there is a plugin
for *mfpx* and another for *mfp5* aka *mfp5*, which is trajectory info written in a *hdf5* file format.






