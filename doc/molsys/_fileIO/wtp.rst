.. molsys documentation master file, created by
   sphinx-quickstart on Thu Oct 18 10:01:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


`wtp`: Web Toper file
#####################

Rationale
=========

The back-end of our web suite MOF+ is made of two machines: one directly communicates with user's client and runs `fireworks` and `MySQL`; the other is not accessible to the user and runs high-demanding jobs which may be parallel and must be scheduled cluster-wise.
Web Toper file takes care of exchanging structure information between these two machines, in particular the blocks (molecular constituencies) and the net (inter-block connectivity). By desig, a machine:
- unweaves a molecular framework to the same net, provided the blocks are the same; [see further]
- may weave out of a wtp file a molecular framework without any further human intervention. Except for isoreticular isomerism, same connectivity is guaranteed.

Specifications
==============

A ``wtp`` file is made of sections. Each section starts with a heading line (the `header`) lead by the ``#*`` characters and specifying the type of the section and its options. A section can be a ``NET`` section or a ``BLOCK`` section.

Net section
-----------

The header of a net section is as follows:
``#* NET $SUPERCELL $NETNAME``
being ``$SUPERCELL`` the supercell defined in the format ``%ix%ix%i``, `e.g.` ``3x3x2``, and ``$NETNAME`` the name of the net in MOF+. If ``$NETNAME`` is not specified, then a mfpx-like structure specification must follow the net header. If both ``$SUPERCELL`` and ``$NETNAME`` are not specified, then a mfpx-like structure specification must follow the net header and a unit cell is assumed. It is not possible to specify ``$NETNAME`` without ``$SUPERCELL``: in that case, ``$NETNAME`` will be interpreted as ``$SUPERCELL`` likely raising an error. To specify a unit cell in MOF+, please always specify ``1x1x1`` as ``$SUPERCELL`` before ``$NETNAME``.

Examples
^^^^^^^^

::

    #* NET 1x1x1 tbo

::

    #* NET 2x1x3 fcu

::

    #* NET 2x1x3
    # type topo
    # cell     2.357100     2.357100     1.490600    90.000000    90.000000    90.000000
    6
    1 n 0.707130 0.707130 0.000000 0 5/13 6/13 6/12 
    2 n 1.649970 1.649970 0.000000 0 5/25 6/13 6/12 
    3 n 0.471420 1.885680 0.745300 0 5/17 5/16 6/13 
    4 n 1.885680 0.471420 0.745300 0 5/23 5/22 6/13 
    5 o 0.000000 0.000000 0.000000 1 1/13  2/1  3/9 3/10  4/3  4/4
    6 o 1.178550 1.178550 0.745300 1 1/13 1/14 2/13 2/14 3/13 4/13

::

    #* NET
    # type topo
    # cell     1.414200     1.414200     1.414200    90.000000    90.000000    90.000000
    4
    1 x12 0.000000 0.000000 0.000000 0 2/13 2/12 2/10  2/9 3/13  3/3  3/4 3/12 4/13  4/1  4/4 4/10
    2 x12 0.000000 0.707100 0.707100 0 1/13 1/14 1/16 1/17 3/13 3/16  3/4  3/7 4/13 4/14  4/4  4/5
    3 x12 0.707100 0.000000 0.707100 0 1/13 1/23 1/22 1/14 2/13 2/10 2/22 2/19 4/13 4/11 4/10 4/14
    4 x12 0.707100 0.707100 0.000000 0 1/13 1/25 1/22 1/16 2/13 2/12 2/22 2/21 3/13 3/15 3/16 3/12

Block section
-------------

The header of a block section is as follows:
``#* BLOCK $VERTEXTYPE $BLOCKNAME``
being ``$VERTEXTYPE`` the vertex the block is assigned to, and ``$BLOCKNAME`` the name of the block in MOF+. If ``$BLOCKNAME`` is not speficied, then a mfpx-like structure specification must follow the block header. If both ``$VERTEXTYPE`` and ``$BLOCKNAME`` are not specified, an error is raised: it is not possible to keep ``$VERTEXTYPE`` implicit.

Examples
^^^^^^^^

::

    #* BLOCK 1 CuPW

::

    #* BLOCK 0
    # type xyz
    # bbcenter com
    # bbconn 1*1 3*3 5*5 
    9
    1 c  0.000000 0.000000  0.000000 c3_*1c2 135-ph3 0 2 6 
    2 c  0.000000 0.000000  1.400000 c3_c2h1 135-ph3 0 1 3 7
    3 c  1.212436 0.000000  2.100000 c3_*1c2 135-ph3 0 2 4 
    4 c  2.424871 0.000000  1.400000 c3_c2h1 135-ph3 0 3 5 8
    5 c  2.424871 0.000000  0.000000 c3_*1c2 135-ph3 0 4 6 
    6 c  1.212436 0.000000 -0.700000 c3_c2h1 135-ph3 0 1 5 9
    7 h -0.943102 0.000000  1.944500 h1_c1   135-ph3 0 2 
    8 h  3.367973 0.000000  1.944500 h1_c1   135-ph3 0 4 
    9 h  1.212436 0.000000 -1.789000 h1_c1   135-ph3 0 6 

