weaver 
=================================


Generalities
-------------

The aim of the weaver project is to construct.

ok we have topologies, we have building blocks, lalala

Get things started
--------------------------
ok let's build HKUST_1, download your stuff and weave


Mofplus X (mfpx) files
--------------------------
A Mofplus X (generally ending with .mfpx, hece mfpx) file contains information about structure and connectivity of the object (molecules, building blocks, blueprints, frameworks, topologies, etc. etc. we like it).

Any mfpx file contains two parts: the header and the body.

The header contains a description of what it is there. Header is optional. (check it)

The body is what it is there. Body is mandatory.

The first line of the body is the number of points (atoms or vertices) in the file, usually addressed as "natoms" in our programme suite.

The following lines enumerate the properties per each point:

. a running index;

. the element type for atoms, just another type for vertices (somehow related to elements btw);

. the three cartesian coordinates of the point;

. depending on the type of the mfpx file, different informations:

.- 

.-

.-

. the connectivity (at the end of the line) as just indices or as indices/images

ok we have topo-type mfpx files and buildingblock-type mfpx files.

Topology (topo) mfpx files
---------------------------------------
Topology files store additional vertex type property and the connectivity includes information about the images the connected vertex belongs to.

EXAMPLE: tbo.mfpx


Building block (bb) mfpx files
---------------------------------------
Building block files contain fragment information (but it is not so necessary in this case because it is only one fragment.

EXAMPLE: btc.mfpx

EXAMPLE: CuPW.mfpx

building block
-------------------------

abcdef
-------------------------
