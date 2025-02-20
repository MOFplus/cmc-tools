moldenx
#######

Introduction
============
moldenx is a wrapper to molden, very famous program of molecular visualization made by etc. Source code of the program is available here (URL).
The wrapper is intended to view a mfpx (MOF+ eXtended) file, which is yet another file format based on Tinker xyz file format.

Essentially, moldenx take care of this:
grep -v \# $mfpxfile | awk '{$6="5"; $7=""; $8=""; print}' > $txyzfile
where $mfpxfile is the attached filename, $txyzfile is your target filename (watch out: ending with .txyz).
You can open $txyzfile via molden or via vmd to properly see the bonds. My molden crashes for huge structures, so use vmd, then file->new molecule->browse $txyzfile and determine file type as "Tinker".


Usage
=====
moldenx has the same signature of molden. So, moldenx -h the output moldenx -S remove the logo moldenx -a avoid Z-matrix calculation (useful for topologies!), etc.

Copyright
=========
molden is (c)
moldenx is GPLv3.
