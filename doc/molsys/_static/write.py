#!/usr/bin/env python
# -*- coding: utf-8 -*-

import molsys
from molsys.util.aftypes import aftype

m = molsys.mol()
m.read("ph-ph.mfpx")

m.addon("ff")

equivs = {"benzene": {"dih": {
    0:aftype("c3_c2h1","ph"),
    11:aftype("c3_c2h1","ph")},
    "oop": {
    0:aftype("c3_c2h1","ph"),
    11:aftype("c3_c2h1","ph"),
    }}}

#azone = [2 7 8 9 10 11 12 13 14 15 21 22 23 24 25 26 37 38 39 40]
azone = [0,11]
#m.ff.assign_params("BABAK-FF" , refsysname = "ZnPW_py")
#azone = [0,1,2,3,4,5,6,7,8,9] 
m.ff.assign_params("BABAK-FF", refsysname="ph-ph", azone = azone, equivs = equivs)
m.ff.write("ph-ph")
