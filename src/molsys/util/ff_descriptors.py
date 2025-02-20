#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:07:12 2017

@author: johannes
"""

bnd = """# Supported bnd potentials mm3, quartic, morse
# mm3:     E(r) = 1/2 k(r-r0)^2*[1-2.55(r-r0)+7/12(2.55(r-r0))^2]
#          r0 in A, k in mdyn/A
#          mm3 k r0
# morse:   E(r) = (1/2a^2)*k*[1-e^(-a(r-r0))] with Ed = k/2a^2
#          r0 in A, k in mdyn/A, Ed in kcal/mol
#          morse k r Ed
# quartic: E(r) = 1/2 k(r-r0)^2*[1-k2(r-r0)+k3(k2(r-r0))^2]
#          r0 in A, k in mdyn/A, k2 in A^-1 and k3 is unitlesl
#          quartic k r0 k2 k3"""


ang = """# Supported ang potentials mm3, fourier, quartic, strbnd
# mm3:     E(a) = 1/2 k(a-a0)^2*[1-0.014(a-a0)+5.6e-5(a-a0)^2-7e-7(a-a0)^3+2.2e-8(a-a0)^4]
#          a0 in deg, k in mdyn/(A*rad)
#          mm3 k a0
# fourier: E(r) = V/a[1+cos(n*a+a0)]
#          a0 in deg, V in kcal/mol, n is unitless
#          fourier V a0 n 1.0 1.0
# quartic: E(a) = 1/2 k(a-a0)^2*[1-k2(a-a0)+k3(k2(a-a0))^2]
#          a0 in deg, k in mdyn/(A*rad), k2 in 1/rad, and k3 is unitless
#          quartic k a0 k2 k3
# strbnd:  E(r1,r2,a) = kss(r1-r10)(r2-r20)+(a-a0)*[ksb1*(r1-r10)+ksb2(r2-r20)]
#          r10, r20 in A, a in deg, kss in mdyn/A, ksb1 and ksb2 in mdyn/(rad*A)
#          strbnd ksb1 ksb2 kss r10 r20 a0"""


dih = """# Supported dih potentials cos3, cos4
# cos3:    E(d) = Va/2[1+cos(d)]+Vb/2[1-cos(d)]+Vc/2[1+cos(d)]
#          Va, Vb and Vc in kcal/mol
#          cos3 Va Vb Vc
# cos4:    E(d) = Va/2[1+cos(d)]+Vb/2[1-cos(d)]+Vc/2[1+cos(d)]+Vd/2[1-cos(d)]
#          Va, Vb, Vd and Vc in kcal/mol
#          cos4 Va Vb Vc Vd"""

oop = """# Supported oop potentials harm
# harm:    E(d) = k/2*(d-d0)**2
#          k in mdyn/(rad*A), d0 in deg
#          harm k d0
# cos4:    E(d) = Va/2[1+cos(d)]+Vb/2[1-cos(d)]+Vc/2[1+cos(d)]+Vd/2[1-cos(d)]
#          Va, Vb, Vd and Vc in kcal/mol
#          cos4 Va Vb Vc Vd"""

cha = """# Supported charge types
# gaussian: q in e-, w in A
#           gaussian q w """

vdw = """# Suppoerted types
# buck6d: ep in kcal/mol, r0 in A
#         buck r0 ep"""

vdwpr = ""

chapr = ""

desc = {"bnd": bnd,
        "ang": ang,
        "dih": dih,
        "oop": oop,
        "cha": cha,
        "vdw": vdw,
        "vdwpr": vdwpr,
        "chapr": chapr
        }


