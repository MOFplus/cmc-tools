# -*- coding: utf-8 -*-

import numpy

"""
converts lattice vectors to abc/alpha/beta/gamma
and vice versa
note: angles are always in degrees!!!!

we use the usual convention
vector a is along the x axis
vector b is in the xy plane
vector c forms a right handed system with a/b
"""
 

def abc_from_vectors(vec):
   av = vec[0]
   bv = vec[1]
   cv = vec[2]
   a = numpy.sqrt(numpy.sum(av*av)) 
   b = numpy.sqrt(numpy.sum(bv*bv))
   c = numpy.sqrt(numpy.sum(cv*cv)) 
   cosa = numpy.sum(bv*cv)/(b*c)
   cosb = numpy.sum(av*cv)/(a*c)
   cosc = numpy.sum(av*bv)/(a*b)
   alpha   = numpy.arccos(cosa)*180.0/numpy.pi
   beta    = numpy.arccos(cosb)*180.0/numpy.pi
   gamma   = numpy.arccos(cosc)*180.0/numpy.pi
   return [a,b,c,alpha,beta,gamma]
   
def vectors_from_abc(abc, molden=False):
    a,b,c,alpha,beta,gamma=abc
    alpha *= numpy.pi/180.0
    beta  *= numpy.pi/180.0
    gamma *= numpy.pi/180.0
    cosa = numpy.cos(alpha)
    cosb = numpy.cos(beta)
    sinb = numpy.sin(beta)
    cosc = numpy.cos(gamma)
    sinc = numpy.sin(gamma)
    bterm = (cosa-cosb*cosc)/sinc
    cterm = numpy.sqrt(1.0-(cosb*cosb)-(bterm*bterm))
    vectors = []
    if molden:
        vectors.append([sinc,cosc,0])
        vectors.append([0,1,0])
        vectors.append([cosb,bterm,cterm])        
    else:
        vectors.append([1,0,0])
        vectors.append([cosc,sinc,0])
        vectors.append([cosb,bterm,cterm])
    vectors=numpy.array(vectors,dtype="float64")
    vectors[0] *= a
    vectors[1] *= b
    vectors[2] *= c
    return vectors
   
def get_cart_deformed_cell(cell, axis=0, size=1):
    '''
    Return the cell (with atoms) deformed along one
    of the cartesian directions
    (0,1,2 = x,y,z ; sheers: 3,4,5 = yz, xz, xy) by
    size percent.
    '''
    uc = cell
    l=size/100.0
    L=numpy.diag(numpy.ones(3))
    if axis < 3 :
        L[axis,axis]+=l
    else :
        if axis==3 :
            L[1,2]+=l
        elif axis==4 :
            L[0,2]+=l
        else :
            L[0,1]+=l
    uc=numpy.dot(uc,L)
    v = numpy.linalg.det(uc)
    print('volume:', v)
    return uc

def monoclinic_strain(cell, size=1):
    uc = cell
    l= size/100.0
    L = numpy.zeros((3,3))
    L[0,1] = 0.5*l
    L[1,0] = 0.5*l
    L[2,2] = l**2/(4-l**2)
    print(L)
    uc = numpy.dot(uc,L)
    return uc


