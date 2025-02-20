# -*- coding: utf-8 -*-

try:
    from . import frotator
except ImportError:
    from . import frotator3 as frotator

from numpy import *
outerproduct = outer

PI2 = pi*2.0

# for debuging set a seed
#random.seed(42)


def make_vec(l):
    return array(l, "d")

def scal_prod(v1, v2):
    return sum(v1*v2,axis=-1)

def length(v):
    return sqrt(sum(v*v),axis=-1)

def norm(v1):
    return sqrt(scal_prod(v1,v1))

def normalize(v1):
    n = norm(v1)
    if isscalar(n):
        return v1/n
    else:
        return v1/n[:,newaxis]

def angle(v1, v2):
    _nv1 = normalize(v1)
    _nv2 = normalize(v2)
    d = scal_prod(_nv1, _nv2)
    if d < -1.0: d=-1.0
    if d > 1.0 : d= 1.0
    return arccos(d)

def project(v1, v2):
    _nv2 = normalize(v2)
    l = scal_prod(v1, _nv2)
    return _nv2*l

def cross_prod(a, b):
    return array( [a[1]*b[2] - a[2]*b[1], \
                     a[2]*b[0] - a[0]*b[2], \
                     a[0]*b[1] - a[1]*b[0]], "d")
                      
def rotmat(v, theta):
    Q = array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], "d")
    Q *= sin(theta)
    uut = outerproduct(v,v)
    Q += (identity(3,"d") - uut)*cos(theta)
    Q += uut
    return Q

def rotate(xyz, v, theta):
    return dot(xyz, transpose(rotmat(v, theta)))
    
def rotmat_from_euler(euler):
    R = zeros([3,3],"d")
    sa = sin(euler[0])
    ca = cos(euler[0])
    sb = sin(euler[1])
    cb = cos(euler[1])
    sg = sin(euler[2])
    cg = cos(euler[2])
    R[0, 0] =  cb * cg 
    R[1, 0] =  cb * sg 
    R[2, 0] = -sb 
    R[0, 1] = -ca * sg  +  sa * sb * cg 
    R[1, 1] =  ca * cg  +  sa * sb * sg 
    R[2, 1] =  sa * cb 
    R[0, 2] =  sa * sg  +  ca * sb * cg 
    R[1, 2] = -sa * cg  +  ca * sb * sg 
    R[2, 2] =  ca * cb
    return R 

def rotate_by_euler(xyz, euler):
    return dot(xyz, transpose(rotmat_from_euler(euler)))
    
def random_quat():
    rand = random.random(3)
    r1 = sqrt(1.0 - rand[0])
    r2 = sqrt(rand[0])
    t1 = PI2 * rand[1]
    t2 = PI2 * rand[2]
    return array([cos(t2)*r2, sin(t1)*r1, cos(t1)*r1, sin(t2)*r2])

def rotation_quat(triple):
    # with an input of three numbers between zero and one we scan the rotational space in an equal fashion
    t0 = triple[0]
    if t0>1.0:t0=1.0
    if t0<0.0:t0=0.0
    r1 = sqrt(1.0 - t0)
    r2 = sqrt(t0)
    t1 = PI2 * (triple[1]%1.0)
    t2 = PI2 * (triple[2]%1.0)
    return array([cos(t2)*r2, sin(t1)*r1, cos(t1)*r1, sin(t2)*r2])


def quat_to_mat(quat):
    q = array(quat, copy=True)
    n = dot(q, q)
    if n < 1.0e-15:
        return identity(3)
    q *= sqrt(2.0 / n)
    q  = outer(q, q)
    return array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

def apply_mat(m,v):
    return dot(v,m)

def rotate_by_triple(xyz, triple,use_new=False):
    if use_new:
        #print('using frotator rotation')
        xyz=array(xyz,dtype='float64')
        triple=array(triple,dtype='float64')
        frotator.rotate_by_triple(len(xyz),xyz.T,triple.T)
        return xyz
    else:
        rotmat = quat_to_mat(rotation_quat(triple))
        #for i in range(3):
            #print(rotmat[:,i])
        return dot(xyz, rotmat)


def rotate_random(v):
    return apply_mat(quat_to_mat(random_quat()),v)
    
