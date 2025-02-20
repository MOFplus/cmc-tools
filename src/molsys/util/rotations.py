# -*- coding: utf-8 -*-
#try:
#    from Numeric import *
#except ImportError:


from numpy import *
import copy
import numpy
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
        if isclose(n,0):
           return v1
        else:
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

def rotate_by_triple(xyz, triple):
    rotmat = quat_to_mat(rotation_quat(triple))
    return dot(xyz, rotmat)

def rotate_random(v):
    return apply_mat(quat_to_mat(random_quat()),v)


def moi2(rs, ms=None):
    """Moment of inertia"""
    if ms is None: ms = numpy.ones(len(rs))
    else: ms = numpy.asarray(ms)
    rs = numpy.asarray(rs)
    N = rs.shape[1]
    # Matrix is symmetric, so inner/outer loop doesn't matter
    return [[(ms*rs[:,i]*rs[:,j]).sum()/ms.sum()
             for i in range(N)] for j in range(N)]

def moi(rs,ms=None):
    if ms is None: ms = numpy.ones(len(rs))
    else: ms = numpy.asarray(ms)
    rs = numpy.asarray(rs)

    Ixx = (ms* (rs[:,1]*rs[:,1] + rs[:,2]*rs[:,2])).sum()
    Iyy = (ms* (rs[:,0]*rs[:,0] + rs[:,2]*rs[:,2])).sum()
    Izz = (ms* (rs[:,0]*rs[:,0] + rs[:,1]*rs[:,1])).sum()
    Ixy =-(ms* rs[:,0] * rs[:,1]).sum()
    Ixz =-(ms* rs[:,0] * rs[:,2]).sum()
    Iyz =-(ms* rs[:,1] * rs[:,2]).sum()
    I = [[Ixx,Ixy,Ixy],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]]
    return numpy.array(I)/ms.sum()

def pax(rs,ms=None):
    if ms is None: ms = numpy.ones(len(rs))
    else: ms = numpy.asarray(ms)
    rs = numpy.asarray(rs)
    I = moi(rs,ms=ms)
    #print(I)
    eigval, eigvec = numpy.linalg.eigh(I)
    return eigval,eigvec
    
def align_pax(xyz,masses=None):
    eigval,eigvec = pax(xyz,ms=masses)
    eigorder = numpy.argsort(eigval)
    rotmat = eigvec[:,eigorder] #  sort the column vectors in the order of the eigenvalues to have largest on x, second largest on y, ... 
    return apply_mat(rotmat,xyz)

def align_bond_to(m,bond,align_xyz):
    """ (JK) align a bond to match the direction of the vector given by 'align_xyz' 
    bond (list of integers, len()=2) """
    dxyz = m.xyz[bond[1]] - m.xyz[bond[0]]
    import scipy.optimize as opt
    def pen(rot,x1,x2):
        x2t = x2.copy()
        x2t = rotate_by_triple(x2t,rot%1.0)
        ''' calculate the angle between the vecotrs and return it'''
        return numpy.arccos(numpy.dot(x1,x2t)/numpy.linalg.norm(x1)/numpy.linalg.norm(x2t))**2.0
    t0 = numpy.array([0.5,0.5,0.5])
    o = opt.minimize(pen,t0,args=(dxyz,align_xyz),method='SLSQP',)
    m.set_xyz(rotate_by_triple(m.xyz,o.x % 1.0))
    return o

def rec_walk_bond(m,ind,inds=[]):
    for i,c in enumerate(m.conn[ind]):
        if inds.count(c) == 0:
            inds.append(c)
            inds = rec_walk_bond(m,c,inds=inds)
        else:
            pass
    return inds

def rotate_around_bond(m,atom1,atom2,degrees=5.0):
    """Rotates the xyz coordinates by n degrees around the distance vector between two atoms
    
    let the situation be X-1-2-3-4-Y, either X,1 or Y,4 will be rotated accordingly 
    
    
    Arguments:
        mol {molsys.mol} -- the mol obect to apply the operation
        atom1 {integer} -- atom index 1
        atom2 {integer} -- atom index 2
    
    Keyword Arguments:
        degrees {float} -- rotation in degrees (default: {5.0})
    """
    ### detect the atoms that are subject to the rotation 
    ### rhs
    #import pdb;  pdb.set_trace()
    inds = sorted(rec_walk_bond(m,atom1,[atom2]))
    #print inds
    xyz = m.xyz
    xyz1 = xyz[atom1,:]
    xyz2 = xyz[atom2,:]
    vect = (xyz2-xyz1)
    vect /=  numpy.linalg.norm(vect)
    a,n1,n2,n3 = numpy.deg2rad(degrees),vect[0],vect[1],vect[2]
    ### formula from wikipedia https://de.wikipedia.org/wiki/Drehmatrix
    R= numpy.array([[n1*n1*(1-cos(a))+   cos(a), n1*n2*(1-cos(a))-n3*sin(a) , n1*n3*(1-cos(a))+n2*sin(a)],
                    [n2*n1*(1-cos(a))+n3*sin(a), n2*n2*(1-cos(a))+   cos(a) , n2*n3*(1-cos(a))-n1*sin(a)],
                    [n3*n1*(1-cos(a))-n2*sin(a), n3*n2*(1-cos(a))+n1*sin(a) , n3*n3*(1-cos(a))+   cos(a)]])
    xyz[inds,:] = numpy.dot(xyz[inds,:] - xyz[atom2,:],R)+xyz[atom2,:]
    m.xyz = xyz
    return xyz

def rotate_xyz_around_vector(xyz,vector,origin=[0.0,0.0,0.0],degrees=5.0):
    """(JK) Rotates the xyz coordinates by n degrees around any given vector
    
    Arguments:
        xyz (numpy.ndarray(3,)} -- the coordinates to apply the operation 
        vector {numpy.ndarray(3,)} -- direction vector along which to apply the rotation
    
    Keyword Arguments:
        origin {numpy.ndarray(3,)} -- origin of the rotation vector, defaults to cartesian origin 
        degrees {float} -- rotation in degrees (default: {5.0})
    """
    ### detect the atoms that are subject to the rotation 
    ### rhs
    #import pdb;  pdb.set_trace()
    xyz = copy.copy(xyz)
    origin = numpy.array(origin)
    vect = vector
    vect /=  numpy.linalg.norm(vect)
    a,n1,n2,n3 = numpy.deg2rad(degrees),vect[0],vect[1],vect[2]
    ### formula from wikipedia https://de.wikipedia.org/wiki/Drehmatrix
    R= numpy.array([[n1*n1*(1-cos(a))+   cos(a), n1*n2*(1-cos(a))-n3*sin(a) , n1*n3*(1-cos(a))+n2*sin(a)],
                    [n2*n1*(1-cos(a))+n3*sin(a), n2*n2*(1-cos(a))+   cos(a) , n2*n3*(1-cos(a))-n1*sin(a)],
                    [n3*n1*(1-cos(a))-n2*sin(a), n3*n2*(1-cos(a))+n1*sin(a) , n3*n3*(1-cos(a))+   cos(a)]])
    xyz -= origin
    xyz = numpy.dot(xyz,R)
    xyz += origin
    return xyz 

def rotate_around_vector(m,vector,origin=[0.0,0.0,0.0],degrees=5.0):
    """(JK) Rotates the xyz coordinates by n degrees around any given vector
    
    Arguments:
        m {molsys.mol} -- the mol obect to apply the operation
        vector {numpy.ndarray(3,)} -- direction vector along which to apply the rotation
    
    Keyword Arguments:
        origin {numpy.ndarray(3,)} -- origin of the rotation vector, defaults to cartesian origin 
        degrees {float} -- rotation in degrees (default: {5.0})
    """
    ### detect the atoms that are subject to the rotation 
    ### rhs
    #import pdb;  pdb.set_trace()
    origin = numpy.array(origin)
    vect = vector
    vect /=  numpy.linalg.norm(vect)
    a,n1,n2,n3 = numpy.deg2rad(degrees),vect[0],vect[1],vect[2]
    ### formula from wikipedia https://de.wikipedia.org/wiki/Drehmatrix
    R= numpy.array([[n1*n1*(1-cos(a))+   cos(a), n1*n2*(1-cos(a))-n3*sin(a) , n1*n3*(1-cos(a))+n2*sin(a)],
                    [n2*n1*(1-cos(a))+n3*sin(a), n2*n2*(1-cos(a))+   cos(a) , n2*n3*(1-cos(a))-n1*sin(a)],
                    [n3*n1*(1-cos(a))-n2*sin(a), n3*n2*(1-cos(a))+n1*sin(a) , n3*n3*(1-cos(a))+   cos(a)]])
    m.xyz -= origin
    m.xyz = numpy.dot(m.xyz,R)
    m.xyz += origin
    return 

def get_spherical_coordinates(xyz):
    ptsnew = numpy.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = numpy.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = numpy.arctan2(numpy.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,1] = numpy.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = numpy.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def get_cartesian_coordinates(sphere):
    if len(sphere) == 2:
        theta = numpy.deg2rad(sphere[0])
        phi   = numpy.deg2rad(sphere[1])
        r = 1.0
    if len(sphere) == 3:
        r     = numpy.deg2rad(sphere[0])
        theta = numpy.deg2rad(sphere[1])
        phi   = numpy.deg2rad(sphere[2])
    x = r * numpy.sin(theta) * numpy.cos(phi)
    y = r * numpy.sin(theta) * numpy.sin(phi)
    z = r * numpy.cos(theta)
    return (numpy.array([x,y,z]))

def normalize_angles_to_angle(angles,ref_angle=None):
    ''' JK
    shifts a trajectory of spherical coordinates in such a way as to have the reference angle
    ref_angle to be zero 
    '''
    if ref_angle is None:
        ref_angle = angles[0]
    new_angles = copy.copy(angles)
    new_angles -= ref_angle
    if ref_angle >= 0.0:
        new_angles[numpy.where(new_angles <= numpy.pi)[0]] += 2.0*numpy.pi
    else:
        new_angles[numpy.where(new_angles >= numpy.pi)[0]] -= 2.0*numpy.pi
    return new_angles

def get_rotmat_to_align(vec,target):
    '''
        taken from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    '''
    v = numpy.cross(vec,target)
    s = numpy.linalg.norm(v)
    c = numpy.dot(vec,target)
    vx = numpy.array([[    0,-v[2], v[1]],
                      [ v[2],    0,-v[0]],
                      [-v[1], v[0],    0]])

    R = numpy.eye(3) + vx + numpy.matmul(vx,vx) * (1-c)/(s*s)
    return R

