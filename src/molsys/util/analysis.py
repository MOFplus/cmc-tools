import numpy 

def _compute_single_orient(mol, atompair, store=True,local_frame=None):
    xyz = mol.xyz
    orient = numpy.zeros([3],dtype=xyz.dtype)
    cxyz = xyz  
    # unwrap the molecule, the reference image is the first atom:
    #         if any other atom is farther then the half cell we need to unwrap it
    #         NOTE this works only for molecules smaller then half the cell!!!
    mxyz   = mol.apply_pbc(cxyz,fixidx = atompair[0])
    vect   = mxyz[atompair[0]]-mxyz[atompair[1]]
    vect  /= numpy.sqrt(numpy.sum(vect*vect))
    orient = vect
    if local_frame is not None:
        xm = vect / numpy.linalg.norm(vect)
        # y_m is tricky. define it via
        ## hard coded for testing purposes
        r1i,r2i = local_frame[0],local_frame[1]
        r1 = mxyz[atompair[0]] - mxyz[r1i]
        r2 = mxyz[atompair[0]] - mxyz[r2i]
        r1 /= numpy.linalg.norm(r1)
        r2 /= numpy.linalg.norm(r2)
        zm1  = numpy.cross(r1,xm)
        zm2  = numpy.cross(xm,r2)
        zm1 /= numpy.linalg.norm(zm1)
        zm2 /= numpy.linalg.norm(zm2)
        zm  = (zm1+zm2) / 2.0
        zm /= numpy.linalg.norm(zm)
        ym  = numpy.cross(xm,zm)
        ym /= numpy.linalg.norm(ym)
    if local_frame is not None:
        return xm,ym,zm
    return orient

def compute_autocorrelation(x,dt):
    '''
    computes the autocorrelation function of a signal x featuring time differences dt
    '''
    from molsys.util.rotations import normalize_angles_to_angle
    def autocorr(x):
        result = numpy.correlate(x, x, mode='full')
        return result[result.size/2:]
    def acf(x, length=None):
        if length is None:
            length = x.shape[0]-1
        return numpy.array([1]+[numpy.corrcoef(x[:-i], x[i:]) for i in range(1, length)])
    #raw_acf = autocorr(x)
    t = numpy.linspace((0,x.shape[0]*dt,))
    nx = len(x)
    oacf = numpy.zeros(nx)
    nsamples = numpy.zeros(nx,dtype='int')
    o0 = 0.0
    for i in range(1,len(x)):
        ot = normalize_angles_to_angle(x[i:],ref_angle=x[i-1])
        nsamples[i:] += 1

    return oacf

def compute_autocorrelation2(x,dt):
    '''
    computes the autocorrelation function of a signal x featuring time differences dt
    '''
    from molsys.util.rotations import normalize_angles_to_angle
    t = numpy.linspace((0,x.shape[0]*dt,))
    nx = len(x)
    oacf = numpy.zeros(nx)
    nsamples = numpy.zeros(nx,dtype='int')
    o0 = 0.0
    for i in range(1,len(x)):
        ot = normalize_angles_to_angle(x[i:],ref_angle=x[i-1])
        nsamples[i:] += 1

    return oacf

def compute_autocorrelation_fft(x,dt):
    '''
    computes the autocorrelation function of a signal x featuring time differences dt
    '''
    from molsys.util.rotations import normalize_angles_to_angle
    t = numpy.linspace((0,x.shape[0]*dt,))
    nx = len(x)
    t_fft = numpy.fft.fft(x)
    S = t_fft * numpy.conj(t_fft)
    acf = numpy.fft.ifft(S)
    return acf
