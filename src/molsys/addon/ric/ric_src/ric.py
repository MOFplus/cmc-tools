#! /usr/bin/env python

import numpy as np

#from .base import _Base
#from .dist import Distance
#from .angl import Angle
#from .dihed import Dihedral
from ._ric import red_int_coords as _ric_build
from ._ric import ric_stretches  as _ric_str
from ._ric import ric_in_bends   as _ric_ibe
from ._ric import ric_out_bends  as _ric_obe
from ._ric import ric_lin_bends  as _ric_lbe
from ._ric import ric_torsions   as _ric_tor
from ._ric import ric_eckart     as _ric_eck

def allocate(array):
    if len(array) != 0:
        return np.transpose(np.array(array))
    return None

class data(object):
    """
    Class to mimic the data structure of the deprecated fortran red_int_coord
    module
    """

    def __init__(self):
        return

class RedIntCoords(object):
  """
  Redundant Internal Coordinates

  A molecular structure can be described in terms of internal degrees of
  freedom (i.e. bond distances, valence angles, etc.) and invariant modes of
  motion (translation and rotation). Together their form a set of redundant
  internal coordinates (RIC).

  RIC are divided into 3 groups:
    * old (FORTRAN-based)
      * stretch
      * in_bend
      * lin_bend
      * out_bend
      * torsion
    * new (Python-based)
      * distance
      * angle
      * dihedral
    * special
      * Eckart coordinates

  The old RICs can be defined in term of individual atoms only. The new RICs
  accept atom groups additionally.

  Example
  -------

  >>> ric = RedIntCoords()
  >>> ric.add_stretch([1, 2])
  >>> ric.add_stretch([2, 3])

  This is an alternative (but not equivalent internally)
  >>> ric = RedIntCoords()
  >>> ric.add_distance([1, 2])
  >>> ric.add_distance([2, 3])

  Only the new RICs can be defined with the atom groups
  >>> g1 = Group([1, 2])
  >>> g2 = Group([3, 4])
  >>> ric = RedIntCoords()
  >>> ric.add_distance([g1, g2])
  >>> ric.add_distance([g2, 5])
  """

  # Maximum number of atoms in a torsion definition
  MAX_TORSION_ATOMS = 12
  assert MAX_TORSION_ATOMS >= 4
  assert MAX_TORSION_ATOMS % 2 == 0

  def __init__(self):

    # State flags
    self._setup = False
    self._eval  = False

    # Check for multiple instances
    # assert not RedIntCoords._ric, 'Only one instance of RedIntCoords can be created!'
#    RedIntCoords._ric = _ric
#    self._ric         = _ric # Just a convenient alias

    # Temporal storage for RIC definitions
    self._stretches    = []
    self._in_bends     = []
    self._out_bends    = []
    self._lin_bends    = []
    self._torsions     = []
    self._eckart_trans = []
    self._eckart_rots  = []

    # Temporal storage for linear bend reference
    self._lin_bend_refs = []

    # Temporal storage for torsion dihedral angle definition
    self._torsion_ivals = []

    # RICs and their B-matrix row indices
    self._rics = []
    self._ibrs = None

  def __del__(self):

    RedIntCoords._ric = None

  #============================================================================

  def _add(self, ric):
    """
      Add an internal coordinate to the set of RICs.
    """

    assert isinstance(ric, _Base)

    self._rics.append(ric)

    return self

  def add_distance(self, points):
    """
      Add a distance to the set of RIC.

      See *ric.Distance* for details.

      Example
      -------

      >>> g = Group([2, 3])
      >>> ric.add_distance([1, g])
    """

    ric = Distance(points)
    self._add(ric)

    return self

  def add_angle(self, points, axis=None):
    """
      Add an planar or linear angle to the set of RIC.

      See *ric.Angle* for details.

      Note
      ----
      In case of the linear angle, this methods add just one RIC.
      Meanwhile *add_lin_bend* adds two RICs.

      Example
      -------

      >>> g = Group([1, 2])
      >>> ric.add_angle([3, g, 4])
    """

    ric = Angle(points, axis=axis)
    self._add(ric)

    return self

  def add_dihedral(self, left, centre, right, ivals=[1,1]):
    """
      Add a dihedral angle to the set of RIC.

      See *ric.Dihedral* for more details.

      Example
      -------

      >>> g = Group([1, 2])
      >>> ric.add_dihedral([g, 3], [4, 5], [6, 7, 8])
    """

    ric = Dihedral(left, centre, right, ivals=ivals)
    self._add(ric)

    return self

  def _get_indices(self, type):
    """
      Get the indices of RICs of the type *type*.
    """

    assert issubclass(type, _Base)

    indices = [i for i, ric in enumerate(self._rics) if isinstance(ric, type)]

    return indices

  @property
  def num_distances(self):
    """Number of distances"""

    return len(self._get_indices(Distance))

  @property
  def num_angles(self):
    """Number of planar and linear angles"""

    return len(self._get_indices(Angle))

  @property
  def num_diherals(self):
    """Number of dihedral angles"""

    return len(self._get_indices(Dihedral))


  def get_angle_axes(self, cart_coords):
    """
      Get the angle axis vectors.

      See *ric.Angle* for details.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      axes: (M, 3) array
            Axis vectors, where M is the number of angles in the set of RICs.
            The axes are sorted in the same order as *add_angle* called.
    """

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[1] == 3

    indices = self._get_indices(Angle)
    axes = [ric.get_axis(cart_coords) for ric in self._rics[indices]]
    axes = np.array(axes)

    return axes

  def _evaluate(self, cart_coords, type=None):

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[1] == 3

    rics = self._rics
    if isinstance(type, _Base):
      indices = self._get_indices(type)
      rics = rics[indices]

    values = [ric.evaluate(cart_coords) for ric in rics]
    values = np.array(values)

    return values

  def get_val_distance(self, cart_coords):
    """
      Compute the distance values from a given atomic positions.

      See *ric.Distance* for details.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      values: (M,) array
              Distance values. The values are sorted in the same order as
              *add_distance* called.
    """

    return self._evaluate(cart_coords, Distance)

  def get_val_angles(self, cart_coords):
    """
      Compute the angle values from a given atomic positions.

      See *ric.Angle* for details.

      Note
      ----
      The values are in the range [0, pi] or (-pi, pi] for the planar
      and linear angles, respectively.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      values: (M,) array
              Angle values. The values are sorted in the same order as
              *add_angle* called.
    """

    return self._evaluate(cart_coords, Angle)

  def get_val_dihedrals(self, cart_coords):
    """
    Compute the dihedral angle values from a given Cartesian coordinates
    *cart_coords*.

    The values are returned in the same order as the dihedral angle coordinates
    were added, e.g. ric.RedIntCoords.add_dihedral called.
    """

    return self._evaluate(cart_coords, Dihedral)

  #============================================================================

  def add_stretch(self, indices):
    """
    Add a bond stretch to a set of redundant internal coordinates

    Parameters
    ----------
    indices: (2,) array
             Array of atomic indices defining a bond stretch.
             The indices can be be from 1 to the number of atoms.

    Returns
    -------
    None

    Examples
    --------
    Add a bond stretch between 2nd and 3rd atoms

    >>> ric.add_stretch([2,3])
    """

    indices = list(indices)
    assert len(indices) == 2
    self._stretches.append(indices)
    self._setup = False

  def add_in_bend(self, indices):
    """
    Add a in-plane bend to a set of redundant internal coordinates

    Parameters
    ----------
    indices: (3,) array
             Array of atomic indices defining an in-plane bend.
             The indices can be be from 1 to the number of atoms.

    Returns
    -------
    None

    Example
    -------

    >>> ric.add_in_bend([1,2,3])
    """

    indices = list(indices)
    assert len(indices) == 3
    self._in_bends.append(indices)
    self._setup = False

  def add_out_bend(self, indices):
    """
    Add an out-of-plane bend to a set of redundant internal coordinates

          D
         /
    B---A
         \\
          C

    Here A is a central atom; B, C, and D -- terminal atoms.

    Parameters
    ----------
    indices: (4,) array
             Array of atomic indices defining an out-of-plane bend.
             The indices can be be from 1 to the number of atoms.

    Returns
    -------
    None

    Example
    -------

          4
         /
    2---1
         \\
          3

    >>> ric.add_out_bend([1,2,3,4])
    """

    indices = list(indices)
    assert len(indices) == 4
    self._out_bends.append(indices)
    self._setup = False

  def add_lin_bend(self, indices, ref):
    """
    Add a pair of linear bends to a set of redundant internal coordinates.

    A---B---C
       / \\
      V   V

    Here A and C are terminal atoms, and B -- central. The bending is in
    a perpendicular plane to a reference axis (-->). The reference axes can be
    defines as the Cartesian basis vectors or generated using a reference
    atom position.

    In case, the reference axes are defined by a reference atom (X). The 1st
    axis is generated to be in a plane ACX and perpendicular to AC. The 2nd
    axis is perpendicular to the 1st axis and AC.

    Parameters
    ----------
    indices: (3,) array
             Array of atomic indices defining an linear bend.
             The indices can be be from 1 to the number of atoms.
    ref: integer or string
         Reference axis definition can be *'xy'*, *'xz'*, and *'yz'* or
         an atomic index.
         The index can be be from 1 to the number of atoms.

    Returns
    -------
    None

    Example
    -------

    1---2---3
       / \\
      X   Y

    >>> ric.add_lin_bend([1,2,3], 'xy')
    """

    indices = list(indices)
    assert len(indices) == 3

    self._lin_bends.append(indices) # 1st linear bend
    self._lin_bends.append(indices) # 2nd linear bend

    if   ref == 'xy':
      self._lin_bend_refs.append('x')
      self._lin_bend_refs.append('y')
    elif ref == 'xz':
      self._lin_bend_refs.append('x')
      self._lin_bend_refs.append('z')
    elif ref == 'yz':
      self._lin_bend_refs.append('y')
      self._lin_bend_refs.append('z')
    else:
      ref = int(ref)
      assert ref != 0
      self._lin_bend_refs.append( ref)
      self._lin_bend_refs.append(-ref)

    self._setup = False

  def add_torsion(self, indices, ivals=[1,1]):
    """
    Add a torsion to a set of redundant internal coordinates

    A
     \\
      B---C
           \\
            D

    Here A and D are terminal atoms, B and C -- axial atoms

    Parameters
    ----------
    indices: (4-12,) array
             Array of atomic indices defining a torsion.
             Array size variable depending on the number of terminal atoms,
             it has to be even with the axial atoms in the middle.
             The indices can be be from 1 to the number of atoms.
    ivals:   (2,) array, optional
             Indices of terminal atoms included in dihedral angle calculation.
             The indices can be from 1 to 5 (inclusive), i.e [1,5]. Each
             terminal side is numbered independently.

    Returns
    -------
    None

    Example
    -------

    1
     \\
      2---3
           \\
            4

    >>> ric.add_torsion([1,2,3,4]) # 1--2--3--4 dihedral angle computed

    1
     \\
      3---4
     /     \\
    2       5

    >>> ric.add_torsion([1,2,3,4,5,0],ivals=[2,1]) # 2--3--4--5 dihedral angle
                                                     computed

     1
      \\
    2--4---5
      /     \\
     3       6

    >>> ric.add_torsion([1,2,3,4,5,6,0,0],ivals=[2,1]) # 2--4--5--6 dihedral
                                                         angle computed

     1       6
      \\     /
    2--4---5
      /     \\
     3       7

    >>> ric.add_torsion([1,2,3,4,5,6,7,0],ivals=[1,2]) # 1--4--5--7 dihedral
                                                         angle computed


    """

    MTA = RedIntCoords.MAX_TORSION_ATOMS # Maximum number of torsion atoms

    indices = list(indices)

    # Append trailing 0 indices
    natom = len(indices); assert natom in range(4,MTA+1,2)
    indices[natom//2-1:natom//2-1]  = [0]*((MTA-natom)//2)
    indices                      += [0]*((MTA-natom)//2)

    assert len(indices) == MTA
    self._torsions.append(indices)

    # Check and store "ivals"
    ivals = [int(i) for i in ivals]
    assert len(ivals) == 2
    self._torsion_ivals.append(ivals)

    self._setup = False

  def add_eckart(self, trans=[True]*3, rots=[True]*3):
    """
    Add Eckart translations and rotations to a set of redundant internal
    coordinates

    Parameters
    ----------
    trans: list
           List of Eckart translations to add. Each element represent Cartesian
           components (x, y, and z).
    rots:  list
           List of Eckart rotations to add. Each element represent Cartesian
           components (x, y, and z).

    Return
    ------
    None

    Example
    -------

    Add all Eckart coordinates by default
    >>> ric.add_eckart()

    Or just specific components
    >>> ric.add_eckart(trans=[True,True,True],rots=[True,False,False])
    """

    trans = [bool(tran) for tran in list(trans)]
    assert len(trans) == 3, '"trans" has to have 3 elements'
    self._eckart_trans = np.array([icomp+1 for icomp in range(3) if trans[icomp]])

    rots = [bool(rot) for rot in list(rots)]
    assert len(rots) == 3, '"rots" has to have 3 elements'
    self._eckart_rots = np.array([icomp+1 for icomp in range(3) if rots[icomp]])

    self._setup = False

  #============================================================================

  def setup(self, masses):
    """
    Setup RIC and other stuff

    This method has to be called after the set of RICs have been constructed.

    Parameters
    ----------
    masses: (N,) array
            Array of atomic masses, where N is the number of atoms in
            the system

    Returns
    -------
    None

    Example
    -------

    >>> ric = RedIntCoords()
    >>> ric.add_stretch([2, 3])
    >>> ric.setup(masses=[1.0, 16.0, 1.0])
    """
    # Instantiate data class as _ric in order to preserve data structure
    # of deprecated _ric fortran module variables
    self._ric = data()

    # Check and allocate atomic masses
    assert isinstance(masses,np.ndarray)
    assert masses.dtype == np.float64
    assert masses.ndim  == 1
    assert masses.size > 0
    assert np.all(masses > 0)
    self._ric.atomic_masses = masses

    # Get the number of atoms
    natom = self._ric.atomic_masses.size

    # Allocate Cartesian coordinate and Hessian arrays
    self._ric.cart_coords  = np.zeros((3,natom))
    self._ric.cart_hessian = np.zeros((3*natom,3*natom))

    # Setup stretch definitions and check
    # Note: self.streches is [# of stretches, 2] array,
    # but for Fortran subroutines it needs to be transposed.
    nstretch = 0
    self._ric.ric_def_stretches = np.transpose(np.array(self._stretches)) # Array isn't allocated if no stretches present
    if not self._ric.ric_def_stretches is None:
      nstretch = self._ric.ric_def_stretches.shape[1] # Get the number of stretches
      assert self._ric.ric_def_stretches.shape[0] == 2
      assert np.all(self._ric.ric_def_stretches >= 1)
      assert np.all(self._ric.ric_def_stretches <= natom)
      #for stretch in ric.ric_def_stretches:
      #  assert np.unique(stretch).size == stretch.size # Check for repeated indices

    # Setup in-plane bend definitions and check
    # Not: self.in_bends is [# of in bends, 3] array,
    # but for Fortran subroutines it needs to be transposed.
    nin_bend = 0
    self._ric.ric_def_in_bends = np.transpose(np.array(self._in_bends))
    if not self._ric.ric_def_in_bends is None:
      nin_bend = self._ric.ric_def_in_bends.shape[1]
      assert self._ric.ric_def_in_bends.shape[0] == 3
      assert np.all(self._ric.ric_def_in_bends >= 1)
      assert np.all(self._ric.ric_def_in_bends <= natom)
      #for in_bend in ric.ric_def_in_bends:
      #  assert np.unique(in_bend).size == in_bend.size

    # Setup out-of-plane bend definitions and check
    nout_bend = 0
    self._ric.ric_def_out_bends = allocate(self._out_bends)
    if not self._ric.ric_def_out_bends is None:
      nout_bend = self._ric.ric_def_out_bends.shape[1]
      assert self._ric.ric_def_out_bends.shape[0] == 4
      assert np.all(self._ric.ric_def_out_bends >= 1)
      assert np.all(self._ric.ric_def_out_bends <= natom)
      #for out_bend in ric.ric_def_out_bends:
      #  assert np.unique(out_bend).size == out_bend.size

    # Setup linear bend definitios and check
    nlin_bend = 0
    self._ric.ric_def_lin_bends = allocate(self._lin_bends)
    if not self._ric.ric_def_lin_bends is None:
      #import pdb; pdb.set_trace()
      nlin_bend = self._ric.ric_def_lin_bends.shape[1]
      assert self._ric.ric_def_lin_bends.shape[0] == 3
      assert np.all(self._ric.ric_def_lin_bends >= 1)
      assert np.all(self._ric.ric_def_lin_bends <= natom)
      #for lin_bend in ric.ric_def_lin_bends:
      #  assert np.unique(lin_bend).size == lin_bend.size
      #self._ric.ric_lin_bend_vecs = np.zeros((3,nlin_bend))
      assert len(self._lin_bend_refs) == nlin_bend
      lin_bend_inds  = []
      lin_bend_axes  = []
      for ref in self._lin_bend_refs:
        if   ref == 'x':
          lin_bend_inds.append(0)
          lin_bend_axes.append([1.,0.,0.])
        elif ref == 'y':
          lin_bend_inds.append(0)
          lin_bend_axes.append([0.,1.,0.])
        elif ref == 'z':
          lin_bend_inds.append(0)
          lin_bend_axes.append([0.,0.,1.])
        elif type(ref) is int:
          assert abs(ref) >= 1
          assert abs(ref) <= natom
          lin_bend_inds.append(ref)
          lin_bend_axes.append([0.,0.,0.])
        else:
          assert False
      self._ric.ric_lin_bend_inds = np.array(lin_bend_inds)
      self._ric.ric_lin_bend_axes = np.array(lin_bend_axes).transpose()
      assert self._ric.ric_lin_bend_inds.size  == nlin_bend
      assert self._ric.ric_lin_bend_axes.shape == (3,nlin_bend)

    # Setup torsions definitions and check
    MTA = RedIntCoords.MAX_TORSION_ATOMS
    ntorsion = 0
    self._ric.ric_def_torsions  = allocate(self._torsions)
    self._ric.ric_torsion_ivals = np.transpose(self._torsion_ivals)
    if not self._ric.ric_def_torsions is None:
      ntorsion = self._ric.ric_def_torsions.shape[1]
      assert self._ric.ric_def_torsions.shape[0] == MTA
      assert np.all(self._ric.ric_def_torsions[0,:] >= 1)
      assert np.all(self._ric.ric_def_torsions[MTA//2-1:MTA//2+2,:] >= 1)
      assert np.all(self._ric.ric_def_torsions >= 0)
      assert np.all(self._ric.ric_def_torsions <= natom)
      #for torsion in ric.ric_def_torsions:
        #for i in range(1,MTA/2-2) + range(MTA/2+2,MTA-1):
        #  if torsion[i] == 0: assert torsion[i+1] == 0
        #torsion = np.array([t for t in torsion if t > 0]) # Remove 0, i.e. [1,2,0,0,0,0,3,4,5,6,0,0,0] => [1,2,3,4,5,6]
        #assert np.unique(torsion).size == torsion.size
      assert self._ric.ric_torsion_ivals.shape[0] == 2
      assert self._ric.ric_torsion_ivals.shape[1] == ntorsion
      assert np.all(self._ric.ric_torsion_ivals >= 1)
      assert np.all(self._ric.ric_torsion_ivals <= 5)
      for itorsion in range(ntorsion):
        i1, i2 = self._ric.ric_torsion_ivals[:,itorsion]
        assert self._ric.ric_def_torsions[      i1-1,itorsion] > 0
        assert self._ric.ric_def_torsions[MTA//2+i2-1,itorsion] > 0

    # Setup Eckart translations definitions and check
    neckart_trans = 0
    self._ric.ric_def_eckart_trans = np.array(self._eckart_trans)
    if not self._ric.ric_def_eckart_trans is None:
      neckart_trans = self._ric.ric_def_eckart_trans.size
      assert neckart_trans >=  0
      assert neckart_trans <= 3
      assert np.all(self._ric.ric_def_eckart_trans >= 1)
      assert np.all(self._ric.ric_def_eckart_trans <= 3)
      assert np.unique(self._ric.ric_def_eckart_trans).size == self._ric.ric_def_eckart_trans.size

    # Setup Eckart rotation definitions and check
    neckart_rots = 0
    self._ric.ric_def_eckart_rots = np.array(self._eckart_rots)
    if not self._ric.ric_def_eckart_rots is None:
      neckart_rots = self._ric.ric_def_eckart_rots.size
      assert neckart_rots >= 0
      assert neckart_rots <= 3
      assert np.all(self._ric.ric_def_eckart_rots >= 1)
      assert np.all(self._ric.ric_def_eckart_rots <= 3)
      assert np.unique(self._ric.ric_def_eckart_rots).size == self._ric.ric_def_eckart_rots.size

    # Setup complex RICs
    ncric = len(self._rics)

    # Count the number of RICs and allocate their B-matrix row indices
    nric = 0
    self._ric.ric_ibr_stretches       = np.arange(nric,nric+nstretch) + 1
    nric += nstretch
    self._ric.ric_ibr_in_bends        = np.arange(nric,nric+nin_bend) + 1
    nric += nin_bend
    self._ric.ric_ibr_out_bends       = np.arange(nric,nric+nout_bend) + 1
    nric += nout_bend
    self._ric.ric_ibr_lin_bends       = np.arange(nric,nric+nlin_bend) + 1
    nric += nlin_bend
    self._ric.ric_ibr_torsions        = np.arange(nric,nric+ntorsion) + 1
    nric += ntorsion
    self._ric.ric_ibr_eckart_trans    = np.arange(nric,nric+neckart_trans) + 1
    nric += neckart_trans
    self._ric.ric_ibr_eckart_rots     = np.arange(nric,nric+neckart_rots) + 1
    nric += neckart_rots
    # Note: complex RICs are handle in Python, so B-matrix rows have 0-based indices
    self._ibrs                        = np.arange(nric,nric+ncric)
    nric += ncric
    #assert nric >= 3*natom, 'Internal coordinate set is neither redundant nor complete!
    
    self.nric = nric

    # Allocate B-marix and inverse B-matrix arrays
    self._ric.bmat     = np.zeros((3*natom,nric),order='F')
    self._ric.bmat_inv = np.zeros((3*natom,nric),order='F')

    # Allocate RIC value and Hessian arrays
    self._ric.ric_val_stretches = np.zeros(nstretch)
    self._ric.ric_val_in_bends  = np.zeros(nin_bend)
    self._ric.ric_val_out_bends = np.zeros(nout_bend)
    self._ric.ric_val_lin_bends = np.zeros(nlin_bend)
    self._ric.ric_val_torsions  = np.zeros(ntorsion)
    self._ric.ric_val_eckarts   = np.zeros(neckart_trans+neckart_rots)
    self._ric.ric_hessian       = np.zeros((nric,nric), order = 'F')

    self._setup = True

  def construct_b_matrix(self, cart_hmat, cart_coords):
    """
    Construct B-matrix and evaluate RICs

    B-matrix is a transformation from the Cartesian coordinates to the RICs.

    The order of RICs in the B-matrix:
      * stretchs
      * in_bends
      * out_bends
      * lin_bends (in pairs)
      * torsions
      * Eckart coordinates (translations and rotations)
      * new (distances, angles, and dihedrals)

    The order of the new (Python-based) RICs corresponds to the order they
    are added (*add_distance*, *add_angle*, and *add_dihedral* called).

    Parameters
    ----------
    cart_hmat: (3,3) array or None
               h-matrix defining lattice basis of vectors for the periodic
               boundary condition. If it is 'None', the system is non-periodic.
    cart_coords: (N, 3) array
                 Atomic positions in Cartesian coordinates, where N is the
                 number of atoms in the system.

    Returns
    -------
    bmat: (M, 3*N) array
          B-matrix, where N is the number of atom in the system and M is
          the number of RICs.

    Example
    -------

    >>> bmat = ric.construct_b_matrix(None, coords)
    """

    assert self._setup, 'RedIntCoords.setup has to be called before.'

    # Check and assign h-matrix (Cartesian)
    if cart_hmat is None:
      cart_hmat = np.zeros([3,3],dtype=np.float64)
    assert isinstance(cart_hmat,np.ndarray)
    assert cart_hmat.dtype == np.float64
    assert cart_hmat.shape == (3,3)
    self._ric.cart_hmat = cart_hmat

    # Check and assign atomic coordinates (Cartesian)
    assert isinstance(cart_coords,np.ndarray)
    assert cart_coords.dtype == np.float64
    #assert cart_coords.shape == ric.cart_coords.shape
    self._ric.cart_coords[:,:] = np.transpose(cart_coords)

    # Construct B-matrix -- simple RICs
    self._ric.bmat[...] = 0
    # str
    if not self._ric.ric_def_stretches is None:
        _ric_str.ric_stretches_bmat(self._ric.cart_hmat, 
                self._ric.cart_coords,
                self._ric.ric_def_stretches,
                self._ric.ric_ibr_stretches,
                self._ric.bmat,
                self._ric.ric_val_stretches,
                )
    # ibe
    if not self._ric.ric_def_in_bends is None:
        _ric_ibe.ric_in_bends_bmat(self._ric.cart_hmat, 
                self._ric.cart_coords,
                self._ric.ric_def_in_bends,
                self._ric.ric_ibr_in_bends,
                self._ric.bmat,
                self._ric.ric_val_in_bends,
                )
    # obe
    if not self._ric.ric_def_out_bends is None:
        _ric_obe.ric_out_bends_bmat(self._ric.cart_hmat, 
                self._ric.cart_coords,
                self._ric.ric_def_out_bends,
                self._ric.ric_ibr_out_bends,
                self._ric.bmat,
                self._ric.ric_val_out_bends,
                )
    # lbe
    if not self._ric.ric_def_lin_bends is None:
#        import pdb; pdb.set_trace()
        _ric_lbe.ric_lin_bends_bmat(self._ric.cart_hmat, 
                self._ric.cart_coords,
                self._ric.ric_def_lin_bends,
                self._ric.ric_lin_bend_inds,
                self._ric.ric_lin_bend_axes,
                self._ric.ric_ibr_lin_bends,
                self._ric.bmat,
                self._ric.ric_val_lin_bends,
                )
    # tor
    if not self._ric.ric_def_torsions is None:
        _ric_tor.ric_torsions_bmat(self._ric.cart_hmat, 
                self._ric.cart_coords,
                self._ric.ric_def_torsions,
                self._ric.ric_torsion_ivals,
                self._ric.ric_ibr_torsions,
                self._ric.bmat,
                self._ric.ric_val_torsions,
                )
    # eck trans
    if not self._ric.ric_def_eckart_trans is None:
        _ric_eck.ric_eckart_trans_bmat( 
                self._ric.cart_coords,
                self._ric.atomic_masses,
                self._ric.ric_def_eckart_trans,
                self._ric.ric_ibr_eckart_trans,
                self._ric.bmat,
                self._ric.ric_val_eckarts,
            )
    # eck
    if not self._ric.ric_def_eckart_rots is None:
        _ric_eck.ric_eckart_rot_bmat( 
                self._ric.cart_coords,
                self._ric.atomic_masses,
                self._ric.ric_def_eckart_rots,
                self._ric.ric_ibr_eckart_rots,
                self._ric.bmat,
                self._ric.ric_val_eckarts,
            )

#    stat = self._ric.bmat_construct()
#    assert stat == 0

    # Construct B-matrix -- complex RICs
    for ibr, ric in zip(self._ibrs, self._rics):
      self._ric.bmat[:,ibr] += ric.project(cart_coords)

    self._eval = True
    return np.transpose(self._ric.bmat)


  def invert_b_matrix(self):
    """
    Invert B-matrix

    The order of RIC is described in the *construct_b_matrix* documentation.

    Parameters
    ----------
    None

    Returns
    -------
    bmat_inv: (3*N,M) array
              Inverse B-matrix, where N is the number of atom in the system and M is
              the number of RICs.
    rank: int
          Rank of the inverse B-matrix. If rank is smaller than the number of
          Cartesian coordinates (3*N), the set of RICs is incomplete.

    Example
    -------

    >>> bmat_inv, rank = ric.invert_b_matrix()
    """

    # Invert B-matrix
    # TODO Add parameters controlling SVD convergence to avoid numerical
    # instabilities.
    rank, stat = _ric_build.bmat_invert(self._ric.bmat, self._ric.bmat_inv)
#    stat = self._ric.bmat_invert()
    assert stat == 0

#    return self._ric.bmat_inv, self._ric.rank
    return self._ric.bmat_inv 

  def project_hessian(self, cart_hessian):
    """
    Project Hessian matrix from Cartesian coordinates to RICs.

    Parameters
    ----------
    cart_hessian: (3*N, 3*N) array
                  A symmetric Hessian matrix in Cartesian coordinates, where N
                  is the number of atoms in the system.

    Returns
    -------
    ric_hessian: (M, M) array
                 A symmetric Hessian matrix in RICs, where M is the number of
                 RICs.

    Example
    -------

    >>> ric_hess = ric.project_hessian(cart_hess)
    """

    # Check and assign Hessian matrix (Cartesian)
    assert isinstance(cart_hessian,np.ndarray)
    assert cart_hessian.dtype == np.float64
    assert cart_hessian.shape == self._ric.cart_hessian.shape
    #assert np.all(cart_hessian == np.transpose(cart_hessian)) # Slow!!!
    self._ric.cart_hessian[:,:] = cart_hessian
    # Project Cartesian Hessian to RICs
#    import pdb; pdb.set_trace()
    stat = _ric_build.hessian_project(self._ric.bmat, 
            self._ric.bmat_inv,
            self._ric.cart_hessian,
            self._ric.ric_hessian)
#    stat = self._ric.hessian_project()
    assert stat == 0

    return self._ric.ric_hessian

  #============================================================================

  def get_val_stretches(self):
    """Get the values of stretch distances"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_stretches

  def get_val_in_bends(self):
    """Get the values of in-plane bend angles"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_in_bends

  def get_val_out_bends(self):
    """Get the values of out-of-plane bend angles

          D
         /
    B---A
         \\
          C

    Here A is a central atom; B, C, and D -- terminal atoms.

    Return
    ------
    vals: (N,) array
          Array of out-of-plane angles. Values are in radians from 0 to Pi,
          i.e. [0,Pi]. Pi/2 corresponds to a flat configuration, while
          <Pi/2 and >Pi/2 for the B atom being above and bellow
          the plane, respectively.
    """

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_out_bends

  def get_val_lin_bends(self):
    """Get the values of linear bend angles"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_lin_bends

  def get_val_eckarts(self):
    """Get the values of Eckart coordinates

    Return
    ------
    values: (N,) array
            Array of Eckart coordinates values.
    """

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_eckarts

  def get_val_torsions(self):
    """Get the values of torsion dihedral angles

    Return
    ------
    vals: (N,) array
          Array of dihedral angles. Values are in radians from -Pi to Pi
          (included), i.e. (-Pi,Pi].
    """

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_val_torsions

  def get_ric_hessian(self):
    """Get the Hessian matrix in RICs"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    assert self._eval, 'RedIntCoords.construct_b_matrix has to be called before.'
    return self._ric.ric_hessian

  def get_lin_bend_vecs(self):

    assert self._setup, 'RedIntCoords.setup has to be called before.'

    return np.transpose(self._ric.ric_lin_bend_vecs)

  @property
  def num_ric(self):
    """Total number of redundant internal coordinates"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    return self._ric.bmat.shape[1]

  @property
  def num_stretch(self):
    """Number of bond stretches"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_stretches is None:
      return 0
    else:
      return self._ric.ric_val_stretches.size

  @property
  def num_in_bend(self):
    """Number of in-plane bends"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_in_bends is None:
      return 0
    else:
      return self._ric.ric_val_in_bends.size

  @property
  def num_out_bend(self):
    """Number of out-of-plane bends"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_out_bends is None:
      return 0
    else:
      return self._ric.ric_val_out_bends.size

  @property
  def num_lin_bend(self):
    """Number of linear bends"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_lin_bends is None:
      return 0
    else:
      return self._ric.ric_val_lin_bends.size

  @property
  def num_torsion(self):
    """Number of torsions"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_torsions is None:
      return 0
    else:
      return self._ric.ric_val_torsions.size

  @property
  def num_eckart(self):
    """Number of Eckart coordinates"""

    assert self._setup, 'RedIntCoords.setup has to be called before.'
    if self._ric.ric_val_eckarts is None:
      return 0
    else:
      return self._ric.ric_val_eckarts.size


if __name__ == '__main__':
  pass

