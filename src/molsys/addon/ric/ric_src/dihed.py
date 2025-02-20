#! /usr/bin/env python

import numpy as np
import numpy.linalg as npl
from .base import _Base

class Dihedral(_Base):
  """
    Dihedral angle

    A       D
     \\     /
      B---C--D'
     /     \\
    A'      D"

    A and A' are left terminal points. D, D' and D" are right terminal points.
    B and C are axial points. The points are defined as atomic positions or
    the geometric centres of atom groups.

    Parameters
    ----------
    left: (N,) array
          List of atom indices or *ric.Group* instances, where N is
          the number of left points.
          The indices have to be from 1 to the number of atoms.
    axis: (2,) array
          List of two atom indices or *ric.Group* instances, defining
          the axis.
          The indices have to be from 1 to the number of atoms.
    right: (M,) array
           List of atom indices or *ric.Group* instances, where M is
           the number of right points.
           The indices have to be from 1 to the number of atoms.
    ivals: (2,) array, optional
           The index of the left and right point used to calculate
           the dihedral angle. See examples of *ric.Dihedral.evaluate*.
           The indices have to be from 1 to the number of the corresponding
           points.

    Example
    -------

     1       6
      \\     /
    2--4---5
      /     \\
     3       7

    >>> d = Dihedral(left=[1,2,3], axis=[4,5], right=[6,7])

  """

  def __init__(self, left, axis, right, ivals=[1,1]):

    self._left  = list(left)
    self._axis  = list(axis)
    self._right = list(right)
    self._ivals = list(ivals)

    # Check the number of points and ivals
    if len(self._left) < 1:
      raise Exception('One or more point are need')
    if len(self._axis) != 2:
      raise Exception('Two points are need')
    if len(self._right) < 1:
      raise Exception('One or more point are need')
    if len(self._ivals) != 2:
      raise Exception('Two "ivals" are needed')

    # Call ric._Base.__init__
    points = self._left + self._axis + self._right
    super(self.__class__, self).__init__(points)

    # Construct vector point indices
    self._iaxis   = (len(self._left), len(self._left)+1)
    self._ilefts  = [(self._iaxis[0], i) for i in range(len(self._left))]
    self._irights = [(self._iaxis[1], i) for i in range(len(self._left)+2,len(points))]

  def evaluate(self, cart_coords, hmat=None, ivals=None):
    """
      Compute the value of the dihedral angle.

#
#

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.
      ivals: (2,) array, optional
             The index of the left and right point used to calculate
             the dihedral angle. The indices have to be from 1 to the number
             of the corresponding points.

      Return
      ------
      distance: float
                Dihedral angle in radians.

    Example
    -------

    1
     \\
      2---3
           \\
            4

    >>> d = Dihedral([1], [2,3], [4])
    >>> v = d.evaluate(coords) # Compute 1--2--3--4 angle

     1       6
      \\     /
    2--4---5
      /     \\
     3       7

    >>> d = Dihedral([1,2,3], [4,5], [6,7])
    >>> v = d.evaluate(coords) # Compute 1--4--5--6 angle
    >>> v = d.evaluate(coords, ivals=[3,1]) # Compute 3--4--5--6 angle

    >>> d = Dihedral([1,2,3], [4,5], [6,7], ivals=[1,2])
    >>> v = d.evaluate(coords) # Compute 1--4--5--7 angle
    >>> v = d.evaluate(coords, ivals=[3,1]) # Compute 3--4--5--6 angle
    """

    if not ivals is None:
      self._ivals = list(ivals)
      if len(self._ivals) != 2:
        raise Exception('Two "ivals" are needed')

    # Get dihedral axis and vectors
    ivecs = [self._iaxis] + [(self._iaxis[0], self._ivals[0]-1),
                             (self._iaxis[1], self._ivals[1]+len(self._left)+1)]
    axis, v1, v2 = self.get_vectors(ivecs, cart_coords, hmat)

    # Compute normal vectors
    n1, n2 = np.cross(axis, v1), np.cross(axis, v2)
    n1, n2 = n1/npl.norm(n1), n2/npl.norm(n2)

    # Compute angle
    angl = np.sum(n1*n2)
    angl = np.fmin(np.fmax(-1,angl),1.)
    angl = np.arccos(angl)
    if np.sum(np.cross(n1, n2)*axis) > 0:
      angl *= -1

    return angl

  def project(self, cart_coords, hmat=None):
    """
      Evaluate the projection vector to the atomic components.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      projection: (3*N,) array
                  Projection vector, where N is the number of atoms in
                  the system.
    """
    # Get vectors between the points and the axis vector
    ivecs = [self._iaxis]+self._ilefts+self._irights
    vecs  = self.get_vectors(ivecs, cart_coords, hmat)

    axis    = vecs[0]
    d_axis  = npl.norm(axis)
    axis   /= d_axis

    left_vecs, right_vecs = vecs[1:len(self._ilefts)+1], vecs[len(self._ilefts)+1:]

    # Compute point vectors
    # TODO clean up and remove repetitive code!
    vectors = np.zeros((len(self.points),3), dtype=np.float64)

    for ileft, vec in zip(self._ilefts,left_vecs):
      d_vec  = npl.norm(vec)
      vec   /= d_vec
      dot    = np.sum(axis*vec)

      n  = np.cross(axis, vec)/(d_vec*d_axis)
      n /= (1. - dot**2)*len(self._ilefts)

      vectors[ileft[1]      ]  = n* d_axis
      vectors[self._iaxis[0]] += n*(d_vec*dot - d_axis)
      vectors[self._iaxis[1]] += n*(- d_vec*dot)

    axis *= -1
    for iright, vec in zip(self._irights,right_vecs):
      d_vec  = npl.norm(vec)
      vec   /= d_vec
      dot    = np.sum(axis*vec)

      n  = np.cross(axis, vec)/(d_axis*d_vec)
      n /= (1. - dot**2)*len(self._irights)

      vectors[iright[1]     ]  = n* d_axis
      vectors[self._iaxis[1]] += n*(d_vec*dot - d_axis)
      vectors[self._iaxis[0]] += n*(-d_vec*dot)

    # Project the point vectors to atomic components
    projections = self.project_points(vectors, cart_coords, hmat)

    return projections


if __name__ == '__main__':
  pass

