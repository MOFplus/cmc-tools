#! /usr/bin/env python

import numpy as np
from .grp import Group

class _Base(object):
  """
    Internal coordinate

    This is a virtual class. DO NOT USE!
  """

  def __init__(self, points):

    self._points = list(points)
    if len(self._points) == 0:
      raise Exception('Empty index list')

    self._indices = []
    for pnt in self._points:
      if    isinstance(pnt, Group):
        self._indices.extend(pnt.indices)
      elif  isinstance(pnt, int):
        if pnt < 1:
          raise Exception('Atomc index has to be positive')
        self._indices.append(pnt-1)
      else:
        raise Exception('Wrong point type')
    self._indices = np.unique(self._indices)

    self._bmat_indices  = np.repeat(3*self._indices, 3)
    self._bmat_indices += np.tile([0,1,2], self._indices.size)

  def __repr__(self):
    return '<' + object.__repr__(self) + ' defined by %s>' % self.points

  @property
  def points(self):
    """Point objects"""
    return self._points

  @property
  def indices(self):
    """Atomic indices (0-based numbering and sorted)"""
    return self._indices

  @property
  def bmat_indices(self):
    """B-matix indices"""
    return self._bmat_indices

  def get_points(self, cart_coords, hmat=None):
    """
      Calculate the position of the points

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.
      Return
      ------
      positions: (M, 3) array
                 Position of points in the Cartesian coordinates, where M is
                 the number of points.
    """

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[0] > np.max(self.indices)
    assert cart_coords.shape[1] == 3

    assert hmat is None

    pnts = np.empty((len(self._points),3), dtype=np.float64)
    for i, pnt in enumerate(self._points):
      if   isinstance(pnt, Group):
        pnts[i,:] = pnt.evaluate(cart_coords, hmat=hmat)
      elif isinstance(pnt, int):
        pnts[i,:] = cart_coords[pnt-1,:]
      else:
        raise Exception('Wrong point type')

    return pnts

  def get_vectors(self, definitions, cart_coords, hmat=None):
    """
      Get vectors defined by pairs of points.

      Parameters
      ----------
      definitions: list of (2,) lists
                   List of vector definitions. Each vector definition consists
                   of two point indices.
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      vectors: (M, 3) array
               Array of vectors, where M is the number of vectors.

      Example
      -------

      >>> a = b.get_vectors([[0, 1], [0, 2]], coords)
      >>> print a[0,:] # A vector from point 0 to 1
      >>> print a[1,:] # A vector from point 0 to 2
    """

    definitions = np.array(definitions, dtype=np.intp)
    assert definitions.ndim == 2
    assert definitions.shape[1] == 2
    assert np.all(definitions >= 0)
    assert np.all(definitions < len(self.points))

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[0] > np.max(self.indices)
    assert cart_coords.shape[1] == 3

    assert hmat is None

    # Compute the point positions
    points = self.get_points(cart_coords, hmat)

    # Compute vectors
    vectors = points[definitions[:,1]] - points[definitions[:,0]]

    return vectors

  def evaluate(self, *argv, **kargs):
    """
      Evaluate

      This method has to be implemented in a derived class
    """
    raise NotImplemented

  def project_points(self, vectors, cart_coords, hmat=None):
    """
      Project point vectors to the atomic components

      Parameters
      ----------
      vectors:     (M, 3) array
                   Point vectors, where M is the number of the points defining
                   the internal coordinate
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      vectors: (3*N, ) array
               Projection of vector points, where N is the number of atoms in
               the system.

      Note
      ----
      Currently *cart_coords* and *hmat* arguments are not used, but required
      as a provision for a more complicated groups.
    """

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (len(self.points),3)

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[0] > np.max(self.indices)
    assert cart_coords.shape[1] == 3

    assert hmat is None

    projection = np.zeros(cart_coords.size, dtype=np.float64)
    for i, pnt in enumerate(self.points):
      if   isinstance(pnt, Group):
        projection[pnt.bmat_indices] += pnt.project(vectors[i])
      elif isinstance(pnt, int):
        ind = 3*(pnt-1)
        projection[ind:ind+3] += vectors[i]
      else:
        raise Exception('Wrong point type')

    return projection

  def project(self, *argv, **kawrgs):
    """
    Project the internal coordinate in the Cartesian space

    This method has to be implemented in a derived class
    """
    raise NotImplemented


if __name__ == '__main__':
  pass

