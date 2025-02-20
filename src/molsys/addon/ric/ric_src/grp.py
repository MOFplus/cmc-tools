#! /usr/bin/env python

import numpy as np

class Group(object):
  """
    Atom group

    Parameters
    ----------
    indices: list
             List of atomic indices. The indices can be to be from 1 to
             the total number of atoms.

    Example
    -------

    >>> grp = Group([1,2,3])
    >>> print grp.indices
    >>> print grp.evaluate(cart_coords)
    >>> bmat[grp.bmat_indices,0] += grp.projection(vector)
  """

  def __init__(self, indices):

    # Get atomic indices
    self._indices = np.require(indices, dtype=np.intp)
    self._indices = np.sort(self._indices)
    assert self._indices.ndim == 1
    assert self._indices.size > 0
    assert np.all(self._indices > 0)
    assert np.unique(self._indices).size == self._indices.size, \
      'Duplicate indices'

    # Convert to 0-based numbering
    self._indices -= 1

    # Compute B-matrix indices
    self._bmat_indices  = np.repeat(3*self._indices, 3)
    self._bmat_indices += np.tile([0,1,2], self._indices.size)

  def __repr__(self):
    return '<' + object.__repr__(self) + ' of %s atoms>' % (self.indices+1)

  def __len__(self):
    return self._indices.size

  @property
  def indices(self):
    """Atomic indices (0-based numbering and sorted)"""
    return self._indices

  @property
  def bmat_indices(self):
    """
    Atomic indices for B-matrix (0-based numbering).
    It is for a convenience to fill B-matrix
    """

    return self._bmat_indices

  def evaluate(self, cart_coords, hmat=None):
    """
      Compute the characteristic vector from the given atomic position *cart_coords*.
      Currently it is just a geometric centre of the constituting atoms.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      vector: (3,) array
              Geometric centre
    """

    assert isinstance(cart_coords, np.ndarray)
    assert cart_coords.ndim == 2
    assert cart_coords.shape[1] == 3

    assert hmat is None

    # Compute geometric centre
    value = np.mean(cart_coords[self._indices], axis=0)

    return value

  def project(self, vector):
    """
      Project a *vector* to atomic components.

      In case of the geometric centre, the vector is divided by the number of
      atoms and assigned for each atomic componet.

      Parameters
      ----------
      vector: (3,) array
              Projected vector

      Return
      ------
      projection: (3*N,) array
                  Projection of the *vector* in atomic components, where N is
                  the number of atoms.

    """

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (3,)

    projections = np.tile(vector/float(len(self)), len(self))

    return projections


if __name__ == '__main__':
  pass

