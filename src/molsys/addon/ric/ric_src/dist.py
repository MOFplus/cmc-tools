#! /usr/bin/env python

import numpy as np
import numpy.linalg as npl
from .base import _Base

class Distance(_Base):
  """
    Distance

    A--B

    A and B are points. The points are defined as atomic positions or
    the geometric centres of atom groups.

    Parameters
    ----------
    points: (2,) array
            List of two atom indices or *ric.Group* instances.
            The indices have to be from 1 to the number of atoms.

    Example
    -------

    >>> g = Group([1,2,4])
    >>> d = Distance([g,3])
  """

  def __init__(self, points):

    # Call ric._Base.__init__
    super(self.__class__, self).__init__(points)

    # Check the number of points
    if len(self.points) != 2:
      raise Exception('Wrong number of points')

  def evaluate(self, cart_coords, hmat=None):
    """
      Compute the value of the distance.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      distance: float
                Distance in the same unit as *cart_coords*.
    """

    # Get a vectors between the two points
    a = self.get_vectors([[0,1]], cart_coords, hmat)

    # Compute the distance
    distance = npl.norm(a)

    return distance

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

    # Get a vector between the two points
    a = self.get_vectors([[1,0]], cart_coords, hmat)

    # Compute point vectors
    vectors = np.empty((2,3), dtype=np.float64)
    vectors[0] = a/npl.norm(a)
    vectors[1] = -vectors[0]

    # Project the point vectors to atomic components
    projections = self.project_points(vectors, cart_coords, hmat)

    return projections


if __name__ == '__main__':
  pass

