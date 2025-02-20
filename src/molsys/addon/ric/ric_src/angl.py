#! /usr/bin/env python

import numpy as np
import numpy.linalg as npl
from .grp import Group
from .base import _Base

class Angle(_Base):
  """
    Planar or linear angle

    A---B          A---B---C
         \\    or      /
          C          V

    A and C are terminal points, and B -- central. The points are defined as
    atomic positions or the geometric centres of atom groups.

    An angle axis (V) is a vector perpendicular to A--B and B--C. In case of
    liner angle, the axis has to specified:
      * explicit vector;
      * Cartesian basis vector (x, y, or z);
      * from a reference atom or atom group (G).

    Note, only the last option gives the translational and rotational
    invariance. The axis vector can be parallel or perpendicular to
    the plane ACG.

    Parameters
    ----------
    points: (3,) array
            List of three atom indices or *ric.Group* instances.
            The indices have to be from 1 to the number of atoms.
    axis: (3,) array or string or *ric.Group* instance or integer, optional
          Only relevant for a linear angle. Axis vector specified as
          an explicit array, Cartesian basis vector ('x', 'y', 'z'),
          *ric.Group* instance, or atomic index. The index has to be from
          1 to the number of atoms.
    inplane: bool, optional
             Only relevant if *axis* is set to an atomic index or *ric.Group*
             instance. If *True*, the axis vector is parallel to the plane,
             otherwise, if *False*, it is perpendicular.

    Example
    -------

    1---2
         \\
          3

    >>> a = Angle([1,2,3])


    1---2---3
       /
      V

    >>> a = Angle([1,2,3], axis=np.array([1,1,1]))


    1---2---3
       /
      X

    >>> a = Angle([1,2,3], axis='x')


    1---2---3
         \\
          g

    >>> g = Group([4,5])
    >>> a = Angle([1,2,3], axis=g)
  """

  def __init__(self, points, axis=None, inplane=True):

    # Call ric._Base.__init__
    super(self.__class__, self).__init__(points)

    # Check the number of points
    if len(self.points) != 3:
      raise Exception('Wrong number of points')

    # Set the method to compute the axis
    self._axis = axis
    self._inplane = bool(inplane)

    # The axis computed from angle vectors
    if   self._axis is None:
      pass # Do nothing

    # The axis is given explicitly
    elif isinstance(self._axis, np.ndarray):
      if self._axis.shape != (3,):
        raise Exception('Invalid axis')
      self._axis /= npl.norm(self._axis)

    # The axis is a Cartesian basis vector
    elif self._axis == 'x':
      self._axis = np.array([1,0,0], dtype=np.float64)
    elif self._axis == 'y':
      self._axis = np.array([0,1,0], dtype=np.float64)
    elif self._axis == 'z':
      self._axis = np.array([0,0,1], dtype=np.float64)

    # The axis is computed from a reference atom
    elif isinstance(self._axis, int):
      if self._axis < 1:
        raise Exception('Invalid atom index')
      self._axis = Group([self._axis])

    # The axis is computed from a reference atom group
    elif isinstance(self._axis, Group):
       pass # Do nothing

    else:
      raise Exception('Undefined axis')

  def get_axis(self, cart_coords, hmat=None):
    """
      Compute axis vector.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      axis: (3,) array
            Axis vector (normalized)
    """

    # The axis is computed from angle vectors
    if   self._axis is None:
      a, b = self.get_vectors([[0,1],[1,2]], cart_coords, hmat)
      axis = np.cross(a,b)
      axis /= npl.norm(axis)

    # The axis is given explicitly
    elif isinstance(self._axis, np.ndarray):
      assert self._axis.shape == (3,)
      axis = self._axis

    # The axis is computed from the reference atom group
    elif isinstance(self._axis, Group):
       ref = self._axis.evaluate(cart_coords, hmat) - self.get_points(cart_coords, hmat)[1]
       vec = self.get_vectors([[0,2]], cart_coords, hmat)[0]
       axis = np.cross(ref,vec)
       if not self._inplane:
         axis = np.cross(axis,vec)
       axis /= npl.norm(axis)

    else:
      raise Exception('Undefined axis')

    return axis

  def evaluate(self, cart_coords, hmat=None):
    """
      Compute the value of the angle.

      For a planar angle (*axis* is not set), the value is from the range [0, pi].
      If the vectors AB and BC pointing to the same directions, it is 0.
      It they are opposite, it is pi.

      For a linear angle (*axis* is set), the value is from the range (-pi, pi].
      If the angle is exactly liner, it is 0. If the angle is bend
      counter-clockwise (the axis is pointing to an observer), it is positive.
      If the angle is bend clockwise, it is negative.

      Parameters
      ----------
      cart_coords: (N, 3) array
                   Atomic positions in Cartesian coordinates, where N is the
                   number of atoms in the system.

      Return
      ------
      distance: float
                Angle in radians.
    """

    # Get vectors between the points
    a, b = self.get_vectors([[1,0],[1,2]], cart_coords, hmat)

    # Transform the vectors for the linear angle
    if not self._axis is None:
      a *= -1
      axis = self.get_axis(cart_coords, hmat)
      a, b = np.cross(a, axis), np.cross(b, axis)

    # Compute the angle
    a, b = a/npl.norm(a), b/npl.norm(b)
    angl = np.sum(a*b)
    angl = np.fmin(np.fmax(-1,angl),1.)
    angl = np.arccos(angl)

    # Set a sing for the linear angle
    if not self._axis is None:
      if np.sum(np.cross(a, b)*axis) < 0:
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
    a, b = self.get_vectors([[1,0],[1,2]], cart_coords, hmat)
    axis = self.get_axis(cart_coords, hmat)

    # Modify the vectors
    if self._axis is None:
      b *= -1 # Planar angle
    else:
      a *= -1 # Linear angle

    # Compute point vectors
    vectors = np.empty((3,3), dtype=np.float64)
    vectors[0]  =  np.cross(axis, a)
    vectors[2]  =  np.cross(axis, b)
    vectors[0] /=  npl.norm(vectors[0])**2
    vectors[2] /=  npl.norm(vectors[2])**2
    vectors[1]  = -(vectors[0]+vectors[2])

    # Project the point vectors to atomic components
    projections = self.project_points(vectors, cart_coords, hmat)

    return projections


if __name__ == '__main__':
  pass

