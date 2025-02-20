# -*- coding: utf-8 -*-
"""

    pgp

    shamelessly but gratefully "adapted" (if you like to say so) from:
        pymatgen.symmetry.analyzer,
        pymatgen.core.operations
    which are on turn based on a bunch of published or well-known algorithms.
    
    Copyright (c) Pymatgen Development Team.
    Distributed under the terms of the MIT License.

Created on Fri Jul  7 11:52:42 2019

@author: roberto
"""

import molsys

import numpy as np
import itertools
from collections import defaultdict
from math import cos, sin, sqrt
from math import pi
import copy

import logging
logger = logging.getLogger("molsys.ptg")
logger.setLevel(logging.INFO)

def find_in_coord_list(coord_list, coord, atol=1e-8):
    """
    Find the indices of matches of a particular coord in a coord_list.

    Args:
        coord_list: List of coords to test
        coord: Specific coordinates
        atol: Absolute tolerance. Defaults to 1e-8. Accepts both scalar and
            array.

    Returns:
        Indices of matches, e.g., [0, 1, 2, 3]. Empty list if not found.
    """
    if len(coord_list) == 0:
        return []
    diff = np.array(coord_list) - np.array(coord)[None, :]
    return np.where(np.all(np.abs(diff) < atol, axis=1))[0]

################################################################################

class SymmOp(object):
    """
    A symmetry operation in cartesian space. Consists of a rotation plus a
    translation. Implementation is as an affine transformation matrix of rank 4
    for efficiency. Read: http://en.wikipedia.org/wiki/Affine_transformation.

    .. attribute:: affine_matrix

        A 4x4 numpy.array representing the symmetry operation.
    """

    def __init__(self, affine_transformation_matrix, tol=0.01):
        """
        Initializes the SymmOp from a 4x4 affine transformation matrix.
        In general, this constructor should not be used unless you are
        transferring rotations.  Use the static constructors instead to
        generate a SymmOp from proper rotations and translation.

        Args:
            affine_transformation_matrix (4x4 array): Representing an
                affine transformation.
            tol (float): Tolerance for determining if matrices are equal.
        """
        affine_transformation_matrix = np.array(affine_transformation_matrix)
        if affine_transformation_matrix.shape != (4, 4):
            raise ValueError("Affine Matrix must be a 4x4 numpy array!")
        self.affine_matrix = affine_transformation_matrix
        self.tol = tol

    @staticmethod
    def from_rotation_and_translation(
            rotation_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            translation_vec=(0, 0, 0), tol=0.1):
        """
        Creates a symmetry operation from a rotation matrix and a translation
        vector.

        Args:
            rotation_matrix (3x3 array): Rotation matrix.
            translation_vec (3x1 array): Translation vector.
            tol (float): Tolerance to determine if rotation matrix is valid.

        Returns:
            SymmOp object
        """
        rotation_matrix = np.array(rotation_matrix)
        translation_vec = np.array(translation_vec)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation Matrix must be a 3x3 numpy array.")
        if translation_vec.shape != (3,):
            raise ValueError("Translation vector must be a rank 1 numpy array "
                             "with 3 elements.")
        affine_matrix = np.eye(4)
        affine_matrix[0:3][:, 0:3] = rotation_matrix
        affine_matrix[0:3][:, 3] = translation_vec
        return SymmOp(affine_matrix, tol)

    def __eq__(self, other):
        return np.allclose(self.affine_matrix, other.affine_matrix,
                           atol=self.tol)

    def __hash__(self):
        return 7

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        output = ["Rot:", str(self.affine_matrix[0:3][:, 0:3]), "tau",
                  str(self.affine_matrix[0:3][:, 3])]
        return "\n".join(output)

    def operate(self, point):
        """
        Apply the operation on a point.

        Args:
            point: Cartesian coordinate.

        Returns:
            Coordinates of point after operation.
        """
        affine_point = np.array([point[0], point[1], point[2], 1])
        return np.dot(self.affine_matrix, affine_point)[0:3]

    def operate_multi(self, points):
        """
        Apply the operation on a list of points.

        Args:
            points: List of Cartesian coordinates

        Returns:
            Numpy array of coordinates after operation
        """
        points = np.array(points)
        affine_points = np.concatenate(
            [points, np.ones(points.shape[:-1] + (1,))], axis=-1)
        return np.inner(affine_points, self.affine_matrix)[..., :-1]

    def apply_rotation_only(self, vector):
        """
        Vectors should only be operated by the rotation matrix and not the
        translation vector.

        Args:
            vector (3x1 array): A vector.
        """
        return np.dot(self.rotation_matrix, vector)

    def transform_tensor(self, tensor):
        """
        Applies rotation portion to a tensor. Note that tensor has to be in
        full form, not the Voigt form.

        Args:
            tensor (numpy array): a rank n tensor

        Returns:
            Transformed tensor.
        """
        dim = tensor.shape
        rank = len(dim)
        assert all([i == 3 for i in dim])
        # Build einstein sum string
        lc = string.ascii_lowercase
        indices = lc[:rank], lc[rank:2 * rank]
        einsum_string = ','.join([a + i for a, i in zip(*indices)])
        einsum_string += ',{}->{}'.format(*indices[::-1])
        einsum_args = [self.rotation_matrix] * rank + [tensor]

        return np.einsum(einsum_string, *einsum_args)

    def are_symmetrically_related(self, point_a, point_b, tol=0.001):
        """
        Checks if two points are symmetrically related.

        Args:
            point_a (3x1 array): First point.
            point_b (3x1 array): Second point.
            tol (float): Absolute tolerance for checking distance.

        Returns:
            True if self.operate(point_a) == point_b or vice versa.
        """
        if np.allclose(self.operate(point_a), point_b, atol=tol):
            return True
        if np.allclose(self.operate(point_b), point_a, atol=tol):
            return True
        return False

    @property
    def rotation_matrix(self):
        """
        A 3x3 numpy.array representing the rotation matrix.
        """
        return self.affine_matrix[0:3][:, 0:3]

    @property
    def translation_vector(self):
        """
        A rank 1 numpy.array of dim 3 representing the translation vector.
        """
        return self.affine_matrix[0:3][:, 3]

    def __mul__(self, other):
        """
        Returns a new SymmOp which is equivalent to apply the "other" SymmOp
        followed by this one.
        """
        new_matrix = np.dot(self.affine_matrix, other.affine_matrix)
        return SymmOp(new_matrix)

    @property
    def inverse(self):
        """
        Returns inverse of transformation.
        """
        invr = np.linalg.inv(self.affine_matrix)
        return SymmOp(invr)

    @staticmethod
    def from_axis_angle_and_translation(axis, angle, angle_in_radians=False,
                                        translation_vec=(0, 0, 0)):
        """
        Generates a SymmOp for a rotation about a given axis plus translation.

        Args:
            axis: The axis of rotation in cartesian space. For example,
                [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.
            translation_vec: A translation vector. Defaults to zero.

        Returns:
            SymmOp for a rotation about given axis and translation.
        """
        if isinstance(axis, (tuple, list)):
            axis = np.array(axis)

        if isinstance(translation_vec, (tuple, list)):
            vec = np.array(translation_vec)
        else:
            vec = translation_vec

        a = angle if angle_in_radians else angle * pi / 180
        cosa = cos(a)
        sina = sin(a)
        u = axis / np.linalg.norm(axis)
        r = np.zeros((3, 3))
        r[0, 0] = cosa + u[0] ** 2 * (1 - cosa)
        r[0, 1] = u[0] * u[1] * (1 - cosa) - u[2] * sina
        r[0, 2] = u[0] * u[2] * (1 - cosa) + u[1] * sina
        r[1, 0] = u[0] * u[1] * (1 - cosa) + u[2] * sina
        r[1, 1] = cosa + u[1] ** 2 * (1 - cosa)
        r[1, 2] = u[1] * u[2] * (1 - cosa) - u[0] * sina
        r[2, 0] = u[0] * u[2] * (1 - cosa) - u[1] * sina
        r[2, 1] = u[1] * u[2] * (1 - cosa) + u[0] * sina
        r[2, 2] = cosa + u[2] ** 2 * (1 - cosa)

        return SymmOp.from_rotation_and_translation(r, vec)

    @staticmethod
    def from_origin_axis_angle(origin, axis, angle, angle_in_radians=False):
        """
        Generates a SymmOp for a rotation about a given axis through an
        origin.

        Args:
            origin (3x1 array): The origin which the axis passes through.
            axis (3x1 array): The axis of rotation in cartesian space. For
                example, [1, 0, 0]indicates rotation about x-axis.
            angle (float): Angle of rotation.
            angle_in_radians (bool): Set to True if angles are given in
                radians. Or else, units of degrees are assumed.

        Returns:
            SymmOp.
        """
        theta = angle * pi / 180 if not angle_in_radians else angle
        a = origin[0]
        b = origin[1]
        c = origin[2]
        u = axis[0]
        v = axis[1]
        w = axis[2]
        # Set some intermediate values.
        u2 = u * u
        v2 = v * v
        w2 = w * w
        cos_t = cos(theta)
        sin_t = sin(theta)
        l2 = u2 + v2 + w2
        l = sqrt(l2)

        # Build the matrix entries element by element.
        m11 = (u2 + (v2 + w2) * cos_t) / l2
        m12 = (u * v * (1 - cos_t) - w * l * sin_t) / l2
        m13 = (u * w * (1 - cos_t) + v * l * sin_t) / l2
        m14 = (a * (v2 + w2) - u * (b * v + c * w) +
               (u * (b * v + c * w) - a * (v2 + w2)) * cos_t +
               (b * w - c * v) * l * sin_t) / l2

        m21 = (u * v * (1 - cos_t) + w * l * sin_t) / l2
        m22 = (v2 + (u2 + w2) * cos_t) / l2
        m23 = (v * w * (1 - cos_t) - u * l * sin_t) / l2
        m24 = (b * (u2 + w2) - v * (a * u + c * w) +
               (v * (a * u + c * w) - b * (u2 + w2)) * cos_t +
               (c * u - a * w) * l * sin_t) / l2

        m31 = (u * w * (1 - cos_t) - v * l * sin_t) / l2
        m32 = (v * w * (1 - cos_t) + u * l * sin_t) / l2
        m33 = (w2 + (u2 + v2) * cos_t) / l2
        m34 = (c * (u2 + v2) - w * (a * u + b * v) +
               (w * (a * u + b * v) - c * (u2 + v2)) * cos_t +
               (a * v - b * u) * l * sin_t) / l2

        return SymmOp([[m11, m12, m13, m14], [m21, m22, m23, m24],
                       [m31, m32, m33, m34], [0, 0, 0, 1]])

    @staticmethod
    def reflection(normal, origin=(0, 0, 0)):
        """
        Returns reflection symmetry operation.

        Args:
            normal (3x1 array): Vector of the normal to the plane of
                reflection.
            origin (3x1 array): A point in which the mirror plane passes
                through.

        Returns:
            SymmOp for the reflection about the plane
        """
        # Normalize the normal vector first.
        n = np.array(normal, dtype=float) / np.linalg.norm(normal)

        u, v, w = n

        translation = np.eye(4)
        translation[0:3, 3] = -np.array(origin)

        xx = 1 - 2 * u ** 2
        yy = 1 - 2 * v ** 2
        zz = 1 - 2 * w ** 2
        xy = -2 * u * v
        xz = -2 * u * w
        yz = -2 * v * w
        mirror_mat = [[xx, xy, xz, 0], [xy, yy, yz, 0], [xz, yz, zz, 0],
                      [0, 0, 0, 1]]

        if np.linalg.norm(origin) > 1e-6:
            mirror_mat = np.dot(np.linalg.inv(translation),
                                np.dot(mirror_mat, translation))
        return SymmOp(mirror_mat)

    @staticmethod
    def inversion(origin=(0, 0, 0)):
        """
        Inversion symmetry operation about axis.

        Args:
            origin (3x1 array): Origin of the inversion operation. Defaults
                to [0, 0, 0].

        Returns:
            SymmOp representing an inversion operation about the origin.
        """
        mat = -np.eye(4)
        mat[3, 3] = 1
        mat[0:3, 3] = 2 * np.array(origin)
        return SymmOp(mat)

    @staticmethod
    def rotoreflection(axis, angle, origin=(0, 0, 0)):
        """
        Returns a roto-reflection symmetry operation

        Args:
            axis (3x1 array): Axis of rotation / mirror normal
            angle (float): Angle in degrees
            origin (3x1 array): Point left invariant by roto-reflection.
                Defaults to (0, 0, 0).

        Return:
            Roto-reflection operation
        """
        rot = SymmOp.from_origin_axis_angle(origin, axis, angle)
        refl = SymmOp.reflection(axis, origin)
        m = np.dot(rot.affine_matrix, refl.affine_matrix)
        return SymmOp(m)

    def as_dict(self):
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "matrix": self.affine_matrix.tolist(), "tolerance": self.tol}
        return d

    def as_xyz_string(self):
        """
        Returns a string of the form 'x, y, z', '-x, -y, z',
        '-y+1/2, x+1/2, z+1/2', etc. Only works for integer rotation matrices
        """
        xyz = ['x', 'y', 'z']
        strings = []

        # test for invalid rotation matrix
        if not np.all(np.isclose(self.rotation_matrix,
                                 np.round(self.rotation_matrix))):
            warnings.warn('Rotation matrix should be integer')

        return transformation_to_string(self.rotation_matrix, translation_vec=self.translation_vector, delim=", ")

    @staticmethod
    def from_xyz_string(xyz_string):
        """
        Args:
            xyz_string: string of the form 'x, y, z', '-x, -y, z',
                '-2y+1/2, 3x+1/2, z-y+1/2', etc.
        Returns:
            SymmOp
        """
        rot_matrix = np.zeros((3, 3))
        trans = np.zeros(3)
        toks = xyz_string.strip().replace(" ", "").lower().split(",")
        re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
        re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")
        for i, tok in enumerate(toks):
            # build the rotation matrix
            for m in re_rot.finditer(tok):
                factor = -1 if m.group(1) == "-" else 1
                if m.group(2) != "":
                    factor *= float(m.group(2)) / float(m.group(3)) \
                        if m.group(3) != "" else float(m.group(2))
                j = ord(m.group(4)) - 120
                rot_matrix[i, j] = factor
            # build the translation vector
            for m in re_trans.finditer(tok):
                factor = -1 if m.group(1) == "-" else 1
                num = float(m.group(2)) / float(m.group(3)) \
                    if m.group(3) != "" else float(m.group(2))
                trans[i] = num * factor
        return SymmOp.from_rotation_and_translation(rot_matrix, trans)

    @classmethod
    def from_dict(cls, d):
        return cls(d["matrix"], d["tolerance"])

################################################################################

class ptg: # formerly known as PointGroupAnalyzer
    """
    A class to analyze the point group of a molecule. The general outline of
    the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.

        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis.  All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.

    .. attribute:: sch_symbol

        Schoenflies symbol of the detected point group.
    """
    inversion_op = SymmOp.inversion()

    def __init__(self, mol, tolerance=0.3, eigen_tolerance=0.01,
                 matrix_tol=0.1):
        """
        The default settings are usually sufficient.

        Args:
            mol (Molecule): Molecule to determine point group for.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            eigen_tolerance (float): Tolerance to compare eigen values of
                the inertia tensor. Defaults to 0.01.
            matrix_tol (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        self.mol = mol
        self.ptg_version = "237b24f"
        logger.info("Addon pgp loaded (version %s)" % self.ptg_version)
        self.cmol = copy.deepcopy(mol)
        self.cmol.center_coc()
        self.cmol.set_real_mass()

    def analyze(self, tol=0.3, eig_tol=0.01, mat_tol=0.1):
        self.tol = tol
        self.eig_tol = eig_tol
        self.mat_tol = mat_tol
        self._analyze()
        if self.sch_symbol in ["C1v", "C1h"]:
            self.sch_symbol = "Cs"
        self.setup_pointgroup()

    def _analyze(self):
        if self.cmol.natoms == 1:
            self.sch_symbol = "Kh"
        else:
            inertia_tensor = np.zeros((3, 3))
            total_inertia = 0
            for i in range(self.cmol.natoms):
                c = self.cmol.xyz[i]
                wt = self.cmol.amass[i]
                for i in range(3):
                    inertia_tensor[i, i] += wt * (c[(i + 1) % 3] ** 2
                                                  + c[(i + 2) % 3] ** 2)
                for i, j in [(0, 1), (1, 2), (0, 2)]:
                    inertia_tensor[i, j] += -wt * c[i] * c[j]
                    inertia_tensor[j, i] += -wt * c[j] * c[i]
                total_inertia += wt * np.dot(c, c)

            # Normalize the inertia tensor so that it does not scale with size
            # of the system.  This mitigates the problem of choosing a proper
            # comparison tolerance for the eigenvalues.
            inertia_tensor /= total_inertia
            eigvals, eigvecs = np.linalg.eig(inertia_tensor)
            self.principal_axes = eigvecs.T
            self.eigvals = eigvals
            v1, v2, v3 = eigvals
            eig_zero = abs(v1 * v2 * v3) < self.eig_tol ** 3
            eig_all_same = abs(v1 - v2) < self.eig_tol and abs(
                v1 - v3) < self.eig_tol
            eig_all_diff = abs(v1 - v2) > self.eig_tol and abs(
                v1 - v3) > self.eig_tol and abs(v2 - v3) > self.eig_tol

            self.rot_sym = []
            self.symmops = [SymmOp(np.eye(4))]
            if eig_zero:
                logger.debug("Linear molecule detected")
                self._proc_linear()
            elif eig_all_same:
                logger.debug("Spherical top molecule detected")
                self._proc_sph_top()
            elif eig_all_diff:
                logger.debug("Asymmetric top molecule detected")
                self._proc_asym_top()
            else:
                logger.debug("Symmetric top molecule detected")
                self._proc_sym_top()

    def _proc_linear(self):
        if self.is_valid_op(ptg.inversion_op):
            self.sch_symbol = "D*h"
            self.symmops.append(ptg.inversion_op)
        else:
            self.sch_symbol = "C*v"

    def _proc_asym_top(self):
        """
        Handles assymetric top molecules, which cannot contain rotational
        symmetry larger than 2.
        """
        self._check_R2_axes_asym()
        if len(self.rot_sym) == 0:
            logger.debug("No rotation symmetries detected.")
            self._proc_no_rot_sym()
        elif len(self.rot_sym) == 3:
            logger.debug("Dihedral group detected.")
            self._proc_dihedral()
        else:
            logger.debug("Cyclic group detected.")
            self._proc_cyclic()

    def _proc_sym_top(self):
        """
        Handles symetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.  More complex
        handling required to look for R2 axes perpendicular to this unique
        axis.
        """
        if abs(self.eigvals[0] - self.eigvals[1]) < self.eig_tol:
            ind = 2
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.eig_tol:
            ind = 0
        else:
            ind = 1
        logger.debug("Eigenvalues = %s." % self.eigvals)
        unique_axis = self.principal_axes[ind]
        self._check_rot_sym(unique_axis)
        logger.debug("Rotation symmetries = %s" % self.rot_sym)
        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(unique_axis)

        if len(self.rot_sym) >= 2:
            self._proc_dihedral()
        elif len(self.rot_sym) == 1:
            self._proc_cyclic()
        else:
            self._proc_no_rot_sym()

    def _proc_no_rot_sym(self):
        """
        Handles molecules with no rotational symmetry. Only possible point
        groups are C1, Cs and Ci.
        """
        self.sch_symbol = "C1"
        if self.is_valid_op(ptg.inversion_op):
            self.sch_symbol = "Ci"
            self.symmops.append(ptg.inversion_op)
        else:
            for v in self.principal_axes:
                mirror_type = self._find_mirror(v)
                if not mirror_type == "":
                    self.sch_symbol = "Cs"
                    break

    def _proc_cyclic(self):
        """
        Handles cyclic group molecules.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = "C{}".format(rot)
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif mirror_type == "v":
            self.sch_symbol += "v"
        elif mirror_type == "":
            if self.is_valid_op(SymmOp.rotoreflection(main_axis,
                                                      angle=180 / rot)):
                self.sch_symbol = "S{}".format(2 * rot)

    def _proc_dihedral(self):
        """
        Handles dihedral group molecules, i.e those with intersecting R2 axes
        and a main axis.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = "D{}".format(rot)
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == "h":
            self.sch_symbol += "h"
        elif not mirror_type == "":
            self.sch_symbol += "d"

    def _check_R2_axes_asym(self):
        """
        Test for 2-fold rotation along the principal axes. Used to handle
        asymetric top molecules.
        """
        for v in self.principal_axes:
            op = SymmOp.from_axis_angle_and_translation(v, 180)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((v, 2))

    def _find_mirror(self, axis):
        """
        Looks for mirror symmetry of specified type about axis.  Possible
        types are "h" or "vd".  Horizontal (h) mirrors are perpendicular to
        the axis while vertical (v) or diagonal (d) mirrors are parallel.  v
        mirrors has atoms lying on the mirror plane while d mirrors do
        not.
        """
        mirror_type = ""

        # First test whether the axis itself is the normal to a mirror plane.
        if self.is_valid_op(SymmOp.reflection(axis)):
            self.symmops.append(SymmOp.reflection(axis))
            mirror_type = "h"
        else:
            # Iterate through all pairs of atoms to find mirror
            for i, j in itertools.combinations(range(self.cmol.natoms), 2):
                if self.cmol.elems[i] == self.cmol.elems[j]:
                    normal = self.cmol.xyz[i] - self.cmol.xyz[j]
                    if np.dot(normal, axis) < self.tol:
                        op = SymmOp.reflection(normal)
                        if self.is_valid_op(op):
                            self.symmops.append(op)
                            if len(self.rot_sym) > 1:
                                mirror_type = "d"
                                for v, r in self.rot_sym:
                                    if not np.linalg.norm(v - axis) < self.tol:
                                        if np.dot(v, normal) < self.tol:
                                            mirror_type = "v"
                                            break
                            else:
                                mirror_type = "v"
                            break

        return mirror_type

    def _get_smallest_set_not_on_axis(self, axis):
        """
        Returns the smallest list of atoms with the same species and
        distance from origin AND does not lie on the specified axis.  This
        maximal set limits the possible rotational symmetry operations,
        since atoms lying on a test axis is irrelevant in testing rotational
        symmetryOperations.
        """

        def not_on_axis(ixyz):
            v = np.cross(ixyz, axis)
            return np.linalg.norm(v) > self.tol

        valid_sets = []
        origin_site, dist_el_sites = cluster_sites(self.cmol, self.tol)
        for test_set in dist_el_sites.values():
            valid_set = list(filter(not_on_axis, test_set))
            if len(valid_set) > 0:
                valid_sets.append(valid_set)

        return min(valid_sets, key=lambda s: len(s))

    def _check_rot_sym(self, axis):
        """
        Determines the rotational symmetry about supplied axis.  Used only for
        symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        max_sym = len(min_set)
        for i in range(max_sym, 0, -1):
            if max_sym % i != 0:
                continue
            op = SymmOp.from_axis_angle_and_translation(axis, 360 / i)
            rotvalid = self.is_valid_op(op)
            if rotvalid:
                self.symmops.append(op)
                self.rot_sym.append((axis, i))
                return i
        return 1

    def _check_perpendicular_r2_axis(self, axis):
        """
        Checks for R2 axes perpendicular to unique axis.  For handling
        symmetric top molecules.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)

        for i, j in itertools.combinations(range(self.cmol.natoms), 2):
            test_axis = np.cross(self.cmol.xyz[i] - self.cmol.xyz[j], axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                r2present = self.is_valid_op(op)
                if r2present:
                    self.symmops.append(op)
                    self.rot_sym.append((test_axis, 2))
                    return True

    def _proc_sph_top(self):
        """
        Handles Sperhical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self._find_spherical_axes()
        if len(self.rot_sym) == 0:
            logger.debug("Accidental speherical top!")
            self._proc_sym_top()
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        if rot < 3:
            logger.debug("Accidental speherical top!")
            self._proc_sym_top()
        elif rot == 3:
            mirror_type = self._find_mirror(main_axis)
            if mirror_type != "":
                if self.is_valid_op(ptg.inversion_op):
                    self.symmops.append(ptg.inversion_op)
                    self.sch_symbol = "Th"
                else:
                    self.sch_symbol = "Td"
            else:
                self.sch_symbol = "T"
        elif rot == 4:
            if self.is_valid_op(ptg.inversion_op):
                self.symmops.append(ptg.inversion_op)
                self.sch_symbol = "Oh"
            else:
                self.sch_symbol = "O"
        elif rot == 5:
            if self.is_valid_op(ptg.inversion_op):
                self.symmops.append(ptg.inversion_op)
                self.sch_symbol = "Ih"
            else:
                self.sch_symbol = "I"

    def _find_spherical_axes(self):
        """
        Looks for R5, R4, R3 and R2 axes in spherical top molecules.  Point
        group T molecules have only one unique 3-fold and one unique 2-fold
        axis. O molecules have one unique 4, 3 and 2-fold axes. I molecules
        have a unique 5-fold axis.
        """
        rot_present = defaultdict(bool)
        origin_site, dist_el_sites = cluster_sites(self.cmol, self.tol)
        xyz = min(dist_el_sites.values(), key=lambda s: len(s))
        for c1, c2, c3 in itertools.combinations(xyz, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = SymmOp.from_axis_angle_and_translation(test_axis,
                                                                    180)
                        rot_present[2] = self.is_valid_op(op)
                        if rot_present[2]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, 2))

            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for r in (3, 4, 5):
                    if not rot_present[r]:
                        op = SymmOp.from_axis_angle_and_translation(
                            test_axis, 360 / r)
                        rot_present[r] = self.is_valid_op(op)
                        if rot_present[r]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, r))
                            break
            if rot_present[2] and rot_present[3] and (
                        rot_present[4] or rot_present[5]):
                break

    def get_pointgroup(self):
        """
        Returns a PointGroup object for the molecule.
        """
        return pto(self.sch_symbol, self.symmops,
                                    self.mat_tol)

    def setup_pointgroup(self):
        """
        Setups as attribute the PointGroup object for the molecule.
        """
        self.pointgroup = self.get_pointgroup()

    def get_symmetry_operations(self):
        """
        Return symmetry operations as a list of SymmOp objects.
        Returns Cartesian coord symmops.

        Returns:
            ([SymmOp]): List of symmetry operations.
        """
        return generate_full_symmops(self.symmops, self.tol)

    def is_valid_op(self, symmop):
        """
        Check if a particular symmetry operation is a valid symmetry operation
        for a molecule, i.e., the operation maps all atoms to another
        equivalent atom.

        Args:
            symmop (SymmOp): Symmetry operation to test.

        Returns:
            (bool): Whether SymmOp is valid for Molecule.
        """
        xyz = self.cmol.xyz
        for i in range(self.cmol.natoms):
            ixyz = symmop.operate(xyz[i])
            ind = find_in_coord_list(xyz, ixyz, self.tol)
            if not (
                    len(ind) == 1 and
                    self.cmol.elems[ind[0]] == self.cmol.elems[i]
                ):
                return False
        return True

    def _get_eq_sets(self):
        """
        Calculates the dictionary for mapping equivalent atoms onto each other.

        Args:
            None

        Returns:
            dict: The returned dictionary has two possible keys:

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        UNIT = np.eye(3)
        eq_sets, operations = defaultdict(set), defaultdict(dict)
        symm_ops = [op.rotation_matrix
                    for op in generate_full_symmops(self.symmops, self.tol)]

        def get_clustered_indices():
            indices = cluster_sites(self.cmol, self.tol,
                                    give_only_index=True)
            out = list(indices[1].values())
            if indices[0] is not None:
                out.append([indices[0]])
            return out

        for index in get_clustered_indices():
            sites = self.cmol.xyz[index]
            for i, reference in zip(index, sites):
                for op in symm_ops:
                    rotated = np.dot(op, sites.T).T
                    matched_indices = find_in_coord_list(rotated, reference,
                                                         self.tol)
                    matched_indices = {
                        dict(enumerate(index))[i] for i in matched_indices}
                    eq_sets[i] |= matched_indices

                    if i not in operations:
                        operations[i] = {j: op.T if j != i else UNIT
                                         for j in matched_indices}
                    else:
                        for j in matched_indices:
                            if j not in operations[i]:
                                operations[i][j] = op.T if j != i else UNIT
                    for j in matched_indices:
                        if j not in operations:
                            operations[j] = {i: op if j != i else UNIT}
                        elif i not in operations[j]:
                            operations[j][i] = op if j != i else UNIT

        return {'eq_sets': eq_sets,
                'sym_ops': operations}

    @staticmethod
    def _combine_eq_sets(eq_sets, operations):
        """Combines the dicts of _get_equivalent_atom_dicts into one

        Args:
            eq_sets (dict)
            operations (dict)

        Returns:
            dict: The returned dictionary has two possible keys:

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        UNIT = np.eye(3)

        def all_equivalent_atoms_of_i(i, eq_sets, ops):
            """WORKS INPLACE on operations
            """
            visited = set([i])
            tmp_eq_sets = {j: (eq_sets[j] - visited) for j in eq_sets[i]}

            while tmp_eq_sets:
                new_tmp_eq_sets = {}
                for j in tmp_eq_sets:
                    if j in visited:
                        continue
                    visited.add(j)
                    for k in tmp_eq_sets[j]:
                        new_tmp_eq_sets[k] = eq_sets[k] - visited
                        if i not in ops[k]:
                            ops[k][i] = (np.dot(ops[j][i], ops[k][j])
                                         if k != i else UNIT)
                        ops[i][k] = ops[k][i].T
                tmp_eq_sets = new_tmp_eq_sets
            return visited, ops

        eq_sets = copy.deepcopy(eq_sets)
        new_eq_sets = {}
        ops = copy.deepcopy(operations)
        to_be_deleted = set()
        for i in eq_sets:
            if i in to_be_deleted:
                continue
            visited, ops = all_equivalent_atoms_of_i(i, eq_sets, ops)
            to_be_deleted |= visited - {i}

        for k in to_be_deleted:
            eq_sets.pop(k, None)
        return {'eq_sets': eq_sets,
                'sym_ops': ops}

    def get_equivalent_atoms(self):
        """Returns sets of equivalent atoms with symmetry operations

        Args:
            None

        Returns:
            dict: The returned dictionary has two possible keys:

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        eq = self._get_eq_sets()
        return self._combine_eq_sets(eq['eq_sets'],
                                     eq['sym_ops'])

    def symmetrize_molecule(self):
        """Returns a symmetrized molecule

        The equivalent atoms obtained via
        :meth:`~pymatgen.symmetry.analyzer.ptg.get_equivalent_atoms`
        are rotated, mirrored... unto one position.
        Then the average position is calculated.
        The average position is rotated, mirrored... back with the inverse
        of the previous symmetry operations, which gives the
        symmetrized molecule

        Args:
            None

        Returns:
            dict: The returned dictionary has three possible keys:

            ``sym_mol``:
            A symmetrized molecule instance.

            ``eq_sets``:
            A dictionary of indices mapping to sets of indices,
            each key maps to indices of all equivalent atoms.
            The keys are guaranteed to be not equivalent.

            ``sym_ops``:
            Twofold nested dictionary.
            ``operations[i][j]`` gives the symmetry operation
            that maps atom ``i`` unto ``j``.
        """
        eq = self.get_equivalent_atoms()
        eq_sets, ops = eq['eq_sets'], eq['sym_ops']
        xyz = self.cmol.xyz.copy()
        for i, eq_indices in eq_sets.items():
            for j in eq_indices:
                xyz[j] = np.dot(ops[j][i], xyz[j])
            xyz[i] = np.mean(xyz[list(eq_indices)], axis=0)
            for j in eq_indices:
                if j == i:
                    continue
                xyz[j] = np.dot(ops[i][j], xyz[i])
        m = copy.deepcopy(self.cmol)
        m.xyz = xyz
        return {'sym_mol': m,
                'eq_sets': eq_sets,
                'sym_ops': ops}


def iterative_symmetrize(mol, max_n=10, tolerance=0.3, epsilon=1e-2):
    """Returns a symmetrized molecule

    The equivalent atoms obtained via
    :meth:`~pymatgen.symmetry.analyzer.ptg.get_equivalent_atoms`
    are rotated, mirrored... unto one position.
    Then the average position is calculated.
    The average position is rotated, mirrored... back with the inverse
    of the previous symmetry operations, which gives the
    symmetrized molecule

    Args:
        mol (Molecule): A pymatgen Molecule instance.
        max_n (int): Maximum number of iterations.
        tolerance (float): Tolerance for detecting symmetry.
            Gets passed as Argument into
            :class:`~pymatgen.analyzer.symmetry.ptg`.
        epsilon (float): If the elementwise absolute difference of two
            subsequently symmetrized structures is smaller epsilon,
            the iteration stops before ``max_n`` is reached.


    Returns:
        dict: The returned dictionary has three possible keys:

        ``sym_mol``:
        A symmetrized molecule instance.

        ``eq_sets``:
        A dictionary of indices mapping to sets of indices,
        each key maps to indices of all equivalent atoms.
        The keys are guaranteed to be not equivalent.

        ``sym_ops``:
        Twofold nested dictionary.
        ``operations[i][j]`` gives the symmetry operation
        that maps atom ``i`` unto ``j``.
    """
    new = mol
    n = 0
    finished = False
    while not finished and n <= max_n:
        previous = new
        PTG = ptg(previous, tolerance=tolerance)
        eq = PTG.symmetrize_molecule()
        new = eq['sym_mol']
        finished = np.allclose(new.xyz, previous.xyz,
                               atol=epsilon)
        n += 1
    return eq


def cluster_sites(mol, tol, give_only_index=False):
    """
    Cluster sites based on distance and species type.

    Args:
        mol (Molecule): Molecule **with origin at center of mass**.
        tol (float): Tolerance to use.

    Returns:
        (origin_site, clustered_sites): origin_site is a site at the center
        of mass (None if there are no origin atoms). clustered_sites is a
        dict of {(avg_dist, species_and_occu): [list of sites]}
    """
    # Cluster works for dim > 2 data. We just add a dummy 0 for second
    # coordinate.
    dists = [[np.linalg.norm(mol.xyz[i]), 0] for i in range(mol.natoms)]
    import scipy.cluster as spcluster
    f = spcluster.hierarchy.fclusterdata(dists, tol, criterion='distance')
    clustered_dists = defaultdict(list)
    for i in range(mol.natoms):
        clustered_dists[f[i]].append(dists[i])
    avg_dist = {label: np.mean(val) for label, val in clustered_dists.items()}
    clustered_sites = defaultdict(list)
    origin_site = None
    for i in range(mol.natoms):
        if avg_dist[f[i]] < tol:
            if give_only_index:
                origin_site = i
            else:
                origin_site = mol.xyz[i]
        else:
            if give_only_index:
                clustered_sites[
                    (avg_dist[f[i]], mol.amass[i])].append(i)
            else:
                clustered_sites[
                    (avg_dist[f[i]], mol.amass[i])].append(mol.xyz[i])
    return origin_site, clustered_sites


def generate_full_symmops(symmops, tol):
    """
    Recursive algorithm to permute through all possible combinations of the
    initially supplied symmetry operations to arrive at a complete set of
    operations mapping a single atom to all other equivalent atoms in the
    point group.  This assumes that the initial number already uniquely
    identifies all operations.

    Args:
        symmops ([SymmOp]): Initial set of symmetry operations.

    Returns:
        Full set of symmetry operations.
    """
    # Uses an algorithm described in:
    # Gregory Butler. Fundamental Algorithms for Permutation Groups.
    # Lecture Notes in Computer Science (Book 559). Springer, 1991. page 15
    UNIT = np.eye(4)
    generators = [op.affine_matrix for op in symmops
                  if not np.allclose(op.affine_matrix, UNIT)]
    if not generators:
        # C1 symmetry breaks assumptions in the algorithm afterwards
        return symmops
    else:
        full = list(generators)

        for g in full:
            for s in generators:
                op = np.dot(g, s)
                d = np.abs(full - op) < tol
                if not np.any(np.all(np.all(d, axis=2), axis=1)):
                    full.append(op)

        d = np.abs(full - UNIT) < tol
        if not np.any(np.all(np.all(d, axis=2), axis=1)):
            full.append(UNIT)
        return [SymmOp(op) for op in full]

class pto(list): # formerly known as pto
    """
    Defines a point group, which is essentially a sequence of symmetry
    operations.

    Args:
        sch_symbol (str): Schoenflies symbol of the point group.
        operations ([SymmOp]): Initial set of symmetry operations. It is
            sufficient to provide only just enough operations to generate
            the full set of symmetries.
        tol (float): Tolerance to generate the full set of symmetry
            operations.

    .. attribute:: sch_symbol

        Schoenflies symbol of the point group.
    """
    def __init__(self, sch_symbol, operations, tol=0.1):
        self.sch_symbol = sch_symbol
        super(pto, self).__init__(
            generate_full_symmops(operations, tol))

    def __str__(self):
        return "%s(len=%d)" % (self.sch_symbol, len(self))

    def __repr__(self):
        return self.__str__()
