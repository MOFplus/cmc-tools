#! /usr/bin/env python
"""
Redundant Internal Coordinates
"""

# Import public classes to the top level of the module
from .ric   import RedIntCoords
from .grp   import Group
from .dist  import Distance
from .angl  import Angle
from .dihed import Dihedral

# Export the public classes
__all__ = ['RedIntCoords', 'Group', 'Distance', 'Angle', 'Dihedral']

# Internal verions
__version__ = '4.0.0'

