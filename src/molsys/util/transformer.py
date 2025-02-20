#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import logging

logger = logging.getLogger("molsys.transformer")

def to_pymatgen(mol):
    """
    Creates a pymatgen structure object from a mol object
    :Parameters:
        - mol: mol object
    """
    try:
        import pymatgen as mg
    except ImportError:
        logger.error("Pymatgen not available!")
        exit()
    latt = mg.core.Lattice.from_parameters(*mol.get_cellparams())
    elems = mol.get_elems()
    nelems = []
    for e in elems: nelems.append(e.title())
    structure = mg.core.Structure(latt, nelems, mol.get_frac_xyz())
    return structure

