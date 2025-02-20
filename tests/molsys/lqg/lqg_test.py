#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import sys
import molsys.util.lqg as lqg
import molsys.util.h5craft as h5craft
from numpy.testing import assert_array_equal, assert_allclose
import h5py

def generate_ref_info(netname, systrefile = 'rcsr_0.6.0.arc', rfile = 'lqg.hdf5'):
    ### load systre keys ###
    r = lqg.reader()
    r.load_keys_from_file(systrefile)
    ### create lqg object
    g = lqg.lqg()
    g.read_systre_key(r(netname)[1],3)
    g.build_lqg()
    ### create ref dictionary container
    dlqg = {}
    datadic = {}
    ### construct barycentric embedding
    dlqg['edges'] = g.edges
    dlqg['labels'] = g.labels
    dlqg['dim'] = g.dim
    dlqg['nvertices'] = g.nvertices
    ### get cocycles
    g.get_cocycle_basis()
    datadic['cocycle_basis'] = g.cocycle_basis 
    ### get cycle basis
    g.get_cyclic_basis()
    datadic['cycle_basis'] = g.cyclic_basis 
    ### build B matrix 
    g.get_B_matrix()
    ### get alpha
    g.get_alpha()
    datadic['alpha'] = g.alpha
    ### get lattice basis
    g.get_lattice_basis()
    datadic['lattice_basis'] = g.lattice_basis
    ### get kernel
    g.get_kernel()
    datadic['kernel'] = g.kernel
    ### get arcs
    g.get_fracs()
    datadic['arcs'] = g.fracs
    ### get cell
    g.get_cell()
    datadic['cell'] = g.cell
    ### dump data to hdf5 file
    rdic = {}
    rdic[netname] = {'lqg':dlqg, 'ref':datadic} 
    dr = h5craft.DataReference(rfile)
    dr.build_rec_dataset(rdic)
    return


class lqg_test(unittest.TestCase):

    def __init__(self, testname, netname):
        super(lqg_test,self).__init__(testname)
        self.netname = netname
        return

    def setUp(self):
        netname = self.netname
        self.h5file = h5py.File('lqg.hdf5', 'r+')
        self.g = lqg.lqg()
        self.g.get_lqg_from_lists(self.h5file[netname]['lqg']['edges'][()],
                self.h5file[netname]['lqg']['labels'][()],
                self.h5file[netname]['lqg']['nvertices'][()],
                self.h5file[netname]['lqg']['dim'][()],
                )
        self.g.build_lqg()
        return

    def test_cycle_basis(self):
        netname = self.netname
        cb = self.g.get_cyclic_basis()
        assert_allclose(cb, self.h5file[netname]['ref']['cycle_basis'][()],
                atol = 10**-12)

    def test_cocycle_basis(self):
        netname = self.netname
        cob = self.g.get_cocycle_basis()
        assert_allclose(cob, self.h5file[netname]['ref']['cocycle_basis'][()],
                atol = 10**-12)

    def test_alpha(self):
        netname = self.netname
        self.g.cyclic_basis =  self.h5file[netname]['ref']['cycle_basis'][()]
        self.g.cocycle_basis =  self.h5file[netname]['ref']['cocycle_basis'][()]
        alpha = self.g.get_alpha()
        assert_allclose(alpha, self.h5file[netname]['ref']['alpha'][()],
                atol = 10**-12)

    def test_lattice_basis(self):
        netname = self.netname
        self.g.cyclic_basis =  self.h5file[netname]['ref']['cycle_basis'][()]
        self.g.cocycle_basis =  self.h5file[netname]['ref']['cocycle_basis'][()]
        self.g.alpha =  self.h5file[netname]['ref']['alpha'][()]
        lb = self.g.get_lattice_basis()
        assert_allclose(lb, self.h5file[netname]['ref']['lattice_basis'][()],
                atol = 10**-12)

    def test_kernel(self):
        netname = self.netname
        self.g.cyclic_basis =  self.h5file[netname]['ref']['cycle_basis'][()]
        self.g.cocycle_basis =  self.h5file[netname]['ref']['cocycle_basis'][()]
        self.g.alpha =  self.h5file[netname]['ref']['alpha'][()]
        self.g.lattice_basis =  self.h5file[netname]['ref']['lattice_basis'][()]
        k = self.g.get_kernel()
        assert_allclose(k, self.h5file[netname]['ref']['kernel'][()],
                atol = 10**-12)

    def test_cell(self):
        netname = self.netname
        self.g.cyclic_basis =  self.h5file[netname]['ref']['cycle_basis'][()]
        self.g.cocycle_basis =  self.h5file[netname]['ref']['cocycle_basis'][()]
        self.g.alpha =  self.h5file[netname]['ref']['alpha'][()]
        self.g.lattice_basis =  self.h5file[netname]['ref']['lattice_basis'][()]
        self.g.kernel =  self.h5file[netname]['ref']['kernel'][()]
        cell = self.g.get_cell()
        assert_allclose(cell, self.h5file[netname]['ref']['cell'][()],
                atol = 10**-12)

    def test_arcs(self):
        netname = self.netname
        self.g.cyclic_basis =  self.h5file[netname]['ref']['cycle_basis'][()]
        self.g.cocycle_basis =  self.h5file[netname]['ref']['cocycle_basis'][()]
        self.g.alpha =  self.h5file[netname]['ref']['alpha'][()]
        self.g.get_B_matrix()
        arcs = self.g.get_fracs()
        assert_allclose(arcs, self.h5file[netname]['ref']['arcs'][()],
                atol = 10**-12)

def main(netname):
    suite = unittest.TestSuite()
    suite.addTest(lqg_test("test_cycle_basis",netname))
    suite.addTest(lqg_test("test_cocycle_basis",netname))
    suite.addTest(lqg_test("test_alpha",netname))
    suite.addTest(lqg_test("test_lattice_basis",netname))
    suite.addTest(lqg_test("test_kernel",netname))
    suite.addTest(lqg_test("test_cell",netname))
    suite.addTest(lqg_test("test_arcs",netname))
    unittest.TextTestRunner(verbosity = 3).run(suite)

if __name__ == '__main__':
    # only a few nets available for this test...
    nets = ["aab", "pto", "qtz", "tbo", "ths"]
    for net in nets:
        main(net)
