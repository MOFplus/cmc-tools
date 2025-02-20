#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ssl
import xmlrpc.client
# try to use xmlrpxlibex from https://github.com/benhengx/xmlrpclibex (install with pip3 install xmlrpclibex)
try:
    from xmlrpclibex import xmlrpclibex
    can_use_socks = True     
except ImportError:
    can_use_socks = False
import logging
import string
import molsys
from .decorator import faulthandler, download
from . import ff
logger = logging.getLogger("mofplus")

def smiles2can(smiles):
    """
    Method to transform a smiles into
    a canonical smiles string.
    
    Args:
        smiles (string): smiles string
    
    Returns:
        string: canonical smiles string
    """
    from openbabel import pybel
    om = pybel.readstring("smi", smiles)
    return om.write("can")[:-2]



class admin_api(ff.FF_api):

    """
    This class implements methods needed to administrate the MOFplus database. This class is inherited
    from the FF_api class.

    Args:
        banner       (bool, optional): If True, the MFP API banner is printed to SDTOUT, defaults to False

    Warning:
        The methods implemented in the admin_api class are only available for MOFplus administrators.
    """

    def __init__(self, banner = False):
        """init the admin API object
        
        We use the __init__ of the user API object here (inherited in the ff API).
        by providing api="admin" the admin api will be connected

        Args:
            banner (bool, optional): print banner. Defaults to False.
        """
        ff.FF_api.__init__(self, banner=banner, api="admin")
        return
    
    def add_net(self,data):
        """
        Insert  or update a net entry in the database currently connected
        Parameters:
            net:  dictionary of net data
        """
        retstring = self.mfp.insert_net(data)
        print(retstring)
        return retstring
    
    def delete_net(self, name):
        """
        Deletes a net from the db

        Parameters:
            name (str): name of the net
        """
        assert type(name) == str
        self.mfp.delete_net(name)
        
    def update_edge_database(self):
        retstring = self.mfp.update_edge_database()
        print(retstring)
        return
   
    def add_bb_penalties(self,data):
        """
        Method to add penalties to building blocks
        """
        retstring = self.mfp.add_bb_penalties(data)
        return

    def upload_weaver_run(self, fwid, scid, fname, energy):
        """
        Method to upload the results of a weaver run to the db

        Parameters:
            fwid: firework id of the job
            fname: filename of the structure file
        """
        data = {}
        data['fwid'] = str(fwid)
        data["scid"] = scid
        data["energy"] = energy
        f = open(fname, 'r')
        data['fmfpx'] = f.read()
        a = self.mfp.upload_weaver_run(data)
        return

    def upload_mof_structure_by_id(self, fname, strucid):
        """
        Method to upload a structure file to the DB

        Parameters:
            fname (str): path to the mfpx file
            strucid (int): id of the structure in the db
        """
        data = {}
        f = open(fname, 'r')
        data['id'] = strucid
        data['fmfpx'] = f.read()
        self.mfp.upload_mof_structure_by_id(data)
        return

    def set_structure(self,scid, name, path, ff, properties = {}):
        """
        Sets the structures for a given supercell db entry id.
        
        Parameters:
            scid(int): id of the supercell entry in the db
            name(str): name of the structure
            path(str): path to the mfpx file of the structure
            ff(str): FF/LOT on which the structure was computed
            properties(dict, optional): dictionary of properties
            of the structures, defaults to {}

        """
        with open(path , "r") as f:
            fstring = f.read()
        self.mfp.set_structure(scid, name, fstring, ff, properties)
        return

    def RTAfinish(self,jobid):
        """
        Method to mark an RTA job in the mofplus db as finished
        
        Parameters:
            jobid(int): id of the job in the db
        """
        self.mfp.RTAfinish(int(jobid))
        return

    def upload_topo_file_by_name(self, fname, name):
        """
        Method to upload a topo file to the DB
        
        Parameters:
            fname (str): path to the mfpx file
            name (str): name of the topology
        """
        data = {}
        f = open(fname, 'r')
        data['name'] = name
        data['fmfpx'] = f.read()
        self.mfp.upload_topo_file_by_name(data)
        return

    def upload_bbfile_by_name(self, fname, name):
        """
        Method to upload a bb file to the DB
        
        Parameters:
            fname (str): path to the mfpx file
            name (str): name of the bb
        """
        data = {}
        f = open(fname, 'r')
        data['name'] = name
        data['fmfpx'] = f.read()
        self.mfp.upload_bbfile_by_name(data)
        return

    def insert_bb(self,name, fname, chemtype, frag = False):
        """
        Method to create a new entry in the bb table.

        Parameters:
            name (str): name of the bb
            fname (str): path to the mfpx file
            chemtype (str): string describing the character of the bb
            frag (bool, optional): Option to set a BB as fragment, defaults to False
        """
        data = {}
        data['name'] = name
        data['fmfpx'] = open(fname, 'r').read()
        data['type'] = chemtype
        self.mfp.insert_bb(data)
        return

    def insert_bb_from_smiles(self, name, smiles):
        """
        Method to insert a bb in the database by specifying 
        its smiles string. It is then converted to a canonical
        smiles string.
        
        Args:
            name (string): name of the bb in the database
            smiles ([type]): smiles string of the structure
        """
        assert smiles.count("*") >= 2, "Smiles has to incorporate at least two connectors"
        assert smiles.count(".") == 0, "More than one molecule specified"
        # transform to canonical
        csmiles = smiles2can(smiles)
        # open molsys
        m = molsys.mol.from_smiles(csmiles)
        ret = self.mfp.insert_bb2(m.to_string(),"organic", name, csmiles)
        return 

    def set_cs(self, name, cs):
        """
        Method to set the cs of a topology.

        Parameters:
            name (str): name of the topology
            cs (list): list of lists with the cs
        """
        data = {}
        data['name'] = name
        data['cs'] = cs
        self.mfp.set_cs(data)
        return
    
    def set_vs(self, name, vs):
        """
        Method to set the vs of a topology.

        Parameters:
            name (str): name of the topology
            vs (list): list with the vs
        """
        data = {}
        data['name'] = name
        data['vs'] = vs
        self.mfp.set_vs(data)
        return
    
    def connect_nets(self, pnet, cnet, pattern):
        """
        Method to create relationchips between nets in the DB

        Parameters:
            pnet (str): name of the parent net
            cnet (str): name of the child net
            pattern (str): derivation type
        """
        assert type(pnet) == str
        assert type(cnet) == str
        assert type(pattern) == str
        assert cnet != pnet
        self.mfp.connect_nets(pnet,cnet,pattern)
        return

    def add_skal_property(self, strucid, ptype, prop):
        """
        Method to add a skalar property to a structure

        Parameters:
            strucid (int): id of the structure in the DB
            ptype (str): name of the property
            prop (float): property value
        """
        assert type(strucid) == int
        assert type(ptype) == str
        self.mfp.add_skal_property(strucid, ptype, prop)
        return

    def add_xy_property(self,strucid,ptype,data):
        """
        Method to add a dataset as property to the DB

        Parameters:
            strucid (int): id of the structure in the DB
            ptype (str): name of the property
            data (dict): dataset as dictionary 
        """
        assert type(strucid) == int
        assert type(ptype) == str
        self.mfp.add_xy_property(strucid, ptype,data)
        return

    def finish(self, wfname):
        """
        Method to register an arbitrary fireworks as finished.
        Implemented ones are fireweaver and firebbopt.
        
        Args:
            wfname (wfname): name of the workflow
        """
        implemented = ["fw", "fbb", "fa"]
        assert wfname.split("_")[0] in implemented
        self.mfp.finish(wfname)


    def fa_finish(self,faid):
        """
        Method to register a fireanalyzer run as finished

        Parameters:

            faid (int): id of fireanalyzer run
        """
        assert type(faid) == int
        self.mfp.fa_finish(faid)

    def fw_finish(self,fwname):
        """
        Method to register a fireweaver run as finished.
        
        Args:
            fwname (string): name of the workflow
        """
        self.mfp.fireweaver_finish(fwname)
        
    
    
    def set_FFref(self, name, hdf5path = None, mfpxpath = None, comment=""):
        """
        Method to create a new entry in the FFref table and to upload a file with
        reference information in the hdf5 file format.

        Parameters:
            name (str): name of the entry in the DB
            hdf5path (str): path to the hdf5 reference file, default is <name>.hdf5
            mfpxpath (str): path to the mfpx file, default is <name>.mfpx
            comment (str): some comment on the reference data, defaults to ''
        """
        assert type(name) == str
        if hdf5path is None: hdf5path=name+".hdf5"
        if mfpxpath is None: mfpxpath=name+".mfpx"
        with open(hdf5path, "rb") as handle:
            binary = xmlrpc.client.Binary(handle.read())
        with open(mfpxpath, "r") as handle:
            mfpx = handle.read()
        self.mfp.set_FFref(name, binary, mfpx, comment)
        return
    
    def set_FFref_graph(self,name, mfpxpath = None):
        """
        Method to upload the structure of a reference system in the mfpx file format.

        Parameters:
            name (str): name of the entry in the DB
            mfpxpath (str): path to the mfpx file
        """
        with open(mfpxpath, "r") as handle:
            mfpx = handle.read()
        self.mfp.set_FFref_graph(name,mfpx)
        return
    
    def set_FFfrag(self,name,mfpxpath=None,comment=""):
        """
        Method to create a new entry in the FFfrags table.

        Parameters:
            name (str): name of the entry in the db
            path (str): path to the mfpx file of the fragment
            comment (str): comment, defaults to ''
        """
        assert type(name) == type(comment) == str
        if mfpxpath is None: mfpxpath=name+".mfpx"
        with open(mfpxpath, "r") as f:
            lines = f.read()
            #m = molsys.mol.from_string(lines)
        m = molsys.mol.from_file(mfpxpath)
        prio = m.natoms-m.elems.count("x")
        self.mfp.set_FFfrag(name, lines, prio, comment)
        return
    
    def set_special_atype(self, at, ft, stype = "linear"):
        """
        Method to assign an attribute to an aftype

        Parameters:
            at (str): atype
            ft (str): fragtype
            stype (str,optional): attribute, defaults to linear
        """
        assert type(at) == type(ft) == type(stype) == str
        self.mfp.set_special_atype(at,ft,stype)
        return

    def set_atype(self, at):
        """
        Method to set atom type in the API
        
        Args:
            at (str): atomtype which should be inserted in the API
        """
        assert type(at) == str
        self.mfp.set_atype(at)
        return

    def set_orients(self, scid, path):
        """
        Method to upload an orients file

        Parameters:
            scid: id of the supercell entry
            path: path to the orientsfile
        """
        with open(path, "r") as f:
            lines = f.read()
        self.mfp.set_orients(scid, lines)
        return

    def set_scaledtopo(self, scid, path):
        """
        Method to upload a scaled topo file

        Parameters:
            scid: id of the supercell entry
            path: path ot the scaled topo file
        """
        with open(path, "r") as f:
            lines = f.read()
        self.mfp.set_scaledtopo(scid,lines)
        return

