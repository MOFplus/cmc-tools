#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xmlrpc.client
import ssl
# try to use xmlrpxlibex from https://github.com/benhengx/xmlrpclibex (install with pip3 install xmlrpclibex)
try:
    from xmlrpclibex import xmlrpclibex
    can_use_socks = True     
except ImportError:
    can_use_socks = False
import logging
import sys
import os
import re
import molsys
import getpass
from .decorator import faulthandler, download

logger = logging.getLogger("mofplus")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
shandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

default_excepthook = sys.excepthook
def custom_excepthook(etype, value, tb):
    """
    Prevents username and password printed in stderr

    Arg:
        etype: exception class
        value: exception instance
        tb:    traceback object
    """
    if etype is xmlrpc.client.ProtocolError:
        pattern = 'ProtocolError for .*:.*@www.mofplus.org'
        replace = 'ProtocolError for <USERNAME>:<PW>@www.mofplus.org'
        value.url = re.sub(pattern,replace,value.url)
    default_excepthook(etype, value, tb)
sys.excepthook = custom_excepthook


class user_api(object):
    """Basic API class to talk to MOFplus

    Via the user_api class the API routines of MOFplus which are accessible for normal users and do not affect
    FF parameters can be used.

    Args:
        banner       (bool, optional): If True, the MFP API banner is printed to SDTOUT, defaults to False
        api          (string, optional): API to connect to, defaults to "user", can be "admin"
    """

    def __init__(self, banner = False, api="user"):
        assert api in ["user", "admin"]
        if banner: self._print_banner()
        try:
            logger.info("Get credentials from .mofplusrc")
            self.username, self.pw = self._credentials_from_rc()
        except IOError:
            try:
                logger.warning(".mofplusrc not found!")
                logger.info("Get credentials from environment variables")
                self.username = os.environ['MFPUSER']
                self.pw       = os.environ['MFPPW']
            except KeyError:
                logger.warning("Environment credentials not found!")
                logger.info("Get credentials from prompt")
                self.username, self.pw = self._credentials_from_cmd()
        ### read from environment variables to which DB should be connected, default is the global www.mofplus.org
        if api == "admin":
            logger.info("CONNECTING TO ADMIN API")
        ### if MFPDB is set then we connect to localhost
        if 'MFPDB' in os.environ:
            self.location = "LOCAL" 
            MFPDBname = os.environ['MFPDB']
        else:
            self.location = 'GLOBAL'
        # now open the connection
        if self.location == 'LOCAL':
            # if we are using a local version and can_use_socks is true then check if MFP_PRXY settings are present
            if can_use_socks and 'MFP_PRXY_IP' in os.environ:
                proxy = {
                    'host'       : os.environ["MFP_PRXY_IP"],
                    'port'       : '8080',
                    # 'username'   : os.environ["MFP_PRXY_USR"],
                    # 'password'   : os.environ["MFP_PRXY_PWD"],
                    'is_socks'   : True,
                    'socks_type' : 'v5',
                }
                logger.info('Trying to connect to local MOFplus API at localhost/%s via proxy at %s' % (MFPDBname, proxy['host']))
                self.mfp = xmlrpclibex.ServerProxy(
                    'http://%s:%s@localhost/%s/API/%s/xmlrpc' % (self.username, self.pw, MFPDBname, api),
                    timeout = 30,
                    proxy = proxy
                )
            else:
                logger.info('Trying to connect to local MOFplus API at localhost/%s' % MFPDBname)
                self.mfp = xmlrpc.client.ServerProxy('http://%s:%s@localhost/%s/API/%s/xmlrpc' % (self.username, self.pw, MFPDBname, api))
        else:
            logger.info('Trying to connect to global MOFplus API')
            self.mfp = xmlrpc.client.ServerProxy('https://%s:%s@www.mofplus.org/API/%s/xmlrpc' % (self.username, self.pw, api), 
                    allow_none = True, context = ssl._create_unverified_context())
        self._check_connection(api)
        return

    def _credentials_from_rc(self):
        """
        Method to get the credentials from ~/.mofplusrc

        Returns:
            username (str): username of current user
            pw (str): pw of current user
    
        """
        mprc_filename = os.environ["HOME"]+'/.mofplusrc'
        with open(mprc_filename, 'r') as mprc:
            username = mprc.readline().split()[0]
            pw       = mprc.readline().split()[0]
        return username, pw

    def _credentials_from_cmd(self):
        """
        Method to get the credentials from the command line
        
        Returns:
            username (str): username of current user
            pw (str): pw of current user
        """
        username = input("Email:")
        pw       = getpass.getpass()
        return username, pw

    def _check_connection(self, api):
        """
        Method to check if the connection to MFP is alive

        Raises:
            IOError: If connections is not possible
        """
        try:
            self.mfp.add(2,2)
            logger.info("Connection to %s API established" % api)
            if api == "admin":
                print("""
            We trust you have received the usual lecture from the MOF+ system administrator.
            It usually boils down to these two things:
                #1) Think before you type.
                #2) With great power comes great responsibility.
            """)                
        except xmlrpc.client.ProtocolError:
            logger.error("Not possible to connect to MOF+ %s API. Check your credentials" % api)
            raise IOError
        return

    def _print_banner(self):
        """
        Prints the MFP banner
        """
        print(":##::::'##::'#######::'########:::::::::::::::'###::::'########::'####:\n\
:###::'###:'##.... ##: ##.....::::'##::::::::'## ##::: ##.... ##:. ##::\n\
:####'####: ##:::: ##: ##::::::::: ##:::::::'##:. ##:: ##:::: ##:: ##::\n\
:## ### ##: ##:::: ##: ######:::'######::::'##:::. ##: ########::: ##::\n\
:##. #: ##: ##:::: ##: ##...::::.. ##.::::: #########: ##.....:::: ##::\n\
:##:.:: ##: ##:::: ##: ##::::::::: ##:::::: ##.... ##: ##::::::::: ##::\n\
:##:::: ##:. #######:: ##:::::::::..::::::: ##:::: ##: ##::::::::'####:\n\
:..:::::..:::.......:::..:::::::::::::::::::..:::::..::..:::::::::....:")

   
    @download('topology')
    def get_net(self,netname, out = 'file'):
        """
        Downloads a topology in mfpx file format
        
        Parameters:
            netname (str): name of the net
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'hdd'
        """
        lines = self.mfp.get_net(netname)
        return lines

    def get_list_of_nets(self):
        """
        Returns a list of all topologies stored at MOFplus.
        """
        return self.mfp.get_list_of_nets()
    

    def get_list_of_bbs(self):
        """
        Returns a list of all building blocks stored at MOFplus.
        """
        return self.mfp.get_list_of_bbs()

    @download('building block')
    def get_bb(self,bbname, out = 'file'):
        """
        Downloads a building block in mfpx file format
        
        Parameters:
            bbname (str): name of the bb
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'hdd'
        """
        lines = self.mfp.get_bb(bbname)
        return lines
    
    @download('MOF')
    def get_mof_structure_by_id(self,strucid, out='file'):
        """
        Downloads a MOF structure in mfpx file format
        
        Parameters:
            strucid (str): id of the MOF structure in the DB
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'hdd'
        """
        lines,name = self.mfp.get_mof_structure_by_id(strucid)
        return lines

    def get_cs(self,name):
        """
        Returns the coordinations sequences of a topology as a list of lists.
        
        Parameters:
            name (str): Name of the topology
            
        """
        return self.mfp.get_cs(name)
    
    def get_vs(self,name):
        """
        Returns the vertex symbol of a topology as a list of strings.
        
        Parameters:
            name (str): Name of the topology
        """
        return self.mfp.get_vs(name)

    def search_cs(self, cs, vs, cfilter = True):
        """
        Searches nets with a given coordination sequences and given vertex symbols and returns
        the corresponding netnames as a list of strings.
        
        Parameters:
            cs (list): List of the coordination sequences
            vs (list): List of the vertex symbols
            cfilter (bool): If True no catenated nets are returned, defaults to True
        """
        assert type(cs) == list
        assert type(vs) == list
        nets = self.mfp.search_cs(cs, vs)
        rl = []
        if cfilter:
            for i,n in enumerate(nets):
                if n.find('-c') != -1: rl.append(n)
            for i in rl: nets.remove(i)
        return nets

    @download('topology')
    def get_scaledtopo(self,id, out = 'file'):
        """
        Gets the scaled topo file for a given supercell id.

        Parameters:
            id(int): if of the supercell entry in the db for which
                the scaledtopo is requested
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'hdd'
        """
        lines = self.mfp.get_scaledtopo(id)
        return lines
    
    @download('orients')
    def get_orients(self,id):
        """
        Gets the orients file for a given  supercell id.

        Parameters:
            id(int): id of the supercell entry in the db for which
                the orients file is requested
        """
        lines = self.mfp.get_orients(id)
        return lines
