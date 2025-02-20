#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
               napi implements XMLRPC-free (!) new api to access FF params on MOFplus    

(C) RS RUB 2024 ... this is currently experimental code

"""

import requests
from requests.auth import HTTPBasicAuth as httpauth
from urllib.parse import urlunparse, urlencode
import os
import json

import logging
from molsys.util.aftypes import aftype
import molsys

logger = logging.getLogger("mofplus")
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

bodymapping = {1:"onebody", 2:"twobody",3:"threebody",4:"fourbody"}

def unlist(l, field):
    return {e[field] : e for e in l} 

class api:
    """
    New API based on http requests and json parsing using the REST API of web2py (avoiding xmlrpc)

    """
    def __init__(self, loglevel = logging.WARNING,
                       scheme = "https",
                       netloc = "www.mofplus.org",
                       path = "/ff/",
                       timeout = 5,
                       verify = True, # this is True by default but SSL cert fails can be bypassed with False
                       ):
        logger.setLevel(loglevel)
        self.scheme = scheme
        self.netloc = netloc
        self.timeout = timeout
        self.verify = verify
        self.path = path # this could be changed to a user login api 
        return

    def connect(self, username, pw):
        self.username = username
        self.pw = pw
        self.auth = httpauth(self.username, self.pw)
        self.timeout = 5
        return self.check_connection()

    def credentials_from_rc(self):
        """
        Method to get the credentials from ~/.mofplusrc

        TODO: could add more things like scheme, netloc here 
              could be a yaml file

        Returns:
            username (str): username of current user
            pw (str): pw of current user
    
        """
        mprc_filename = os.environ["HOME"]+'/.mofplusrc'
        with open(mprc_filename, 'r') as mprc:
            username = mprc.readline().split()[0]
            pw       = mprc.readline().split()[0]
        return username, pw

    def get_response(self, func, plug="gen", all=False, **params):
        urlp = self.scheme + "://" + self.netloc + self.path + plug
        if type(func) == type([]):
            reskey = func[0]
            urlp += "/" + "/".join(func) # need leading slash
        elif type(func) == type(""):
            reskey = func
            urlp += "/"+func
        else:
            logger.error("Wrong type for func")
            return None
        logger.info("URL: %s", urlp)
        response = requests.get(urlp, auth=self.auth, params=params, timeout=self.timeout, verify=self.verify)
        if response.status_code == 200:
            if response.json()['status'] == 'OK':
                if all:
                    return response.json()
                else:
                    return response.json()[reskey]
            else:
                logger.error("Error in response: %s", response.json())
                return None
        else:
            logger.error("HTTP request to MOFplus failed with status code %s", response.status_code)
            return None 
                
    def get_response_file(self, fname):
        urlp = self.scheme + "://" + self.netloc + "/ff/get_file/" + fname
        logger.info("URL: %s", urlp)
        response = requests.get(urlp, auth=self.auth, timeout=self.timeout, verify=self.verify)
        if response.status_code == 200:
            return response.text
        else:
            logger.error("HTTP request to MOFplus failed with status code %s", response.status_code)
            return None        

    def send_data(self, func, params, plug="gen"):
        urlp = self.scheme + "://" + self.netloc + self.path + plug
        if type(func) == type([]):
            urlp += "/" + "/".join(func) # need leading slash
            reskey = func[0]
        elif type(func) == type(""):
            urlp += "/"+func
            reskey = func
        else:
            logger.error("Wrong type for func")
            return None
        logger.info("URL: %s", urlp)
        response = requests.post(urlp, auth=self.auth, params=params, timeout=self.timeout, verify=self.verify)
        if response.status_code == 200:
            if response.json()['status'] == 'OK':
                if reskey in response.json():
                    return response.json()[reskey]
                return response.json()
            else:
                logger.error("Error in response: %s", response.json())
                return None
        else:
            logger.error("HTTP request to MOFplus failed with status code %s", response.status_code)
            return None 

    def check_connection(self):
        """
        Method to check if the connection to MOFPLUS is working
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        return self.get_response("check")
            
    def get_special_atypes(self):
        """
        Method to get the special atom types

        from MOFplus we get a list of tuples with (atype,fragtype,sptype) which we convert to a dict
        
        Returns:
            dict: dictionary with a list of special atom types as aftypes
        """
        raw = self.get_response("special_atypes")
        # convert atype,frag tuples to atype objects
        spatypes = {}
        for e in raw:
            if e[2] in spatypes.keys():
                spatypes[e[2]].append(aftype(e[0],e[1]))
            else:
                spatypes[e[2]] = [aftype(e[0],e[1])]
        return spatypes
    
    def get_FF(self):
        raw = self.get_response("FF")
        return unlist(raw, "name")
    
    def get_FFrefs(self, **query):
        raw = self.get_response("FFrefs", **query)
        refs = {e["name"]: e for e in raw}
        return refs
    
    def get_FFfits(self, **query):
        fits = self.get_response("FFfits", **query)
        return fits
    
    def set_FFfit(self, ffid, f):
        if "id" in f:
            del f["id"]
        f["FFID"] = ffid
        # some fields need to be json format ... it seesm we do not get the right format back from the server
        # TODO: this could be checked server side where we know the field types .. probably web2py has a method for that
        for field in ["settings", "active", "upgrades", "atfix"]:
            if field in list(f.keys()):
                if f[field] != None:
                    f[field] = json.dumps(f[field])
                else:
                    del f[field]
        ret = self.send_data("FFfits", f)
        if ret is not None:
            if len(ret["errors"]) > 0:
                logger.error("Errors in setting fit: %s", ret["errors"])
                return None
            return ret["id"]
        return None
    
    def get_FFfrags_all(self, **query):
        """get all fragments from DB

        TODO: this is using the old plug "gen" and should be adapted ... for assignement we use
            this function but for fragementize the new one

        Returns:
            dict: dictionary mapping id on entry
        """        
        raw = self.get_response("FFfrags", **query)
        frags = {e["id"]: e for e in raw}
        return frags
    
    def get_FFfrags(self, vtypes):
        # this is a hack .. there should be a better option
        # reason: on server side we use belongs() which fails if the length of vtypes is 1
        #         so we add a dummy entry that will never occur .. fixes the problem
        #        this is a hack and should be fixed on server side
        if len(vtypes) == 1:
            vtypes.append("xxx")
        data = self.get_response("frags", plug="frags", atypes=vtypes)
        return data

    def get_atypes(self, **query):
        """get all atypes from the DB

        Returns:
            dict: dict mapping id on name
        """        
        raw = self.get_response("atypes", **query)
        atypes = {e["id"]: e["name"] for e in raw}
        return atypes
        
    def get_refs_from_frags(self, frags):
        return self.get_response("refs4frags", frags=",".join(frags))
    
    def get_frags_from_refs(self, refs):
        return self.get_response("refs4frags", refs=",".join(refs))
    
    def get_atypes_from_refs(self, refs):
        return self.get_response("refs4atypes", refs=",".join(refs))
    
    def get_mol_from_db(self, fname, as_string=False):
        s = self.get_response_file(fname)
        if as_string:
            return s
        m = molsys.mol.from_string(s)
        return m

    def get_params(self, ric, fitid):
        raw = self.get_response(["params", ric, str(fitid)])
        return raw
    
    def set_params(self, ric, fitid, params):
        # enforce fitID (especially if it is missing)
        params["fitID"] = fitid
        # delte the id if it exists
        if "id" in params:
            del params["id"]
        # convert the params to json
        assert "params" in params, "No params!!"
        params["params"] = json.dumps(params["params"])
        assert ric in ("bnd", "ang", "dih", "oop", "cha", "vdw"), "Unknown ric"
        ret = self.send_data(["set_params", ric], params)
        if ret == None:
            logger.error("Error in set_params")
            return None
        return ret["id"]
    
    def update_params(self, ric, fitid, pid, params):
        # pid is the ID of the params which shoudl exist .. from get_params
        # instead of deleting "id" as in set_params we add/overwrite it explicit
        # ideally your params dict only contains the entries that need to be updated
        params["fitID"] = fitid
        params["id"] = pid
        if "params" in params:
            params["params"] = json.dumps(params["params"])
        assert ric in ("bnd", "ang", "dih", "oop", "cha", "vdw"), "Unknown ric"
        ret = self.send_data(["set_params", ric], params)
        if ret == None:
            logger.error("Error in update_params")
            return None
        return ret
    
    def delete_params(self, ric, ids):
        ret = self.send_data(["del_params", ric], {"ids": ids})
        return ret


        
    
