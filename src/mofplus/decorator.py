#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xmlrpc.client
import molsys
import logging
from functools import wraps
import time

logger = logging.getLogger("mofplus")

def download(dtype, binary = False):
    """
    mfp download decorator
    """
    def download_decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            time.sleep(0.1)
            try:
                lines = func(*args, **kwargs)
                if "out" in list(kwargs.keys()):
                    if kwargs["out"] == 'mol':
                        if dtype == "topology":
                            return molsys.topo.from_string(lines)
                        else:
                            return molsys.mol.from_string(lines)
                    elif kwargs["out"] == 'str':
                        return lines
                if binary == False:
                    if dtype == "orients":
                        f=open(str(args[1])+'.orients', 'w')
                    else:
                        f=open(str(args[1])+'.mfpx', 'w')
                    f.write(lines)
                    f.close()
                else:
                    with open("%s.hdf5" % str(args[1]), "wb") as handle:
                        handle.write(lines)
                logger.info('%s %s downloaded from mofplus' % (dtype,args[1]))
            except xmlrpc.client.Fault:
                logger.error('Requested %s %s not available on mofplus' % (dtype, args[1]))
        return inner
    return download_decorator

def batch_download(dtype):
    """
    mfp batch download decorator
    """
    def download_decorator(func):
        def inner(*args, **kwargs):
            ### excecute the decorated function
            try:
                dic_lines = {}
                list_lines = func(*args,**kwargs)
                ### put it to the dictionary
                for i,n in enumerate(args[1]):
                    dic_lines[n]=list_lines[i]
                ### now handle the output
                if "out" in list(kwargs.keys()):
                    if kwargs["out"] == "str":
                        return dic_lines
                ### case of files
                for n,lines in list(dic_lines.items()):
                    with open("%s.mfpx" % n, "w") as f:
                        f.write(lines)
            except xmlrpc.client.Fault:
                logger.error('Requested systems not available on mofplus')
        return inner
    return download_decorator


def faulthandler(func):
    def inner(*args,**kwargs):
        ret = func(*args, **kwargs)
        if type(ret) == dict:
            logger.error(ret['faultString'])
            return ret["faultString"]
        return ret
    return inner
