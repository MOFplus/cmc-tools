#   -*- coding: utf-8 -*-
###########################################
#
#   auxiliary module to forge hdf data v1.0
#   originally implemented for the
#       pydlpoly testing framework
#
#   by Roberto Amabile (2016)
#
#   PRE-ALPHA VERSION: USE AT YOUR OWN RISK
#       NO WARRANTY, NO LIABILITY
#   The author will use best judgement and
#       reasonable effort to keep the code
#       clear and tidy
#   NOT YET LICENSED
#
###########################################

import os
import h5py
import numpy as np

class DataReference:
    
    def __init__(self, fname, modeaccess = "a"):
        self.name = fname
        self.access = modeaccess
        self.h5file = h5py.File(fname, modeaccess)
    
    
    def build_rec_dataset(self, data, h5file = None, path = None, exists_warning=True):
        """
        Method to store a the data of a dictionary in a recursive manner in a
        hdf5 file, preserving the structure of the dictionary.
        
        :Parameters:
            - data (dict): dictionary conataining the data you want to write to
            the hdf5 file
            - h5file (obj): h5py.File instance
            - path (str): entrypoint of the hdf5 file
            - exists_warning: if file exists and mode access is not explicit w, raise IOError 
        """
        if exists_warning:
            if os.path.isfile(fname) and self.access[0] != "w":
                raise  IOError("File %s exists, explicit exists_warning=False for DataReference instance" % (self.name,))
        if h5file is None:
            h5file = self.h5file
        if path is not None:
            try:
                h5file = h5file[path]
            except KeyError:
                h5file = h5file.create_group(path)
        for key in data:
            value = data[key]
            if type(value) is dict:
                try:
                    group = h5file.create_group(key)
                except:
                    group = h5file[key]
                self.build_rec_dataset(value, group, exists_warning=exists_warning)
            else:
                value = np.array(value)
                if "U" in str(value.dtype):
                    dt = h5py.string_dtype(encoding='utf-8')
                    dset = h5file.create_dataset(key, value.shape, dtype=dt)
                    dset[...] = value
                else:
                    h5file.create_dataset(key, data=value)
    
            
    def load_rec_dataset(self, h5file = None, path = None):
        """
        Method to dump the data of an hdf5 file to a dictionary in a recursive
        manner.
        
        :Parameters:
            - h5file (obj): h5py.File instance
            - path (str): entrypoint of the hdf5 file
        
        :Returns:
            - res (dict): dictionary containing the requested data
        """
        res = {}
        if h5file is None:
            h5file = self.h5file
        if path is not None:
            h5file = h5file[path]
        for key, item in h5file.items():
            if type(item) is h5py._hl.dataset.Dataset:
                res[key] = item.value
            elif type(item) is h5py._hl.group.Group:
                res[key] = self.load_rec_dataset(h5file = h5file[key])
        return res
