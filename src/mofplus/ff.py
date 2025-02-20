#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import logging
import sys
from mofplus.decorator import faulthandler, download, batch_download
from mofplus import user
from molsys.util.aftypes import aftype, aftype_sort #, afdict

logger = logging.getLogger("mofplus")

bodymapping = {1:"onebody", 2:"twobody",3:"threebody",4:"fourbody"}

allowed_ptypes = {1: ["charge", "vdw", "equil"],
        2: ["bnd", "chargemod", "vdwpr", "bnd5"],
        3: ["ang", "ang5"],
        4: ["dih", "oop"]
        }

allowed_potentials = {"charge": [["point",1], ["gaussian",2], ["slater",2]],
        "equil": [["equil", 1]],
        "vdw": [["LJ",2], ["buck",2], ["buck6d",2]],
        "bnd": [["harm",2], ["mm3",2], ["quartic",5], ["morse",3], ["equiv", 2]],
        "bnd5": [["harm",2], ["mm3",2], ["quartic",5], ["morse",3], ["equiv", 2]],
        "chargemod": [["point",1], ["gaussian",2], ["slater",2]],
        "vdwpr": [["LJ",2], ["buck",2], ["damped_buck",2]],
        "ang": [["harm",2],["mm3",2], ["quartic",5], ["fourier",5],  ["strbnd", 6]],
        "ang5": [["harm",2],["mm3",2], ["quartic",5], ["fourier",5],  ["strbnd", 6]],
        "dih": [["harm",2], ["cos3",3], ["cos4",4]],
        "oop": [["harm",2]]}



class FF_api(user.user_api):   
    """API class to query FF dependent data from MOFplus

    Via the FF_api class the API routines of MOFplus concerning the retrieval of MOF-FF parameters can be used.
    A detailed description of the way parameters are stored in the database and how our assignment algorithm is 
    working can be found at www.mofplus.org.
    
    The FF_api class inherits from the user_api class.

    Note:
        In several methods for receiving and uploading parameters the argument atypes has to be stated. This is 
        always a string of atypes, seperated by a ':'. An atype is in consists in this case of the original atype
        and the fragment it belongs, seperated by an '@'. Consider the following example string:
        'c3_c2h1@ph:c3_c2h1@ph'.

    Warning:
        Every registered MOFplus user has the possibility to download parameters from the database. The upload of
        parameters for a FF is only possible to those users which are maintainers of the FF. If you want to become
        a FF maintainer or wants to host your own FF at MOFplus please contact Johannes P. Dürholt.

    Args:
        banner       (bool, optional): If True, the MFP API banner is printed to SDTOUT, defaults to False
    """

    
    def _format_atypes(self, atypes, ptype, potential):
        """
        Helper function to extract fragments out of atypes and to
        order atypes and fragments in dependence of the ptype. 
        
        Parameters:
            atypes (list): list of atom types in the form "atype@fragtype"
            ptype (str): ric type
            potential (str): potential type
        """
        assert type(ptype) == str
        ### split into tuples of aftypes, then sort and then split into 
        ### frags and atypes
        aftypes = []
        for i in atypes.split(":"):
            aftypes.append(aftype(*i.split("@")))
        aftypes = aftype_sort(aftypes, ptype)
        atypes  = [i.atype for i in aftypes]
        fragments = [i.fragtype for i in aftypes]
        if ptype not in allowed_ptypes[len(atypes)]:
            raise ValueError("ptype %s not allowed for %s term" % (ptype, bodymapping[len(atypes)]))
        if potential not in [i[0] for i in allowed_potentials[ptype]]:
            raise ValueError("potential %s not allowed for ptype %s" % (potential, ptype))
        return atypes, fragments

    def get_params_from_ref(self, FF, ref):
        """
        Method to look up all FF parameters that are available for a reference system
        
        Parameters:
            FF (str): Name of the FF the parameters belong to
            ref (str): Name of the reference system the parameters belong to

        Returns:
            dictionary of parameters sorted in respect of the number of involved atoms.

        Example:
            >> api.get_params_from_ref('MOF-FF', 'benzene')
        """
        paramsets = self.mfp.get_params_from_ref(FF,ref)
        paramdict = {"onebody":{"charge":afdict(),"vdw":afdict(),"equil":afdict()},
                "twobody":{"bnd":afdict(),"bnd5":afdict(),"chargemod":afdict(), "vdwpr":afdict()},
                "threebody":{"ang":afdict(),"ang5":afdict()},
                "fourbody": {"dih":afdict(),"oop":afdict()}}
        # RS (explanation to be improved by JPD)
        # paramset is a nested list of lists provided by MOF+
        # it is resorted here in to a number of nested directories for an easier retrieval of data
        # i loops over the lists from paramset
        # each entry is
        #      i[0] : atomtype (len i[0] determines n-body via gloabl bodymapping)
        #      i[1] : fragment
        #      i[2] : type (e.g. charge, vdw, equiv)   TODO: change euilv -> equiv
        #      i[3] : ptype
        #      i[4] : paramstring
        for i in paramsets:
            #typestr =""
            #for a,f in zip(i[0],i[1]):
            #    typestr+="%s@%s:" % (a,f)
            ## cut off last ":"
            #typestr = typestr[:-1]
            typelist = [aftype(a,f) for a,f in zip(i[0],i[1])]
            typedir = paramdict[bodymapping[len(i[0])]][i[2]]
            tt = tuple(typelist)
            #if tt in typedir:
            if typedir.index(tt) >= 0:
                # another term .. append
                typedir.appenditem(tt, (i[3],i[4]))
            else:
                typedir[tt] = [(i[3],i[4])]
        return paramdict
 
    def get_params(self,FF, atypes, ptype, potential,fitsystem):
        """
        Method to look up parameter sets in the DB
        
        Parameters:
            FF (str): Name of the FF the parameters belong to
            atypes (str): list of atypes belonging to the term
            ptype (str): type of requested term
            potential (str): type of requested potential
            fitsystem (str): name of the FFfit/reference system the
                parameterset is obtained from

        Returns:
            list of parameters

        Example:
            >> api.get_params('MOF-FF', 'c3_c2h1:c3_c2h1', 'bnd', 'mm3', 'benzene')
        """
        assert type(FF) == type(ptype) == type(atypes) == type(potential) == str
        atypes, fragments = self._format_atypes(atypes,ptype, potential)
        params = self.mfp.get_params(FF, atypes, fragments, ptype, potential, fitsystem)
        return params[0]
           

    def list_FFrefs(self,FF):
        """
        Method to list names and meta properties of all available reference systems in the DB
        
        Parameters:
            FF (str): Name of the FF the reference systems belong to, give "*" in order to 
                get all available references independent from the FF

        Returns:
            dictionary of reference systems 
        """
        res = self.mfp.list_FFrefs(FF)
        dic = {}
        for i in res:
            dic[i[0]] = i[1:]
        return dic


    @download("FFref")
    def get_FFref_graph(self,name, out='file'):
        """
        Downloads the reference system in mfpx file format
        
        Parameters:
            name (str): name of the reference system
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'hdd'
        """
        assert type(name) == str
        lines = self.mfp.get_FFref_graph(name)
        return lines


    @batch_download("FFref")
    def get_FFrefs_graph(self,names,out='file'):
        """
        Downloads the reference systems as mfpx files

        Parameters:
            names (list): names of the reference systems which should be
                downloaded
            out (str,optional): if "file", mfpx files are written to file,
                if "str" files are returned as dictionary.
        """
        return self.mfp.get_FFrefs_graph(names)


    @download("FFref", binary = True)
    def get_FFref(self,name):
        """
        Method to download the available reference information in the hdf5 file format.
        
        Parameters:
            name (str): name of the entry in the DB
        """
        assert type(name) == str
        bstr = self.mfp.get_FFref(name).data
        return bstr

    @download("FFfrag")
    def get_FFfrag(self,name, out='file'):
        """
        Downloads a FF fragment in mfpx file format
        
        Parameters:
            name (str): name of the fragment
            out    (str,optional): if 'file', mfpx file is written to file,
                if 'mol' mol object is returned, if 'str' data is returned
                as string, defaults to 'file'
        """
        assert type(name) == str
        lines = self.mfp.get_FFfrag(name)
        return lines

    @batch_download("FFfrag")
    def get_FFfrags(self, names, out='file'):
        """
        Downloads a list of FF fragments in mfpx file format
        
        Parameters:
            names (list): names of the fragment as list
            out    (str,optional): if 'file', mfpx files is written to files,
                if 'str' data is returned as dictionary, defaults to 'file'
        """
        return self.mfp.get_FFfrags(names)

    def list_FFfrags(self):
        """
        Method to list names and meta properties of all available FFfrags in the DB
        """
        return self.mfp.list_FFfrags()

    def list_special_atypes(self):
        """
        Method to get a dictionary of atypes with special properties
        """
        res = self.mfp.list_special_atypes()
        dic = {"linear": [], "sqp":[]}
        for l in res:
            af = aftype(l[0], l[1])
            dic[l[2]].append(af)
        return dic

    def create_fit(self, FF, ref, azone=None, atfix = None, comment = "",parent=None):
        """
        Method to create a FFfit entry in the database which is necessary
        for storing parameters for a predefined FF

        Parameters:
            FF (str): name of the FF
            ref (str): name of the reference system
            azone (list): list of integers describing the active zone of the fit
            atfix (dict): dictionary containing special atypes information
            comment (str): comment
            parent (str): name of the parent reference system the actual system
                should be linked to, defaults to None

        """
        return self.mfp.create_fit(FF, ref, azone, atfix, comment,parent)

    def set_params(self, FF, atypes, ptype, potential, fitsystem,params):
        """
        Method to upload parameter sets in the DB

        Parameters:
            FF (str): Name of the FF the parameters belong to
            atypes (str): list of atypes belonging to the term
            ptype (str): type of requested term
            potential (str): type of requested potential
            params (list): parameterset
            fitsystem (str): name of the FFfit/reference system the
                parameterset is obtained from

        Example:
            >> api.set_params('MOF-FF', 'c3_c1o2@co2:c3_c3@ph','bnd','mm3','CuPW',[5.6,1.4])
        """
        #assert type(FF) == type(ptype) == type(potential) == type(atypes) == str
        assert type(params) == list
        atypes, fragments = self._format_atypes(atypes,ptype, potential)
        rl = {i[0]:i[1] for i in allowed_potentials[ptype]}[potential]
        if len(params) != rl:
            raise ValueError("Required lenght for %s %s is %i" %(ptype,potential,rl))
        ret = self.mfp.set_params(FF, atypes, fragments, ptype, potential, fitsystem,params)
        return ret

    def set_equiv(self, FF, fitsystem, condition, equivalence):
        """
        Method to set an equivalence.

        Parameters:
            FF (str): Name of the FF the equivalence belongs to
            fitsystem (str): Name of the FFfit/reference system the 
                equivalence belongs to
            condition (str): atypes of bonded partners
            equivalence (str): atypes indicating the equivalence

        Example:
            >> api.set_equiv('MOF-FF', 'CuPW', 'c3_c1o2@co2:c3_c3@ph','c3_c3@ph:c3_c2h1@ph')

            Here an equivalence is generated for the case that the atom with atomtype
            c3_c1o2@co2 is bonded to atom c3_c3@ph. If this is the case the atomtype
            c3_c3@ph is changed to c3_c2h1@ph.
        """
        self.set_params(FF, condition, 'bnd', 'equiv', fitsystem, 
                equivalence.split(':'))
        return
 
    def delete_params(self, FF, ref, tables = [1,2,3,4]):
        """
        Method to delete params from the database. 

        Parameters:
        
            FF (str): Name of the FF the parameters belong to
            ref (str): Name of the reference system the parameters belong to
            tables(list): list specifying which parameters tables should be
                deleted, 1 for onebody, 2 for twobody, 3 for threebody, 
                4 for fourbody, giving [1,2,3,4] as argument results in
                deleting all parameters from the fit, defaults to [1,2,3,4]
        """
        self.mfp.delete_params(FF, ref, tables)
        return
   
    def set_params_interactive(self, FF, atypes, ptype, potential, fitsystem, params):
        """
        Method to upload parameter sets in the DB interactively. You can check and modify
        the data, from the command line interactively before the upload.

        Parameters:
            FF (str): Name of the FF the parameters belong to
            atypes (str): list of atypes belonging to the term
            ptype (str): type of requested term
            potential (str): type of requested potential
            params (list): parameterset
            fitsystem (str): name of the FFfit/reference system the parameterset is obtained from
        """
        stop = False
        while not stop:
            print("--------upload-------")
            print("FF      : %s" % FF)
            print("atypes  : " +len(atypes)*"%s " % tuple(atypes))
            print("type    : %s" % ptype)
            print("pot     : %s" % potential)
            print("ref     : %s" % fitsystem)
            print("params  : ",params)
            print("--------options---------")
            print("[s]: skip")
            print("[y]: write to db")
            print("[a]: modify atypes")
            print("[t]: modify type")
            print("[p]: modify pot")
            print("[r]: modify ref")
            x = input("Your choice:  ")
            if x == "s":
                stop = True
                print("Entry will be skipped")
            elif x == "y":
                ret = self.set_params(FF, ":".join(atypes), ptype, potential, fitsystem, params)
                print(ret)
                if type(ret) != int:
                    "Error occurred during upload, try again!"
                else:
                    print("Entry is written to db")
                    stop = True
            elif x == "a":
                inp = input("Give modified atypes:  ")
                atypes = inp.split()
            elif x == "t":
                ptype = input("Give modified type:  ")
            elif x == "p":
                potential = input("Give modified pot:  ")
            elif x == "r":
                fitsystem = input("Give modified ref:  ")
        return


if __name__ == '__main__':
    from mofplus import admin
    if len(sys.argv) > 1:   
        if sys.argv[1] == "user":
            api = user.user_api(banner=True)
        elif sys.argv[1] == "admin":
            api = admin.admin_api(banner=True)
        elif sys.argv == "ff":
            api = FF_api(banner=True)
    else:
        api = FF_api(banner=True)



