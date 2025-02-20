"""a simple wrapper to run the javascript systrekey from python

    added a class for systrekey handling

    author: R. Schmid
"""

import numpy as np
import os
import subprocess
import molsys


"""
INSTALLATION Instructions:

For this to work you need to have the javascript systrekey code from Olaf Delgado-Friedrichs installed.

You need to have a working node.js environment working (do not use the distro version .. they are often too old) from https://nodejs.org

then clone the systreKey repo from here https://github.com/odf/systreKey to the same directory where your molsys repo is located (e.g. ~/sandbox)
because systre_path is determined from the root path of molsys.
For the installation follow the instructions from the systreKey repos README 

    git clone https://github.com/odf/systreKey.git
    cd systreKey
    npm install
    npm run build

Do not forget to run "npm run build" every time you pulled updates from Olaf's repo.

"""


# get my path 
molsys_path = os.path.dirname(molsys.__file__)
systre_path = os.path.dirname(os.path.dirname(molsys_path)) + "/systreKey"

# check presence of node in order to run javascript code
node_avail = (subprocess.call(args=["which", "node"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0)


class systre_db:

    def __init__(self, arc_file):
        self.arc_file = arc_file
        self.key2name = {}
        self.name2key = {}
        self.arc_read = False
        self.read_arc()
        return

    def read_arc(self):
        global db_key2name, db_name2key, arc_read
        if os.path.isfile(self.arc_file):
            f = open(self.arc_file, 'r')
            for line in f:
                sline = line.split()
                if len(sline)>0:
                    if sline[0] == 'key':
                        key = sline[1:]
                    elif sline[0] == 'id':
                        name = sline[1]
                    elif sline[0] == "end":
                        # end of record .. store in directories only 3dim nets
                        if key[0] == "3":
                            key = " ".join(key)
                            if key in self.key2name:
                                print ("WARNING the following systrekey is already registered for %s" % self.key2name[key])
                                print (key)
                            self.key2name[key] = name
                            self.name2key[name] = key
                    else:
                        pass
            f.close()
            self.arc_read = True
        else:
            print("""
            WARNING: the file %s
            is not available. Please link it into the molsys/util directory.

            To get rid of this annoying warning message just du the following:
            - go to your molsys installation directory
            - cd molsys/util
            - wget https://rcsr.anu.edu.au/downloads/RCSRnets-2019-06-01.arc
            - ln -s RCSRnets-2019-06-01.arc RCSR.arc
            """ % self.arc_file)
        return

    def get_key(self, name):
        if self.arc_read:
            return self.name2key[name]
        else:
            return "None"

    def get_name(self, key):
        if self.arc_read:
            return self.key2name[key]
        else:
            return "None"

# path to the curent arc file -- should be called RCSR.arc by default (softlink to current version)
RCSR_arc_file = os.path.dirname(__file__)+"/RCSR.arc"
RCSR_db = systre_db(RCSR_arc_file)

db_key2name = RCSR_db.key2name
db_name2key = RCSR_db.name2key


def str2edge(s):
    l = [int(i) for i in s.split()]
    e = l[:2]
    l = np.array(l[2:])
    return e,l

class lqg:

    def __init__(self, edges, labels):
        self.edges = edges
        self.labels = labels
        self.is_systrekey = None
        self.systrekey = None
        self.ne = len(edges)
        nv = 0
        for e in self.edges:
            for v in e:
                if v > nv:
                    nv = v
        self.nv = nv+1
        return

    @classmethod
    def from_string(cls, lqgs):
        lqg = lqgs.split()
        assert len(lqg)%5 == 0
        ne = len(lqg)//5
        edges = []
        labels = []
        for i in range(ne):
            edges.append([int(lqg[i*5])-1, int(lqg[i*5+1])-1])
            labels.append([int(lqg[i*5+2]), int(lqg[i*5+3]), int(lqg[i*5+4]),])
        return cls(edges, labels)

    def __repr__(self):
        out = "3 "
        for e,l in zip(self.edges, self.labels):
            out += "%d %d %d %d %d " % (e[0]+1, e[1]+1, l[0], l[1], l[2])
        return out[:-1]     # remove the last space 

    def pretty_print(self):
        for i in range(self.ne):
            e = self.edges[i]
            l = self.labels[i]
            print ("edge %2d: %3d-%3d %s" % (i+1, e[0], e[1], l))
        return

    def get_edge_index(self, e, l, incr=0):
        """get index of edge

        Args:
            e (list): vertices
            l (np.darray): label

        Returns:
            int: index of the edge (WARNING .. we need to count from 1 in order to discriminate reverse edges as -1)
        """
        if e[0] != e[1]:
            er = e.copy()
            er.reverse()
            for i in range(self.ne):
                if e == self.edges[i]:
                    if l.tolist() == self.labels[i]:
                        return i+incr
                elif er == self.edges[i]:
                    if (l*-1).tolist() == self.labels[i]:
                        return -(i+incr)
                else:
                    pass
        else:
            # case for equal vertices .. edge defined by label only
            for i in range(self.ne):
                if e == self.edges[i]:
                    if l.tolist() == self.labels[i]:
                        return i+incr
                    elif (l*-1).tolist() == self.labels[i]:
                        return -(i+incr)
                else:
                    pass
        print ("ERROR: edge %s %s not found" % (str(e), str(l)))
        print (self.edges)
        print (self.labels)
        return None

    def get_edge_index_nolab(self, e, incr=0):
        """same as above but without labels

        check if the edges are sufficient .. otherwise fail

        Args:
            e (list of int): vertex indices
            incr (int, optional): increment. Defaults to 0.
        """
        sign = 1
        if e[0] > e[1]:
            sign = -1
            e.reverse()
        assert self.edges.count(e) == 1
        i = self.edges.index(e)
        return sign*(i+incr)

    def get_systrekey(self):
        """run javascript systrekey
        
        revised version using json strings to pass data to javascript
        """
        import json
        if self.is_systrekey:
            return self
        if self.systrekey is not None:
            return self.systrekey
        # the systrekey is not yet available -> compute it
        if not node_avail:
            # return nothing to avoid an error
            print("""
            WARNING: systrekey was called but node is not installed to run javascript
            """
            )
            return
        lqg_in = []
        for e,l in zip(self.edges, self.labels):
            el = []
            el.append(int(e[0])+1)
            el.append(int(e[1])+1)
            el.append(l)
            lqg_in.append(el)
        json_lqg = str(lqg_in)
        try:
            json_result = subprocess.check_output(args=["node", molsys_path+"/util/run_systrekey.js", json_lqg, systre_path], stderr=subprocess.STDOUT).decode()
        except subprocess.CalledProcessError as err:
            raw_err = err.stdout.decode().split("\n")
            for el in raw_err:
                if el[:6] == "Error:":
                    err = el
                    break
            print ("ERROR: systrekey says -> %s" % err)
            return
        result = json.loads(json_result)
        skey = result["key"][2:] # cut off the "3 " for the 3D
        self.systrekey = lqg.from_string(skey)
        self.systrekey.is_systrekey = True
        # convert the mappings for vertices to dictionaries with indices -1 (start counting from 0)
        #  and then to a flat list 
        mapping = result["mapping"]
        edge_mapping = result["edgeMapping"]
        sk_mapping = {}
        for k in mapping:
            sk_mapping[int(k)-1] = mapping[k]-1
        sk_edge_mapping = {}
        for e in edge_mapping:
            lqg_e, lqg_l = str2edge(e)
            lqg_e = (np.array(lqg_e)-1).tolist()  # we use indices from 0 internaly but systrekey returns mapping from 1
            lqg_ei = self.get_edge_index(lqg_e, lqg_l)
            sk_e, sk_l  = str2edge(edge_mapping[e])
            sk_e = (np.array(sk_e)-1).tolist()
            sk_ei = self.systrekey.get_edge_index(sk_e, sk_l, incr=1)
            # print ("%20s --> %20s  %4d" % (e, edge_mapping[e], sk_ei))
            sk_edge_mapping[lqg_ei] = sk_ei
        # now convert to a flat list
        self.sk_vmapping = [sk_mapping[i] for i in range(self.nv)]       # if vertices are missing in the dict (which should not happen) we get an error here
        self.sk_emapping = [sk_edge_mapping[i] for i in range(self.ne)]  # if edges are missing we should see an error here
        return self.systrekey


if __name__=="__main__":
    edges = [[0,0], [0,0], [0,0]]
    labels = [[1,0,0], [0,1,0], [0,0,1]]
    g = lqg(edges, labels)
    print (g.get_systrekey())
    print (g.sk_vmapping)
    print (g.sk_emapping)








