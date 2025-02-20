"""systre

This helper module contains functions to convert a systrekey to an embedding by calling systre

REMARKS:
    - systre gives both vertices and edges in terms of fractional coordintes but these can be in the range [0.0, 1.0]
      which means vertices can have either 0.0 or 1.0 as a coordinate which is the same.
      In order to make the edge search working we use convention coords to be in the range [0.0, 1.0[
      (essentially all 1.0 are converted to 0.0)


"""
import os
import numpy as np
import subprocess
import tempfile
import molsys
from molsys.util import unit_cell
from molsys.util import elems

# get my path 
molsys_path = os.path.dirname(molsys.__file__)
systreCmd_path = os.path.dirname(molsys_path) + "/external"
# RS Jan2021 .. looks like Jython 2.7 is not properly working with setting CLASSPATH .. so we search for the Systre*.jar directly
import glob
jars = glob.glob("%s/Systre*.jar" % systreCmd_path)
if len(jars) == 0:
    print("There is no Systre jar-file in %s" % systreCmd_path)
    print("Download Systre-19.6.0.jar or newer from here https://github.com/odf/gavrog/releases into the above path")
    raise ImportError
jars.sort()
systre_jar = jars[-1]
# print ("Using the following systre jar file: %s" % systre_jar)


def run_systre(key, debug=False,run_symm_for_q=False):
    """run systreCmd via jython

    probably this could all be run using jython directly, but i will try to keep these things seperate
    so jypthon calls will be done as a subprocess and we analyze the output.
    
    Args:       
        key (string): systrekey to be converted

    HACK .. skip jython and do it with java directly .. ha this is not a hack but a really good idea .. why did i not do it this way from the start
            jython sucks (sorry)

    """
    lsk = key.split()
    assert lsk[0] == "3"
    key = " ".join(lsk) # cononicalize key string for later comparison
    nedges = int((len(lsk)-1)/5)
    # now generate a cgd input for systre
    delete = True
    if debug:
        delete = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cgd", delete=delete) as fcgd:
        fcgd.write("PERIODIC_GRAPH\nEDGES\n")
        for i in range(nedges):
            edge = lsk[i*5+1:i*5+6]
            fcgd.write("%s\n" % " ".join(edge))
        fcgd.write("END\n")
        fcgd.flush()
        # jython_args = ["jython", "-Dpython.path=%s" % systre_jar, systreCmd_path+"/systreCmd.py"]
        # print (jython_args)
        java_args = ["java", "-cp", systre_jar, "org.gavrog.apps.systre.SystreCmdline"]
        try:
            systre_result = subprocess.check_output(args=java_args + ["--fullUnitCell", fcgd.name], stderr=subprocess.STDOUT)
            if run_symm_for_q:
                systre_result_symm = subprocess.check_output(args=java_args + [fcgd.name], stderr=subprocess.STDOUT)
                q = systre_result_symm.decode().split('Edges:\n')[-1].split('Edge centers:')[0].count('\n')
            else:
                q = -1
            if debug is True and run_symm_for_q is True:
                print('systre output without -u:')
                print(systre_result_symm.decode())
                print('end systre output without -u:')
        except subprocess.CalledProcessError as err:
            raw_err = err.stdout.decode().split("\n")
            print (raw_err)
    # now parse the systre output - we use the regular output but for a P1 embedding (-u / --fullUnitCell)
    if debug:
        print (systre_result.decode())
    systre_result = systre_result.decode().split("\n")
    l = 0 # line pointer
    # first get equivalences
    equiv = []
    line = systre_result[l].split()
    stop = False
    while not stop:
        if len(line)>1:
            if (line[1] == "nodes" or line[1] == "node"):
                stop = True
                break
        l += 1
        line = systre_result[l].split()
    nvert = int(line[0])
    stop = False
    has_equiv = True
    while not stop:
        if len(line)>0:
            if line[0] == "Equivalences":
                stop = True
            if line[0] == "Coordination":
                stop = True
                has_equiv = False
        l += 1
        line = systre_result[l].split()
    equiv = list(range(nvert))
    if has_equiv:
        stop = False
        while not stop:
            equiv[(int(line[0])-1)] = (int(line[2])-1)
            l += 1
            line = systre_result[l].split()
            if len(line) == 0:
                stop = True
    # equiv is not necessarily continuous from 0 to nunique
    symlabels = list(set(equiv))
    rev_equiv = [symlabels.index(e) for e in equiv]
    # now read Coordination sequence
    coord_seq = []
    if has_equiv:
        # read till Coordination ... if not we are there already
        stop = False
        while not stop:
            if len(line)>0:
                if line[0] == "Coordination":
                    stop = True
            l += 1
            line = systre_result[l].split()
    while len(line)>0:
        coord_seq.append(" ".join(line[2:]))
        l += 1
        line = systre_result[l].split()
    # next get space group
    stop = False
    while not stop:
        if len(line) == 5:
            if line[:3] == ["Ideal", "space", "group"]:
                spgroup = line[4][:-1]
                stop = True
        l += 1
        line = systre_result[l].split()
    # get cell params
    stop = False
    while not stop:
        if len(line) == 3:
            if line == ["Relaxed", "cell", "parameters:"]:
                stop = True
        l += 1
        line = systre_result[l].split()
    a = float(line[2][:-1])
    b = float(line[5][:-1])
    c = float(line[8][:-1])
    l += 1
    line = systre_result[l].split()
    alpha = float(line[2][:-1])
    beta  = float(line[5][:-1])
    gamma = float(line[8][:-1])
    # get vertices
    vertices = []
    stop = False
    while not stop:
        if len(line) == 2:
            if line == ["Relaxed", "positions:"] or line == ["Barycentric", "positions:"]:
                stop = True
        l += 1
        line = systre_result[l].split()
    stop = False
    while not stop:
        if line [0] == "Node":
            vertices.append([float(x) for x in line[2:5]])
            l += 1
            line = systre_result[l].split()
        else:
            stop = True
    nvertices = len(vertices)
    assert line[0] == "Edges:"
    # get edges from to
    edge_from = []
    edge_to   = []
    l += 1
    line = systre_result[l].split()
    stop = False
    while not stop:
        if line [0] != "Edge":
            edge_from.append([float(x) for x in line[0:3]])
            edge_to.append([float(x) for x in line[4:7]])
            l += 1
            line = systre_result[l].split()
        else:
            stop = True
    nedges = len(edge_from)
    # now convert edges to a conn/pconn for our molsys topo object
    cell = unit_cell.vectors_from_abc((a, b, c, alpha, beta, gamma))
    fxyz = np.array(vertices)
    # make sure that all fxyz are in [0.0, 1.0[
    fxyz = np.where(fxyz==1.0, 0.0, fxyz)
    xyz = np.dot(fxyz, cell)
    m = molsys.mol.from_array(xyz)
    m.set_cell(cell)
    m.set_empty_conn()
    m.set_empty_pconn()
    # determine conn and pconn from the edges .. first we need to identify the vertices in edge_from
    v1 = np.array(edge_from)
    v2 = np.array(edge_to)
    # get image and base vectors
    v1_img = v1//1.0
    v1= v1%1.0
    v2_img = v2//1.0
    v2 = v2%1.0
    for e in range(v1.shape[0]):
        # get i vertex (from)
        d = fxyz-v1[e]
        d2 = (d*d).sum(axis=1)
        verti = np.argmin(d2)
        assert d2[verti] < 1.0e-4, "could not identify vertex with a short dist of %12.6f" % d2[verti]
        # get j vertex (to)
        d = fxyz-v2[e]
        d2 = (d*d).sum(axis=1)
        vertj = np.argmin(d2)
        assert d2[vertj] < 1.0e-4, "could not identify vertex with a short dist of %12.6f" % d2[vertj]
        # compute image offset
        img = v2_img[e]-v1_img[e]
        # now set conn and pconn (edges from systre are allready bidirectional)
        img = img.astype("int32")
        m.conn[verti].append(int(vertj))
        m.pconn[verti].append(img)
        #print ("bond %3d %3d img %s" % (verti, vertj, str(img)))
        # m.conn[vertj].append(verti)
        # m.pconn[vertj].append(img*-1)
    # DEBUG DEBUG check bonds 
    #for i in range(m.get_natoms()):
    #    for ij, j in enumerate(m.conn[i]):
    #        if i < j:
    #            ii = m.conn[j].index(i)
    #            print ("bond %3d %3d img %s .. reverse img %s" % (i, j, m.pconn[i][ij], m.pconn[j][ii]))
    # conn/pconn done
    el = []
    for i in range(m.get_natoms()):
        e = elems.pse[len(m.conn[i])]
        el.append(e)
    m.set_elems(el)
    m.is_topo = True
    m.use_pconn = True
    # now make sure that the conversion went ok by running systrekey with the built topo .. the key must match the input
    m.addon("topo")
    new_key = m.topo.systrekey # new_key is a lqg class object -> the string is produced by the repr() function
    if key != repr(new_key):
        print ("someting went wrong here")
        print (key)
        print (new_key)
    # at this point we can be sure that the embedding went well and we have all the mappings
    # now the symmetry unique vertices have to be written as strings to atype
    key_mapping = m.topo.sk_vmapping
    sym_mapping = []
    for v in key_mapping:
        sym_mapping.append(str(rev_equiv[v]))   # NOTE: i am not sure anymore what it is and why it is done :-)
    m.set_atypes(sym_mapping)
    # set transitivity values ## 1 is set above already!
    p,r,s =  len(list(set(rev_equiv))), -1,-1 # r and s can not be determined by systre --> 3ds?
    m.topo.set_topoinfo(spgr=spgroup, coord_seq=coord_seq, transitivity='%s %s %s %s' % (p,q,r,s))
    return m


def convert_all():
    """this function converts all systrekeys in the systreky db imported from systrekey into embeddings as long as they are not present, yet.
    """
    from molsys.util import systrekey
    import os
    nnets = len(systrekey.db_name2key.keys())
    i = 0
    for n in systrekey.db_name2key.keys():
        if not os.path.isfile(n+".mfpx"):
            print ("generating embedding for net %s (%d of %d)" % (n, i, nnets))
            m = run_systre(systrekey.db_name2key[n],run_symm_for_q=True)
            m.write(n+".mfpx")
            print ("done!")
        i += 1
    return

def skey_string_to_list(skey):
    """ Converts a systre string key, something like 
    '3 1 2 0 0 0 1 2 1 0 0 1 3 0 0 0 1 3 0 1 0 2 4 0 0 0 2 4 1 0 0 3 4 0 0 1 3 4 0 1 1'
    to a list. Crops of the dimensionality
    """
    slist = []
    skey=skey.split(' ')[1:]
    for i,s in enumerate(skey):
        if i % 5 == 0:
            slist.append([])
        if i % 5 == 0:
            slist[-1].append([int(s)])
        elif i % 5 == 1:
            slist[-1][0].append(int(s))
        elif i % 5 == 2:
            slist[-1].append([int(s)])
        else:
            slist[-1][-1].append(int(s))
    return slist


# for debugging call this with a RCSR name
if __name__=="__main__":
    import sys
    from molsys.util import systrekey
    name = sys.argv[1]
    if len(sys.argv) >2:
        arc = sys.argv[2]
        db = systrekey.systre_db(arc)
        db_name2key = db.name2key
    else:
        db_name2key = systrekey.db_name2key
    assert name in db_name2key
    key = db_name2key[name]
    m = run_systre(key, debug=True)
    m.write(name+".mfpx")


