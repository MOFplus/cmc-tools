#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import numpy as np
import string
import pickle as pickle
import os
import sys
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import copy



## net class dependencies
##from spgr import *
##from spgr import vertex_names
##from ase.spacegroup import Spacegroup
##from ase import Atoms
##import ase
##import unit_cell
##import geoms2 as geo
##import spglib
from collections import defaultdict

# netbase dependencies
# class for net search
import molsys
import weaver.rotator as rotator
import weaver


def make_upload_dictionary(m,name=None,
                           vertex=None,
                           penalties=None,
                           spacegroup='n.a.',
                           spacegroup_number=-1):
    '''
    method to create a dictionary to be used for net upload_bb_penalties
    '''
    data = {}
    # net name
    if name is None:
        name = m.name
    net = {'name':name,
            'spacegroup':spacegroup,
            'spacegroup_number':spacegroup_number,
            'cella':float(m.cellparams[0]),
            'cellb':float(m.cellparams[1]),
            'cellc':float(m.cellparams[2]),
            'cellalpha':float(m.cellparams[3]),
            'cellbeta':float(m.cellparams[4]),
            'cellgamma':float(m.cellparams[5]),
            'p':m.get_natypes(),
            'q':len(m.get_unique_neighbors()),
            'r':-1,
            's':-1,
            'natoms':m.natoms}
    # vertex table
    data['net'] = net
    if vertex is None:
        vertex = make_vertex_dictionary(m)
    data['vertex'] = vertex
    # connectivity
    conn = make_conn_entry(m)
    data['conn'] = conn
    # files - we need topofile + ptxyzfile
    ptxyzfile = name +'_p' + '.txyz'
    topofile = name + '.mfpx'
    write_ptxyz(m,filename=ptxyzfile)
    m.write(topofile,ftype='mfpx')
    filenames = {'ptxyzfile':ptxyzfile,
                 'topofile':topofile}
    ptxyzfilecontent = open(ptxyzfile,'r').read()
    topofilecontent = open(topofile,'r').read()
    files = {topofile:topofilecontent,ptxyzfile:ptxyzfilecontent}
    
    data['filenames'] = filenames
    data['files'] = files  # files contain as string the  file, 
    if penalties is not None:
        data['penalties'] = penalties
    else:
        try:
            f = weaver.framework(name)
            f.net = m
            f.load_geometries()
            penalties = f.scan_vertex_penalties()
            data['penalties'] = penalties
        except:
            print('vertex penalties not available')
            
    return data

def make_vertex_dictionary(m):
    unique_neighbors = m.get_unique_neighbors()
    ## unique is something like this [[['0', '1'], 64], [['0', '2'], 32]]  
    d    = []
    conn = {i:[] for i in list(set(m.get_atypes()))}
    for i,un in enumerate(unique_neighbors):
        #print i,un
        conn[un[0][0]].append(int(un[0][1]))
        if un[0][0] != un[0][1]:
            conn[un[0][1]].append(int(un[0][0]))
    for i,a in enumerate(list(set(m.get_atypes()))):
        dd = {}
        dd.update({'idx':int(a)})
        dd.update({'coordination_number':len(m.conn[m.atypes.index(a)])})
        #dd.update({'symmetry':self.symmetry[i]})   ### TODO: read from html
        #dd.update({'symmetry':None})
        dd.update({'connections':conn[a]})
        d.append(dd)
    return d


def make_conn_entry(m):
    unique_neighbors = m.get_unique_neighbors()
    conn = []
    for i, un in enumerate(unique_neighbors):
        idx0 = m.atypes.index(un[0][0])
        idx1 = m.atypes.index(un[0][1])
        atypes_of_conn0 = [m.atypes[x] for x in m.conn[idx0]]
        count = atypes_of_conn0.count(m.atypes[idx1])
        conn.append([un[0],count])
        if idx0 != idx1:
            atypes_of_conn1 = [m.atypes[x] for x in m.conn[idx1]]
            count = atypes_of_conn1.count(m.atypes[idx0])
            conn.append([[un[0][1],un[0][0]],count])
    print('conn:', conn)
    #for i,c in enumerate(conn):
    #    print i,c
    #    c[0][0] = int(c[0][0])
    #    c[0][1] = int(c[0][1])
    return conn
        
def write_ptxyz(m,filename=False):
    ff=open(filename,'w')
    newxyz = (copy.copy(m.xyz)).tolist()
    newn = len(newxyz)
    newconn = [[] for i in range(len(m.conn))]
    node_coordination = {m.atypes[x]:len(m.conn[x]) for x in range(m.natoms)}
    kinds = [m.atypes[x] for x in range(m.natoms)]
    newatype = [m.elems[x] for x in range(m.natoms)]
    newpconn = []
    newkinds = copy.copy(kinds)
    for i in range(m.natoms):
        for j in range(len(m.conn[i])):
            c = m.conn[i][j]
            p = m.pconn[i][j]
            if not (p==0).all():
                newxyz.append(m.get_image(m.xyz[c],p).tolist())
                newconn[i].append(newn)
                newconn.append([i])
                newkinds.append(copy.copy(kinds[c]))
                newatype.append(m.elems[c])
                newn += 1
            else:
                newconn[i].append(c)
            #print i,j,newconn
            pass
    ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f\n' % (len(newxyz),m.cellparams[0],m.cellparams[1],m.cellparams[2],m.cellparams[3],m.cellparams[4],m.cellparams[5]) )

    for i in range(newn):
        line = ("%3d %-3s" + 3*"%12.6f" + " %5s") % \
            tuple([i+1]+[newatype[i]]+ newxyz[i] + [newkinds[i]])
        conn = (numpy.array(newconn[i])+1).tolist()
        if len(conn) != 0:
            line += (len(conn)*"%7d") % tuple(sorted(conn))
        ff.write("%s \n" % line)
    ff.close()
    return
        
    
    
    


###### BEWARE THOSE CLASSES. THEY ARE PROBABLY NOT WORKING ANYMORE, since
###### THEY ARE VERY OLD. I KEEP THEM HERE SO THEY ARE NOT LOST

class net(object):
    '''
    This net class was originally used by JK to get going with the RCSR nets
    now this should be replaced with molsys, but there is still a lot of stuff here 
    that is not yet in molsys, so i decided to put it here for the moment
    '''
    def __init__(self,ss=[]):
        self.name = ''
        self.natoms = 0
        self.n = 0
        self.sg_name = ''
        self.sg_number = -1
        self.sg = []
        self.sgerror = False
        self.cell = numpy.zeros((1,6))
        self.cellvec = numpy.zeros((3,3))
        self.node = []
        self.node_coordination = []
        self.conn_quality = ['none', 'none']
        self.error = []
        self.pconn = []
        self.use_pconn=True
        self.verts = []
        self.edge1 = []
        self.edge2 = []
        self.edges1= []
        self.edges2= []
        self.ekinds1=[]
        self.ekinds2=[]
        self.centers=[]
        self.ckinds =[]
        self.edge_center=[]
        self.conn=[]
        self.conn_level = -1
        self.xyz=[]
        self.images_cellvec = []
        self.good = True
        self.unique_neighbors = False
        self.p = -1
        self.q = -1
        self.r = -1
        self.s = -1
        if ss:
            self.read_RCSR(ss)
        return
    
    #### string parsing routines
    
    def parse_3dall(self,txt):
        self.tdall = txt
        lines = txt.split('\n')[1:]
        self.tdall_id= int(lines.pop(0))
        self.name= lines.pop(0).split()[0]
        self.embedding = lines.pop(0)
        nsymbols = int(lines.pop(0).split('!')[0])
        self.symbols = [lines.pop(0) for i in range(nsymbols)]
        nnames = int(lines.pop(0).split('!')[0])
        self.knownas = [lines.pop(0) for i in range(nnames)]
        nnames = int(lines.pop(0).split('!')[0])
        self.knownas += [lines.pop(0) for i in range(nnames)]
        nkeys = int(lines.pop(0).split('!')[0])
        self.keywords = [lines.pop(0) for i in range(nkeys)]
        nrefs = int(lines.pop(0).split('!')[0])
        self.references = [lines.pop(0) for i in range(nrefs)]
        t = lines.pop(0).split()
        self.sg_name,self.sg_number = t[0],int(t[1])
        self.cell = numpy.array(list(map(float,lines.pop(0).split())))
        self.make_cellvec()
        
        nverts = int(lines.pop(0))
        
        self.symbolic   = []
        self.wyckoff= []
        self.symmetry = []
        self.order = []
        for i in range(nverts):
            self.node_coordination.append(int(lines.pop(0).split()[-1]))
            self.node.append(list(map(float,lines.pop(0).split())))
            self.symbolic.append(lines.pop(0))
            self.wyckoff.append(lines.pop(0))
            self.symmetry.append(lines.pop(0))
            self.order.append(int(lines.pop(0)))
    
        nedges = int(lines.pop(0))
        
        self.center_symbolic   = []
        self.center_wyckoff= []
        self.center_symmetry = []
    
        for i in range(nedges):
            temp = lines.pop(0)
            self.edge_center.append(list(map(float,lines.pop(0).split())))
            self.center_symbolic.append(lines.pop(0))
            self.center_wyckoff.append(lines.pop(0))
            self.center_symmetry.append(lines.pop(0))
        #r s is missing ! at some point, add it!
        
        return
    
        
    def parse_systre(self,txt):
        pass
        return
    
    #### file writer & I/O
    
    def save_database_item(self,path=None,penalties=None):
        parentpath=os.getcwd()
        if not path: path = './data/database/'
        if os.listdir(path).count(self.name) ==0:
            os.mkdir(path+self.name)
        os.chdir(path+self.name)
        #pickle.dump(self, open(self.name+'.pickle','wb'))
        # first do the stuff concerning the net table
        net = {'name':self.name,
               'spacegroup':self.sg_name,
               'spacegroup_number':self.sg_number,
               'cella':float(self.cell[0]),
               'cellb':float(self.cell[1]),
               'cellc':float(self.cell[2]),
               'cellalpha':float(self.cell[3]),
               'cellbeta':float(self.cell[4]),
               'cellgamma':float(self.cell[5]),
               'p':len(self.node),
               'q':len(self.edge_center),
               'r':-1,
               's':-1,
               'natoms':self.natoms}
        txyzfile = self.name+'.txyz'
        ptxyzfile = self.name +'_p' + '.txyz'
        topofile = self.name + '.topo'
        self.write_txyz(filename=txyzfile)
        self.write_ptxyz(filename=ptxyzfile)
        self.write_topo(filename=topofile)
        filenames = {'txyzfile':txyzfile,
                     'ptxyzfile':ptxyzfile,
                     'topofile':topofile}
        ### net is now done ! now go for vertex table
        self.make_vertex_db_entry()
        vertex = self.vdct
        self.get_unique_conns()
        conns = self.uniconns
        
        penf = {}
        if penalties:
            for i,idx in enumerate(penalties.keys()):
                peni = {}
                for j,p in enumerate(penalties[idx]):
                    name = p[4].split('-')[-1].split('.')[0]
                    peni.update({name:float(p[5])})
                penf.update({str(idx):peni})
        files = {}
        for i,f in enumerate(filenames.keys()):
            fname = filenames[f]
            ff = open(fname,'r')
            files.update({fname:ff.read()})
            ff.close()
            
        
        alldata = {'net':net,
                   'vertex':vertex,
                   'filenames':filenames,
                   'files':files,
                   'conn':conns,
                   'penalties':penf}
        pickle.dump(alldata, open(self.name+'.pickle','wb'))
        
        os.chdir(parentpath)
        return alldata
    
    def write_blueprint(self,filename):
        ff=open(filename,'w')
        ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f %s\n' % (len(self.xyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5], 'molden' ))
        ff.write('BLUEPRINT FOR '+self.name+'\n')
        for i,x in enumerate(self.xyz):
            if self.node_coordination[self.kinds[i]] <10:
                elem = vertex_names[self.node_coordination[self.kinds[i]]]#+'_'+str(self.kinds[i])
            else:
                elem ='X'+str(self.node_coordination[self.kinds[i]])+'_'+str(self.kinds[i])
            ff.write('%s %16.10f %16.10f %16.10f\n' % (elem, x[0],x[1],x[2]) )
        ff.close()
        return
    
    def write_xyz(self, filename):
        ff=open(filename,'w')
        ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f %s\n' % (len(self.xyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5], 'molden' ))
        ff.write('net '+self.name+'\n')
        for i,x in enumerate(self.xyz):
            elem = vertex_names[self.node_coordination[self.kinds[i]]]
            ff.write('%s %16.10f %16.10f %16.10f\n' % (elem, x[0],x[1],x[2]) )
        ff.close()
        return
    
    def write_txyz(self,filename=False):
        if not filename:
            filename = self.name+'.txyz'
        ff=open(filename,'w')
        ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f\n' % (len(self.xyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5]) )
        #ff.write('BLUEPRINT FOR '+self.name+'\n')
        #for i,x in enumerate(self.xyz)
        for i in range(self.natoms):
            line = ("%3d %-3s" + 3*"%12.6f" + " %5s") % \
               tuple([i+1]+[vertex_names[self.node_coordination[self.kinds[i]]]]+ self.xyz[i].tolist() + [self.kinds[i]])
            conn = (numpy.array(self.conn[i])+1).tolist()
            if len(conn) != 0:
                line += (len(conn)*"%7d") % tuple(conn)
            ff.write("%s \n" % line)
        ff.close()
        pass
    
    def write_topo(self,filename=False):
        if not filename: filename=self.name+'.topo'
        ff=open(filename,'w')
        ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f\n' % (len(self.xyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5]) )
        for i in range(self.natoms):
            line = ("%3d %-3s" + 3*"%12.6f" + " %5s") % \
               tuple([i+1]+[vertex_names[self.node_coordination[self.kinds[i]]]]+ self.xyz[i].tolist() + [self.kinds[i]])
            conn = (numpy.array(self.conn[i])+1).tolist()
            pconn = self.pconn[i]
            pimg = []
            for pc in pconn:
                for ii,img in enumerate(images):
                    if all(img==pc): 
                        pimg.append(ii)
                        break
            if len(conn) != 0:
                for cc,pp in zip(conn,pimg):
                    if pp < 10:
                        line += "%8d/%1d" % (cc,pp)
                    else:
                        line += "%7d/%2d" % (cc,pp)
                #line += (len(conn)*"%7d") % tuple(conn)
            ff.write("%s \n" % line)
        ff.close()
    
    def write_ptxyz(self,filename=False):
        if not filename:
            filename = self.name+'_p'+'.txyz'
        ff=open(filename,'w')
        newxyz = (copy.copy(self.xyz)).tolist()
        newn = len(newxyz)
        newconn = [[] for i in range(len(self.conn))]
        newatype = [vertex_names[self.node_coordination[self.kinds[i]]] for i in range(self.natoms)]
        #newconn = []
        newpconn = []
        newkinds = copy.copy(self.kinds)
        #ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f\n' % (len(self.xyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5]) )
        for i in range(self.natoms):
            for j in range(len(self.conn[i])):
                #work on conn and pconn to find periodic connecions, delete them and add a new atom  for the sake of visualization!
                c = self.conn[i][j]
                p = self.pconn[i][j]
                #print i,j,c,p
                if not (p==0).all(): # atom c has to be created inside the proper neighboring cell
                    #dispvect = numpy.sum(self.cellvec*p[:,numpy.newaxis],axis=0)   # could have used self.get_image() see below!
                    #newxyz.append((self.xyz[c]+dispvect).tolist())
                    newxyz.append(self.get_image(self.xyz[c],p).tolist())
                    newconn[i].append(newn)
                    newconn.append([i])
                    newkinds.append(copy.copy(self.kinds[c]))
                    newatype.append(vertex_names[self.node_coordination[self.kinds[c]]])
                    newn += 1
                else:
                    newconn[i].append(c)
                #print i,j,newconn
                pass
        ff.write('%i %16.10f %16.10f %16.10f %10.6f %10.6f %10.6f\n' % (len(newxyz),self.cell[0],self.cell[1],self.cell[2],self.cell[3],self.cell[4],self.cell[5]) )
        #print newatype
        #print newconn
        #print newxyz
        #print newn, len(newatype), len(newconn), len(newxyz), len(newkinds)
        #print newkinds
        for i in range(newn):
            line = ("%3d %-3s" + 3*"%12.6f" + " %5s") % \
               tuple([i+1]+[newatype[i]]+ newxyz[i] + [newkinds[i]])
            conn = (numpy.array(newconn[i])+1).tolist()
            if len(conn) != 0:
                line += (len(conn)*"%7d") % tuple(sorted(conn))
            ff.write("%s \n" % line)
        ff.close()
        return # newxyz,newconn,newn
        #pass
    
                        #def get_image(self,xyz, img):
                        #xyz = numpy.array(xyz)
                        #try:
                            #l = len(img)
                            #dispvec = numpy.sum(self.cellvec*numpy.array(img)[:,numpy.newaxis],axis=0)
                        #except TypeError:
                            #dispvec = numpy.sum(self.cellvec*numpy.array(images[img])[:,numpy.newaxis],axis=0)
                        #return xyz + dispvec


                        #if numpy.sum(numpy.abs(self.pconn[i][j])) == 0:
                            #ax.plot([self.xyz[i][0],self.xyz[conn[j]][0]],[self.xyz[i][1],self.xyz[conn[j]][1]],[self.xyz[i][2],self.xyz[conn[j]][2]])
                        #else:
                            #xyznew = self.get_image(self.xyz[conn[j]],self.pconn[i][j])
                            #ax.scatter(xyznew[0],xyznew[1],xyznew[2],color='orange')
                            #ax.plot([self.xyz[i][0],xyznew[0]],[self.xyz[i][1],xyznew[1]],[self.xyz[i][2],xyznew[2]])
                    #if shift:
                        #dispvect = num.sum(self.cell*num.array([a-1,b-1,c-1])[:,num.newaxis],axis=0)
                    #else:
                        #dispvect = num.sum(self.cell*num.array([a,b,c])[:,num.newaxis],axis=0)
                    #new_xyz += (self.xyz+dispvect).tolist()    

    def get_unique_conns(self):
        self.uniconns=[]
        nkinds = self.kinds[-1]+1
        for i in range(nkinds):
            idx = self.kinds.index(i)
            ddict = defaultdict(lambda : 0) #defaultdict is nice ! no more errors and trys
            for j,c in enumerate(self.conn[idx]):
                ckind = self.kinds[c]
                ddict.update({(i,ckind):ddict[(i,ckind)]+1}) # sum up No conns to each kind
            for k in list(ddict.keys()):
                self.uniconns.append([k, ddict[k]])
        
        #seems to work, WARNING: connections to the same kind are not stored redundantly
        #sanity check
        #[n.kinds[j] for j in [i for i in n.conn[n.kinds.index(0)]]]
        cc = [0 for i in range(nkinds)]
        for c in self.uniconns:
            cc[c[0][0]] += c[1]
        #print cc
        for i,c in enumerate(cc):
            if c != self.node_coordination[i]:
                print('warning, inconsisitent uniconns at net:', self.name)
        # nice, works for all 'good.pickle' nets, 10.5.2016    
        return 
    
    def get_unique_neighbors(self):
        nkinds = self.kinds[-1]
        un = []
        counter = []
        for i,c in enumerate(self.conn):
            for j,cc in enumerate(c):
                neighs = sorted([self.kinds[i], self.kinds[cc]])
                kneighs= [[self.node_coordination[neighs[0]],self.node_coordination[neighs[1]]],[neighs[0],neighs[1]]]
                try: 
                    ii=un.index(kneighs)
                    counter[ii] += 1
                except:
                    un.append(kneighs)
                    counter.append(1)
        self.unique_neighbors = []
        for i in range(len(un)):  
            self.unique_neighbors.append([un[i],counter[i]])
        #self.unique_neighbors = un
        return un
    
    def make_dict(self):
        d = {}
        
        d['name'] = self.name
        d['spacegroup'] = self.sg_name
        d['spacegroup_number'] = self.sg_number
        d['origin_choice'] = self.sg.setting
        d['cella'] = self.cell[0]
        d['cellb'] = self.cell[1]
        d['cellc'] = self.cell[2]
        d['cellalpha'] = self.cell[3]
        d['cellbeta'] = self.cell[4]
        d['cellgamma'] = self.cell[5]
        d['p'] = len(self.node)
        d['q'] = len(self.edge1)
        d['r'] = -1  ### TODO read from html
        d['s'] = -1  ### TODO read from html
        d['natoms'] = self.natoms
        self.dct = d
        return d
        
    def make_vertex_db_entry(self):
        self.get_unique_neighbors()
        d    = []
        conn = [[] for i in range(len(self.node_coordination))]
        for i,un in enumerate(self.unique_neighbors):
            print(i,un)
            #conn.update({un[1][0]:un[1][1]})
            conn[un[0][1][0]].append(un[0][1][1])
            if un[0][1][0] != un[0][1][1]:
                conn[un[0][1][1]].append(un[0][1][0])
        for i in range(len(self.node)):
            dd = {}
            dd.update({'idx':i})
            dd.update({'coordination_number':self.node_coordination[i]})
            #dd.update({'symmetry':self.symmetry[i]})   ### TODO: read from html
            dd.update({'symmetry':None})
            dd.update({'connections':conn[i]})
            d.append(dd)
        
        self.vdct = d
        return d

#### symmetry data generation                    
                    
    def clean(self):
        self.natoms = self.n
        self.error = []
        self.conn_level = -1
        self.use_pconn =True
        self.conn=[]
    
    def make_cellvec(self):
        self.cellvec = unit_cell.vectors_from_abc(self.cell)
        self.images_cellvec = numpy.dot(images,self.cellvec)
        return

        
        
    def read_RCSR(self,ss):
        if ss:
            for i,s in enumerate(ss):
                if s[0]=='NAME':
                    self.name = s[1]
                    #print self.name
                if s[0]=='GROUP':
                    self.sg_name = s[1]
                    if self.sg_name.find(':') != -1:
                        self.sg_name = self.sg_name[0:self.sg_name.find(':')]
                    #print self.name
                if s[0] == 'CELL':
                    self.cell = numpy.array(list(map(float,s[1:])))
                    self.cellvec = unit_cell.vectors_from_abc(self.cell)
                    self.images_cellvec = numpy.dot(images,self.cellvec)
                    #print self.cell
                if s[0] == 'NODE':
                    self.node.append(list(map(float,s[3:])))
                    self.node_coordination.append(int(s[2]))
                if s[0] == 'EDGE':
                    self.edge1.append(list(map(float,s[1:4])))
                    self.edge2.append(list(map(float,s[4:7])))
                try:
                    blubb=s[1]
                    if s[1] == 'EDGE_CENTER':
                        self.edge_center.append(list(map(float,s[2:])))
                except IndexError:
                    pass
            self.verts = copy.copy(self.node_coordination)
            self.verts.sort()
            self.get_spacegroup()
            self.get_xyz()
            self.get_edges()
            self.scale()
            #self.get_ee
        
    def get_xyz(self,sprec=1e-3):
        if self.sg:
            try:
                self.xyz, self.kinds = self.sg.equivalent_sites(self.node,symprec=sprec)
            except ase.spacegroup.spacegroup.SpacegroupValueError:
                print(self.name,'has', self.sg_name, self.sg_number, ': something is wrong with its spacegroup!!')
                self.sg=[]
                self.error.append('xyz_expansion_failed1')
            except ValueError:
                print(self.name,': valueError at get_xyz()')
                self.error.append('xyz_expansion_failed2')
        else:
            print('spacegroup could not be determined !')
            self.sgerror = True
        self.natoms = len(self.xyz)
        self.n = self.natoms
        for i in range(self.natoms):
            self.conn.append([])
        
        return
    def get_xyz_using_edges(self,sprec=1e-3):
        if self.sg:
            try:
                xyz, kinds = self.sg.equivalent_sites(self.node,symprec=sprec)
            except ase.spacegroup.spacegroup.SpacegroupValueError:
                print(self.name,'has', self.sg_name, self.sg_number, ': something is wrong with its spacegroup!!')
                self.sg=[]
                self.error.append('xyz_expansion_failed1')
            except ValueError:
                print(self.name,': valueError at get_xyz()')
                self.error.append('xyz_expansion_failed2')
        else:
            print('spacegroup could not be determined !')
            self.sgerror = True
        self.natoms = len(self.xyz)
        self.n = self.natoms
        for i in range(self.natoms):
            self.conn.append([])
        
        return
    
    def scale(self):
        for i in range(len(self.xyz)):
            self.xyz[i] =  numpy.dot(self.xyz[i],self.cellvec)
        for i in range(len(self.edges1)):
            for j in range(len(self.edges1[i])):
                self.edges1[i][j] = numpy.dot(self.edges1[i][j],self.cellvec)
        for i in range(len(self.edges2)):
            for j in range(len(self.edges2[i])):
                self.edges2[i][j] = numpy.dot(self.edges2[i][j],self.cellvec)
        for i in range(len(self.centers)):
            for j in range(len(self.centers[i])):
                self.centers[i][j] = numpy.dot(self.centers[i][j],self.cellvec)
        return
    
    def get_edges(self,sprec=1e-3):
        self.edges1,self.edges2,self.ekinds1,self.ekinds1 = [],[],[],[]
        self.centers,self.ckinds = [],[]
        if self.sg:
            for i,e in enumerate(self.edge1):
                try:
                    e1,k1 =self.sg.equivalent_sites(e,symprec=sprec)
                    self.edges1.append(e1)
                    self.ekinds1.append(k1)
                except ase.spacegroup.spacegroup.SpacegroupValueError:
                    print(self.name,'has', self.sg_name, ': something is wrong with its edge1 !!')
                    print(self.edge1)
            for i,e in enumerate(self.edge2):
                try:
                    e1,k1 =self.sg.equivalent_sites(e,symprec=sprec)
                    self.edges2.append(e1)
                    self.ekinds2.append(k1)
                except ase.spacegroup.spacegroup.SpacegroupValueError:
                    print(self.name,'has', self.sg_name, ': something is wrong with its edge2 !!')
                    print(self.edge2)
            for i,e in enumerate(self.edge_center):
                try:
                    e1,k1 =self.sg.equivalent_sites(e,symprec=sprec)
                    self.centers.append(e1)
                    self.ckinds.append(k1)
                except ase.spacegroup.spacegroup.SpacegroupValueError:
                    print(self.name,'has', self.sg_name, ': something is wrong with its edge_center !!')
                    print(self.edge_center)
        else:
            self.sgerror=True
            pass
            #print 'spacegroup could not be determined, sry for duplicate notice!', self.name
        self.natoms = len(self.xyz)
        return
    
    def report(self):
        for i in self.xyz:
            print(i)
        print('-- edges --')
        for i in range(len(self.edges1)):
            for j in range(len(self.edges1[i])):
                print(i, self.edges1[i][j])
            for j in range(len(self.edges2[i])):
                print(i, self.edges2[i][j])
            for j in self.centers[i]:
                print(i, j)
    
    
    def get_spacegroup(self,sg_setting=2):
        
        if self.sg_number == -1:
            spgnum = spgr[self.sg_name]
        else:
            spgnum = self.sg_number
        try: 
            self.sg = Spacegroup(spgnum,setting=sg_setting)
        except:
            try:
                self.sg = Spacegroup(spgnum,setting=1)
            except:
                print(self.name, 'failed to get its spacegroup class setup')
                return False
        return True

    def reset_sg_setting(self,sg_setting=2):
        self.clean()
        self.get_spacegroup(sg_setting=sg_setting)
        self.get_xyz()
        self.get_edges()
        self.scale()
        return
        
#### Connectivity Detection       
        
    def get_conn(self,fixed_cutoff = 1.05): #  this works in 1330 / 2100 cases (approx)
        self.conn = []
        self.pconn=[]
        self.use_pconn=True
        for i in range(self.natoms):
            self.conn.append([])
            if self.use_pconn: self.pconn.append([])
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                d,r,imgi=self.get_distvec(i,j)
                bond = False
                if fixed_cutoff:
                    if d<fixed_cutoff: bond = True
                #else:
                    #covradi = elements.cov_radii[self.elems[i]]
                    #covradj = elements.cov_radii[self.elems[j]]
                    #if d<(covradi+covradj+cov_rad_buffer) : bond = True
                # exclude pairs testing
                #if exclude_pairs and bond:
                    #el_p1,el_p2 = (self.elems[i], self.elems[j]),(self.elems[j], self.elems[i])
                    #for expair in exclude_pairs:
                        #if (expair == el_p1) or (expair == el_p2):
                            #bond= False
                            #break 
                if bond:
                    if len(imgi)>1 and not self.use_pconn:
                        raise ValueError("Error in connectivity detection: use pconn!!!")
                    for ii in imgi:
                        self.conn[i].append(j)
                        self.conn[j].append(i)
                        if self.use_pconn:
                            image = images[ii]
                            self.pconn[i].append(image)
                            self.pconn[j].append(image*-1)
        return
    
    def get_conn2(self,fixed_cutoff = 0.05,additional_cutoff=None,subspace_size=3): #  this works in 1330 / 2300 cases (approx)
        self.conn = []
        self.pconn = []
        self.use_pconn=True
        for i in range(self.natoms):
            self.conn.append([])
            if self.use_pconn: self.pconn.append([])
        dxyz =  numpy.zeros((self.natoms,self.natoms))
        vxyz =  numpy.zeros((self.natoms,self.natoms,3))
        ixyz = []
        for i in range(self.natoms):
            ixyz.append([])
            for j in range(self.natoms):
                d,r,imgi=self.get_distvec(i,j)
                ixyz[i].append(imgi)
                dxyz[i,j] = d
                vxyz[i,j,:] = r
                #vxyz[j,i,:] = -r
        centers = []
        for i,c in enumerate(self.centers):
            for j,cc in enumerate(c):
                centers.append(cc)
        centers =  numpy.array(centers)
        #print '---------centers---------'
        #print len(centers), numpy.dim(centers)
        #print centers
        #print ixyz[0]
        #print ixyz[4]
        #print ixyz[10]
        #print self.cellvec
        #dc = numpy.zeros((self.natoms,len(self.centers)))
        #vc = numpy.zeros((self.natoms,len(self.centers),3))
        ic = []
        #for i in xrange(self.natoms):
            #ic.append([])
            #for j in xrange(len(centers)):
                #d,v,imgi = self.get_distvec(i,centers[j])
                #dc[i,j]  = d
                #vc[i,j,:] = v
                #ic[i].append(imgi)
        #print '---------atom-center---------'
        #print numpy.array2string(dc,precision=4)
        #print numpy.array2string(v,precision=4)
        #print numpy.array2string(dc,precision=4)
        for i in range(self.natoms):
            dsort = numpy.argsort(dxyz[:,i])#[1:]
            #print dsort
            target = self.node_coordination[self.kinds[i]]
            mxlen = numpy.min([subspace_size*target, len(dsort)])
            #print i,target,dsort, dxyz[i,dsort[0:6]]
            #print ixyz[i][dsort[0]],ixyz[i][dsort[1]],ixyz[i][dsort[2]],ixyz[i][dsort[3]],ixyz[i][dsort[4]]
            for j,s in enumerate(dsort[0:mxlen]):
                p1 = self.xyz[i]#
                #print '-----',j,'-----', p1
                for ii in ixyz[i][s]:
                    p2 = self.get_image(self.xyz[s],ii)
                    l1 = geo.line(p1,p2)
                    p3 = p1 + 0.5 * l1.v
                    ###if additional_cutoff != None:
                        ###if additional_cutoff > dxyz[i,dsort]
                    #print '--------------------',i,s,'--------------------'
                    #print dxyz[s,i], l1.normv,numpy.array2string(self.xyz[j],precision=3),images[ii], numpy.array2string(p2,precision=3)
                    #print self.get_distvec(p1,p3)[0], self.get_distvec(p2,p3)[0]
                    for k in range(len(centers)):
                        #print k, centers[k]
                        d,v,imgi = self.get_distvec(centers[k],p3,exclude_self=False)
                        #if d < 0.5:
                            #print d, numpy.array2string(v,precision=3)
                        if d < fixed_cutoff:
                            #print 'AAAAAAAAAAAAAA', k,d,i,s
                            if self.conn[i].count(s) == 0:
                                self.conn[i].append(s)
                                self.pconn[i].append(images[ii])
                                self.conn[s].append(i)
                                self.pconn[s].append(images[ii]*-1)
                            else:
                                done=False
                                nn = self.conn[i].count(s)
                                idx=-1
                                for ll in range(nn):
                                    idx = self.conn[i][idx+1:].index(s)+idx +1
                                    comp = self.pconn[i][idx] == images[ii]
                                    #print i,s,nn,idx,ll, comp,self.pconn[i][idx], images[ii]
                                    if comp.all():
                                        done=True
                                if not done:
                                    self.conn[i].append(s)
                                    self.pconn[i].append(images[ii])
                                    self.conn[s].append(i)
                                    self.pconn[s].append(images[ii]*-1)

    def get_conn3(self,fixed_cutoff = 0.01): #  funktioniert nichtmal bei srs
        self.conn = []
        self.pconn = []
        self.use_pconn=True
        for i in range(self.natoms):
            self.conn.append([])
            if self.use_pconn: self.pconn.append([])
        dxyz =  numpy.zeros((self.natoms,self.natoms))
        vxyz =  numpy.zeros((self.natoms,self.natoms,3))
        ixyz = []
        for i in range(self.natoms):
            ixyz.append([])
            for j in range(self.natoms):
                d,r,imgi=self.get_distvec(i,j)
                ixyz[i].append(imgi)
                dxyz[i,j] = d
                vxyz[i,j,:] = r
                #vxyz[j,i,:] = -r
        if numpy.max(abs(dxyz -dxyz.T)) > 0.01:
            print('distance matrix broken!!!!! ------------------------------------------------------')
        ic = []
        for i in range(self.natoms):
            dsort = numpy.argsort(dxyz[:,i])[1:]
            target = self.node_coordination[self.kinds[i]]
            mxiter = 2*target
            if 2*target > len(dsort):
                mxiter = len(dsort)
            #print i,target,dsort, dxyz[i,dsort[0:6]]
            #print ixyz[i][dsort[0]],ixyz[i][dsort[1]],ixyz[i][dsort[2]],ixyz[i][dsort[3]],ixyz[i][dsort[4]]
            sames = numpy.zeros((mxiter,mxiter),dtype='int')
            ddd = numpy.zeros((mxiter,mxiter))#,dtype='float64')
            close=True 
            for j in range(mxiter):
                for k in range(mxiter):
                    ddd[j,k] = abs(dxyz[i,dsort[j]]-dxyz[i,dsort[k]])
                    if j != k:
                        if abs(dxyz[i,dsort[j]]-dxyz[i,dsort[k]]) < fixed_cutoff:
                            sames[j,k] += 1
            #print target,'sames', i
            #print sames
            #print numpy.array2string(ddd,precision=3,formatter={'float_kind':lambda x: "%.2f" % x})
            #print dxyz[i,dsort[0:mxiter]]
            cc = []
            for j in range(mxiter):
                ss =sum(sames[j,:])
                if ss == target-1:
                    cc.append(j)
            if len(cc) == target:
                for j in range(target):
                    aa =dsort[cc[j]]
                    d,v,imgi = self.get_distvec(i,aa)
                    s1=False
                    try:
                        self.conn[i].index(aa)
                    except ValueError:
                        s1=True
                        s2 = False
                        for ii in imgi:
                            try:
                                self.pconn[i].index(ii)
                            except ValueError:
                                s2 = True
                    if s1 and s2:    
                        self.conn[i].append(aa)
                        self.conn[aa].append(i)
                        if self.use_pconn:
                            image = images[ii]
                            self.pconn[i].append(image)
                            self.pconn[aa].append(image*-1)
                
        return  

    def get_conn_centers2(self,subspace=12):
        #i) every center must result in exactly one bond
        pass
        return

    def get_conn_centers(self,fixed_cutoff = 0.005,dc_subspace=6): 
        dc_subspace = numpy.min([self.natoms,dc_subspace])
        self.conn = []
        self.pconn = []
        for i in range(self.natoms):
            self.conn.append([])
            if self.use_pconn: self.pconn.append([])
        centers=[]
        for i,c in enumerate(self.centers):
            for j,cc in enumerate(c):
                centers.append(cc)
        centers =  numpy.array(centers)
        n=len(centers)
        dnc =numpy.zeros((self.natoms,n))
        dvec =numpy.zeros((self.natoms,n,3))
        iimgi = []
        for i in range(self.natoms):
            iimgi.append([])
            for j in range(n):
                dnc[i,j],dvec[i,j,:],imgi = self.get_distvec(i,centers[j])
                iimgi[i].append(imgi)
            #print i, numpy.array2string(dnc[i,:],precision=3)
        for j in range(n):
            #print j,numpy.array2string(dnc[:,j],precision=3)
            sort = numpy.argsort(dnc[:,j])[0:dc_subspace]
            #print sort
            #print j,numpy.array2string(dnc[sort,j],precision=3)
            run = True
            mm,nn = 0,0
            for m in range(dc_subspace):
                for o in range(m+1,dc_subspace):
                    if (abs(dnc[sort[m],j]-dnc[sort[o],j]) <= fixed_cutoff) and run:
                        #print m,o,'|', sort[m], sort[o]
                        mm,nn = copy.copy(m),copy.copy(o)
                        run=False
            #print 'found', mm,nn, '(which hopefully are not 0,0, otherwise its a bug!)'
            #print 'atoms', sort[mm], 'and', sort[nn], 'have the same connector' 
            for ii in iimgi[sort[mm]][sort[nn]]:
                self.conn[sort[mm]].append(sort[nn])
                self.conn[sort[nn]].append(sort[mm])
                self.pconn[sort[mm]].append(images[ii])
                self.pconn[sort[nn]].append(images[ii]*-1)

        return
    
#### Sanity Checks

    def check_quality(self):
        ncenters = 0
        for i,c in enumerate(self.centers):
            ncenters += len(c)
        nbonds = 0
        try:
            for i, kinds in enumerate(self.kinds):
                nbonds += self.node_coordination[kinds]
        except AttributeError:
            print('kinds not there')
            self.good=False
            return False
        if nbonds != ncenters * 2:
            self.good = False
            print('number of bonds not equal to number of centers *2 !!!')
            return False
        return True

    def check_conn(self,layer='none'):
        #self.conn_quality = 'none'
        broken = []
        for i in range(self.natoms):
            lenc = self.node_coordination[self.kinds[i]]
            if lenc != len(self.conn[i]):
                broken.append(i)
                self.conn_quality = ['broken',layer]
                #print self.name, 'has broken atom', i, lenc, len(self.conn[i])
        return broken
    
#### Coordinate Calcualations & Transformations
        
    def get_distvec(self, i, j,exclude_self=True):
        """ vector from i to j
        This is a tricky bit, because it is needed also for distance detection in the blueprint
        where there can be small cell params wrt to the vertex distances.
        In other words: i can be bonded to j multiple times (each in a different image)
        and i and j could be the same!! """
        leni = True
        lenj = True
        try:
            l=len(i)
            if l > 1:
                ri = numpy.array(i)
            else:
                leni = False
                ri = self.xyz[i]
        except:
            ri = self.xyz[i]
        try:
            l=len(j)
            if l > 1:
                rj = numpy.array(j)
            else:
                rj = self.xyz[j]
        except:
            rj = self.xyz[j]
            lenj = False 
        if 1:
            all_rj = rj + self.images_cellvec
            all_r = all_rj - ri
            all_d = numpy.sqrt(numpy.add.reduce(all_r*all_r,1))
            d_sort = numpy.argsort(all_d)
            #if (leni==False) and (lenj == False):
                #if i == j: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    #print 'same atom', i,j
                    #print d_sort
                    #d_sort = d_sort[1:]
                    #print d_sort
            if (numpy.linalg.norm(ri-rj) <= 0.001) and exclude_self:
                #print 'same atom', i,j
                #print d_sort
                d_sort = d_sort[1:]
                #print d_sort
                
                # if this was requested for i==j then we have to eliminate the shortest 
                # distance which NOTE unfinished!!!!!!!!
                #pass 
            closest = d_sort[0]
            closest=[closest]  # THIS IS A BIT OF A HACK BUT WE MAKE IT ALWAYS A LIST ....
            if (abs(all_d[closest[0]]-all_d[d_sort[1]]) < SMALL_DIST):
                # oops ... there is more then one image atom in the same distance
                #  this means the distance is larger then half the cell width
                # in this case we have to return a list of distances
                for k in d_sort[1:]:
                    if (abs(all_d[d_sort[0]]-all_d[k]) < SMALL_DIST):
                        closest.append(k)
            d = all_d[closest[0]]
            r = all_r[closest[0]]
        #else:
            #if i == j: return
            #r = rj-ri
            #d = numpy.sqrt(numpy.sum(r*r))
            #closest=[0]
        return d, r, closest

            
    def make_supercell(self, supercell=[1,1,1]):
        supercell = numpy.array(supercell)
        cellfact = supercell.prod()
        new_xyz = []
        for a in range(supercell[0]):
            for b in range(supercell[1]):
                for c in range(supercell[2]):
                    dispvect = numpy.sum(self.cellvec*numpy.array([a,b,c])[:,numpy.newaxis],axis=0)
                    new_xyz += (self.xyz+dispvect).tolist()
        self.xyz = numpy.array(new_xyz, "d") 
        self.n *= cellfact
        return
    
    
    def make_333(self, supercell=[3,3,3],out=False):
        supercell = numpy.array(supercell)
        cellfact = supercell.prod()
        new_xyz = []
        for a in range(supercell[0]):
            for b in range(supercell[1]):
                for c in range(supercell[2]):
                    dispvect = numpy.sum(self.cellvec*numpy.array([a-1,b-1,c-1])[:,numpy.newaxis],axis=0)
                    new_xyz += (self.xyz+dispvect).tolist()
        if out:
            return new_xyz
        self.xyz = numpy.array(new_xyz, "d") 
        self.n *= cellfact
        return
        
    def make_333_temp(self, supercell=[3,3,3]):
        supercell = numpy.array(supercell)
        cellfact = supercell.prod()
        new_xyz = []
        new_index = []
        for a in range(supercell[0]):
            for b in range(supercell[1]):
                for c in range(supercell[2]):
                    dispvect = numpy.sum(self.cellvec*numpy.array([a-1,b-1,c-1])[:,numpy.newaxis],axis=0)
                    new_xyz += (self.xyz+dispvect).tolist()
                    new_index.append(list(range(self.n)))
        #self.xyz = numpy.array(new_xyz, "d") 
        #self.n *= cellfact
        return numpy.array(new_xyz, "d"), new_index
    
    def get_image(self,xyz, img):
        xyz = numpy.array(xyz)
        try:
            l = len(img)
            dispvec = numpy.sum(self.cellvec*numpy.array(img)[:,numpy.newaxis],axis=0)
        except TypeError:
            dispvec = numpy.sum(self.cellvec*numpy.array(images[img])[:,numpy.newaxis],axis=0)
        return xyz + dispvec
    
    #def get_image2(self,xyz, img):
        #xyz = numpy.array(xyz)
        #try:
            #l = len(img)
            #numpy.sum(self.cellvec*numpy.array(img)[:,numpy.newaxis],axis=0)
        #except TypeError:
            #dispvec = numpy.sum(self.cellvec*numpy.array(images[img])[:,numpy.newaxis],axis=0)
        #return xyz + dispvec 
    
    def get_distmat(self):
        self.distmat = numpy.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(i,self.n):
                self.distmat[i,j] = self.get_distvec(i,j)[0]
        print(numpy.array2string(self.distmat+self.distmat.T, precision = 4))
        return
    
    def make_supercell_centers(self, supercell,shift=False):
        supercell = numpy.array(supercell)
        cellfact = supercell.prod()
        for i in range(len(self.centers)):
            new_centers = []
            for a in range(supercell[0]):
                for b in range(supercell[1]):
                    for c in range(supercell[2]):
                        if shift:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a-1,b-1,c-1])[:,numpy.newaxis],axis=0)
                        else:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a,b,c])[:,numpy.newaxis],axis=0)
                        new_centers += (self.centers[i]+dispvect).tolist()
            self.centers[i] = numpy.array(new_centers, "d")
        for i in range(len(self.edges1)):
            new_centers = []
            for a in range(supercell[0]):
                for b in range(supercell[1]):
                    for c in range(supercell[2]):
                        if shift:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a-1,b-1,c-1])[:,numpy.newaxis],axis=0)
                        else:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a,b,c])[:,numpy.newaxis],axis=0)
                        new_centers += (self.edges1[i]+dispvect).tolist()
            self.edges1[i] = numpy.array(new_centers, "d")
        for i in range(len(self.edges2)):
            new_centers = []
            for a in range(supercell[0]):
                for b in range(supercell[1]):
                    for c in range(supercell[2]):
                        if shift:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a-1,b-1,c-1])[:,numpy.newaxis],axis=0)
                        else:
                            dispvect = numpy.sum(self.cellvec*numpy.array([a,b,c])[:,numpy.newaxis],axis=0)
                        new_centers += (self.edges2[i]+dispvect).tolist()
            self.edges2[i] = numpy.array(new_centers, "d")
            #self.cell *= supercell[:,num.newaxis]
            #self.cellparams[0:3] *= supercell
            #self.natoms*=cellfact
            #self.elems *=cedllfact
            #self.atypes*=cellfact
            #self.images_cellvec = num.dot(images, self.cell)
            return    

#### Plotting
            
    def plot(self,scell=False,bonds=False,labels=True):
        col = ['r','g','b','m','c','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k']+['k']*200
        fig = plt.figure(figsize=plt.figaspect(1.0)*1.5)
        ax = fig.add_subplot(111, projection='3d')
        if bonds:
            for i in range(self.natoms):
                conn = self.conn[i]
                for j in range(len(conn)):
                    if self.pconn:
                        if numpy.sum(numpy.abs(self.pconn[i][j])) == 0:
                            ax.plot([self.xyz[i][0],self.xyz[conn[j]][0]],[self.xyz[i][1],self.xyz[conn[j]][1]],[self.xyz[i][2],self.xyz[conn[j]][2]])
                        else:
                            xyznew = self.get_image(self.xyz[conn[j]],self.pconn[i][j])
                            ax.scatter(xyznew[0],xyznew[1],xyznew[2],color='orange')
                            ax.plot([self.xyz[i][0],xyznew[0]],[self.xyz[i][1],xyznew[1]],[self.xyz[i][2],xyznew[2]])
                    else:
                        ax.plot([self.xyz[i][0],self.xyz[conn[j]][0]],[self.xyz[i][1],self.xyz[conn[j]][1]],[self.xyz[i][2],self.xyz[conn[j]][2]])
                        
        #xyz1=numpy.array(self.xyz)
        if labels:
            for i in range(self.natoms):
                label = str(i)+'-'+str(self.kinds[i])+'-'+str(self.node_coordination[self.kinds[i]])
                ax.text(self.xyz[i][0], self.xyz[i][1], self.xyz[i][2]+0.005, label, color='k',fontsize=7)
        if scell:
            xyz3 = self.make_333(out=True) 
            xyz3 =  numpy.array(xyz3)
            ax.scatter(xyz3[:,0],xyz3[:,1],xyz3[:,2],color='r',alpha=0.5)
        xyz=numpy.array(self.xyz)
        cc=numpy.array(self.centers)
        for i in range(len(cc)):
            print('center', i, 'magnitude:', len(cc[i]))
            c=cc[i]
            ax.scatter(c[:,0],c[:,1],c[:,2],color=col[i],alpha=0.5,marker='^')
            if labels:
                for j in range(len(c)):
                    label = str(i)+'-'+str(j)
                    ax.text(c[j][0], c[j][1], c[j][2]+0.005, label, color='orange',fontsize=7)
        #e1=numpy.array(self.edges1)[0]
        #e2=numpy.array(self.edges2)[0]
        #ax.scatter(e1[:,0],e1[:,1],e1[:,2],color='b')
        #ax.scatter(e2[:,0],e2[:,1],e2[:,2],color='k')
        #ax.scatter(xyz1[:,0],xyz1[:,1],xyz1[:,2],color='k')
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],color='k')
        minbound = numpy.min([numpy.min(xyz[:,0]),numpy.min(xyz[:,1]),numpy.min(xyz[:,2])])
        maxbound = numpy.max([numpy.max(xyz[:,0]),numpy.max(xyz[:,1]),numpy.max(xyz[:,2])])
        ax.auto_scale_xyz([0.0, maxbound], [0.0, maxbound], [0.0, maxbound])
        #ax.scatter(xyz1[:,0],xyz1[:,1],xyz1[:,2],color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        
# end of nets class


class netbase(object):
    
    def __init__(self,net_list=False,print_status = 4,homepath='/home/julian/sandbox/net_database/'):
        self.rundir=os.getcwd()
        self.homepath = homepath
        self.rcsrpath = homepath+'./RCSR/'
        self.database=False
        numpy.set_printoptions(suppress=True,precision=4)
        if net_list:
            self.nets = net_list
        else: 
            #self.load_nets()
            self.nets=[]
            pass
        self.nnets = len(self.nets)
        self.print_status = print_status
        self.workdir = os.getcwd()
        self.vertex_models = []
        self.bb = []
        self.bbs = {}
        self.vertices_with_conn = []         # contains the indices of vertex with conn i for self.vertex_models
        self.load_vertex_models() 
        self.api=None
        return
    
    def load_nets(self,path=None):
        self.nets = pickle.load(open(self.homepath+'data/goodall.pickle','rb'))
        self.bad  = pickle.load(open(self.homepath+'data/badall.pickle', 'rb'))
        self.netd = {}
        for i in self.nets:
            self.netd.update({i.name:i})
        return
    
    def make_net_dict(self):
        self.net={}
        self.neti={}
        for i,n in enumerate(self.nets):
            self.net.update({n.name:n})
            self.neti.update({n.name:i})
    
    #### Read-Write database files, 3dall.txt & systre .cgd in RCSR/ 
    
    def read_3dall(self,path=None,replace=False):
        self.tdall = []
        if not path: path = self.rcsrpath+'3dall.txt'
        txt = open(path,'r').read().split('start')[1:-1] #yields raw text including '\n'
        print(str(len(txt))+' nets in '+path.split('/')[-1])
        for i in txt:
            n = nets.net()
            n.parse_3dall(i)
            n.get_spacegroup()
            n.get_xyz()
            n.get_edges()
            n.scale()
            self.tdall.append(n)
        #n = nets
        return txt
    
    def make_conn(self,level=1,use='self.tdall',ssize=3):
        good,bad=[],[]
        g=0
        exec('netstotry = '+use)
        for i,n in enumerate(netstotry):
            if level==1:
                n.get_conn()
                if n.check_conn(): #true means wrong connectivity
                    bad.append(n)
                else:
                    good.append(n)
                    n.conn_level = level
            if level==2:
                n.get_conn2(subspace_size=ssize)
                if n.check_conn(): #true means wrong connectivity
                    bad.append(n)
                else:
                    good.append(n)
                    n.conn_level = level  
            if level==3:
                n.get_conn_centers()
                if n.check_conn(): #true means wrong connectivity
                    bad.append(n)
                else:
                    good.append(n)
                    n.conn_level = level             
            if level==4:
                n.get_conn3()
                if n.check_conn(): #true means wrong connectivity
                    bad.append(n)
                else:
                    good.append(n)
                    n.conn_level = level 
            if n.conn_level != -1: g += 1     
            if i % 20 == 0: print(g, '/', i, 'level: '+str(level))
        return good, bad
    
    def make_allconn(self):
        self.good=[]
        self.bad=[]
        self.good2=[]
        self.bad2 =  []
        self.good3 = []
        self.bad3=[]
        self.good4=[]
        self.bad4=[]
        for i,n in enumerate(self.tdall):
            self.good,self.bad=self.make_conn(level=1,use='self.tdall')
            self.good2,self.bad2=self.make_conn(level=2,use='self.bad')
            for j, nn in enumerate(self.bad2):
                nn.clean()
                nn.get_spacegroup(sg_setting=1)
                nn.conn=[]
            self.good3,self.bad3=self.make_conn(level=1,use='self.bad2')
            self.good4,self.bad4=self.make_conn(level=2,use='self.bad3')
            
    
    def read_systre(self,path=None):
        if not path: path = self.rcsrpath+'RCSRnets.cgd'
        return
    
    def check_updates(self):
        pass
        return
    
    ####
        
    def p(self,pstr,stat):
        pp = ''
        if self.print_status >= stat:
            for i in range(len(pstr)):
                pp += str(pstr[i])+' '
            print(pp)
        return
    
    def add_nets(self,net_list):
        for j in range(self.nnets):
            for i in range(len(net_list)):
                if self.nets[j].name !=net_list[i].name :
                    self.nets += net_list[i]
        self.nnets = len(self.nets)
        
    def weaver_instance(self,neti):
        if type(neti) == type(1):
            n = self.nets[neti]
        else:
            n = neti
        f=weaver.framework(n.name,use_new=True)
        f.pass_net_class(n)
        
        return f
    
    def load_penalties(self,fname='data/penalties.pickle'):
        self.penalties = pickle.load(open(fname,'rb'))
        pass
    
    def save_penalties(self,fname='data/penalties.pickle'):
        pickle.dump(self.penalties,open(fname,'wb'))
        return
    
    def format_penalties(self,pens=None):
        # for i,k in enumerate(pen[0][1]):
        
        #print i, k[1], base.nets[0].kinds[k[1]]
        self.errors = []
        self.penf   = {}
        self.vnames = []
        if not pens: pens = self.penalties
        for i,pp in enumerate(pens):
            penn = {}
            if pp[0] != self.nets[i].name:
                print(pp[0], self.nets[i].name)
                print('ERROR, there is something wrong with the enumeration scheme!')
                return 
            num = -1
            if pp[1][0] == False:
                print(i, pp[1][1])
                self.errors.append([i,pp[1][1]])
                continue
            print(i,pp)
            for j,P in enumerate(pp[1].keys()):
                #if num == -1: num = p[1]
                p=pp[1][P]
                print(i,j,p)
                # now every entry of p is the set of penalty evaluations for one vertex
                try:
                    v = p[0][1]# alt: v=P
                    # we get the vertex models the respective vertex is screened for
                    # upon conversion of the order so we do not have to save the names once more.
                    #models =  self.vertices_with_conn[p[0][3]]
                    penj  = [[o[5],o[4]] for o in p]
                    penn.update({j:penj})
                except: 
                    penn.update({j:False})
                    self.errors.append(j)
            self.penf.update({pp[0]: penn})
                #for k, o in enumerate(p):
        print('no error with enumeration scheme, self.penf[i] belongs to net[i], hf')
        pass
    
            
    def get_net(self,name=None):
        if name:
            for i,n in enumerate(self.nets):
                if n.name == name:
                    print('net number', i, n.name)
                    return n
        return None
    
    def screen_all_nets(self):
        self.penalties = []
        for i in range(self.nnets):
        #for i in range(11):
            try:
                print('  -----  starting net', i,self.nets[i].name, '     -----     ')
                f =self.weaver_instance(i)
                self.penalties.append([self.nets[i].name,f.scan_bb_models(self)])
            except KeyboardInterrupt:
                return
            except:
                print('-------------------------NET', i, self.nets[i].name, 'failed----------------------------------')
                exc = sys.exc_info()
                print(exc)
                self.penalties.append([self.nets[i].name,[False, exc[0:-1]]])
        return
    
    def screen_nets(self,nets):
        self.penalties = []
        for i,n in enumerate(nets):
        #for i in range(11):
            try:
                print('  -----  starting net', i,n.name, '     -----     ')
                f =self.weaver_instance(n)
                self.penalties.update({n.name:f.scan_bb_models(self)})
            except KeyboardInterrupt:
                return
            except:
                print('-------------------------NET', i, n.name, 'failed----------------------------------')
                exc = sys.exc_info()
                print(exc)
                self.penalties.append([n.name,[False, exc[0:-1]]])
        return
    
    def load_bbs(self,bbpath= './bbs/'):
        #os.chdir()
        ldir = os.listdir(bbpath)
        for i,fname in enumerate(ldir):
            fnsplit = fname.rsplit('.',1)
            if fnsplit[-1] == 'mfpx':
                m = molsys.mol()
                m.addon('bb')
                m.read(bbpath+fname)
                m.bb.setup(fnsplit[0])
                self.bbs.update({fnsplit[0]:m})
    
    def test_bbs(self):
        if len(self.bbs) == 0:
            self.load_bbs()
        for i,bbk in enumerate(self.bbs.keys()):
            bb = self.bbs[bbk]
            self.bbs[bbk].penalties = self.test_bb(bb=bb)
            
    
    def add_bb(self,bb_fname,specific_conn=None, linker=False, zflip=False, nrot=1,ftype='mfpx'):
        m = molsys.mol()
        m.addon('bb')
        m.read(bb_fname,ftype=ftype)
        m.bb.setup(bb_fname,specific_conn=specific_conn,linker=linker, zflip=zflip, nrot=nrot)
        self.bb.append(m)
        return
    
    def test_bb(self,idx=0,bb=None):
        if bb == None:
            bb = self.bb[idx]
        nconn = len(bb.connectors)
        vc = self.vertices_with_conn[nconn]
        bbstats = {}
        for j, v1 in enumerate(vc):
             s1,s2 = bb,self.vertex_models[v1].bb
             rot = rotator(copy.copy(s1.connector_xyz),copy.copy(s2.connector_xyz))
             penalty,_,_ = rot.screen_orientations(50,51,0.1)
             bbstats.update({self.vertex_models[v1].name:penalty})
        print(bbstats)
        return bbstats
             
        
    def crosstest_vertex_models(self):
        self.vertex_corr = []
        for i in range(0,40):
            vc = self.vertices_with_conn[i]
            self.vertex_corr.append(numpy.zeros((len(vc),len(vc))))
            for j, v1 in enumerate(vc):
                for k, v2 in enumerate(vc[j:]):
                    #print i,j,k,v1,v2,penalty
                    s1,s2 = self.vertex_models[v1].bb,self.vertex_models[v2].bb
                    rot = rotator(copy.copy(s1.connector_xyz),copy.copy(s2.connector_xyz))
                    penalty,_,_ = rot.screen_orientations(75,76,0.1)
                    print(i,j,k,v1,v2,s1.name,s2.name,penalty)
                    self.vertex_corr[i][j,j+k] = penalty
                    self.vertex_corr[i][j+k,j] = penalty
            if len(self.vertex_corr[i] != 0):
                print([self.vertex_models[st].name for st in self.vertices_with_conn[i]])
                print(self.vertex_corr[i])
                
    def debug(self):
        #okay, 3-tshaped and 4-squareplanar does not work properly, why?
        # the files are checked and just fine
        for i,v in enumerate(self.vertex_models):
            if v.name == 'tshaped': tsh = v
            if v.name == 'squareplanar': sq = v
        #rot = rotator(tsh.bb.connector_xyz())
        
        
        return tsh,sq
        


    def load_vertex_models(self,foldername='vertex_geometries'):
        #cwd = os.getcwd()
        cwd = self.homepath
        os.chdir(cwd+foldername)
        li = os.listdir(os.getcwd())
        for i,l in enumerate(li):
            ls = l.rsplit('.',1)
            if len(ls) != 2:
                self.p([l,'is not a proper file'],3)
                continue
            if ls[-1] == 'mfpx':
                self.p([ls[0], 'will be used as vertex model'], 2)
                self.vertex_models.append(vertex_model(l))
            else:
                self.p([l, 'is not recognised as a proper xyz file'], 4)
        #self.p(self.vertex_models,4)    
        dicti = []
        for i in range(99):
            dicti.append([i,[]])
        #print dicti
        for i in range(len(self.vertex_models)):
            nc = self.vertex_models[i].nconn
            dicti[nc][1].append(i)
        self.vertices_with_conn = dict(dicti)
        #print self.vertices_with_conn
        #print self.vertices_with_conn[3]
        #print self.vertices_with_conn[4]
        os.chdir('../')    
        return
    
    def get_unique_neighbors(self,fname ='vertex_conn_stats.pickle'): 
        dct = {}
        for i,n in enumerate(self.nets):
            dct.update({n.name:n.get_unique_neighbors()})
        pickle.dump(dct,open(fname,'wb'))
        return
    
    def built_vertex_dict(self,fname='vertex_dict.pickle'):
        ''' before, get_unique_neighbors has to be called --> builds necessary data! '''
        dct = {}
        for i,n in enumerate(self.nets):
            dct.update({n.name:n.make_vertex_db_entry()})
        pickle.dump(dct,open(fname,'wb'))
        return
    
    ####### API stuff, as much as is needed atm.
    
    def connect_api(self):
        if self.api == None:
            self.api = weaver.mofplus_api()
    
    def upload_bb_penalties(self,bb):
        self.connect_api()
        print('MAKE SURE THE BB DATA IS THERE!!!!')
        print('----------_____ %s _____----------' % bb)
        data = {'name':bb}
        data['penalties'] = self.bbs[bb].penalties
        
        self.api.add_bb_penalties(data)
        return data
        pass
    

class vertex_model(object):
    def __init__(self,fname):
        self.xyz = []
        #print fname
        self.fname = fname
        self.name = fname.split('-')[1].rsplit('.',1)[0]
        self.nconn = int(fname.split('-')[0])
        self.bb = []
        print(fname, self.name, self.nconn, self.fname)
        self.assign_bb(self.fname)
        
    def __str__(self):
        return str(self.name)+' '+str(self.nconn)
            
        
    def assign_bb(self, bb_fname,specific_conn=None, linker=False, zflip=False, nrot=1):
        """if specific_conn is given, then it is a list of vertex types of the length
           of the connector types. The first connector binds only to the first vertex
           type etc """
        m = molsys.mol()
        m.addon('bb')
        m.read(bb_fname)
        self.bb = m
        self.bb.bb.setup(bb_fname,specific_conn=specific_conn,\
                  linker=linker, zflip=zflip, nrot=nrot)





class rcsr(object):
    ''' 
    very old, we keep it only for legacy reasons
    '''

    def __init__(self):
        self._nets = {}
        return
    
    def read_arc(self,fname):
        f = open(fname, 'r')
        for line in f:
            sline = line.split()
            if len(sline)>0:
                if sline[0] == 'key':
                    dim = int(sline[1])
                    key = string.join(sline[2:])
                if sline[0] == 'id':
                    name = sline[1]
                    if name.count('*') > 0:
                        name = name.replace('*','s')
        if name in self._nets: 
            self._nets[name]['dim'] = dim
            self._nets[name]['key'] = key
        else:
            self._nets[name] = {'dim':dim, 'key':key}
        return

    def write_arc(self,fname):
        pass

    def read_cgd(self,fname):
        entries = open(fname, 'r').read().split('CRYSTAL')[1:]
        #entries = open(fname, 'r').read().split('CRYSTAL')[1:-1]
        #entries = open(fname, 'r').read()
        for i, e in enumerate(entries):
            self.parse_cgd(e)
        pass

    def parse_cgd(self, entry):
        dic = {}
        nodes = []
        edges = []
        lines = entry.split("\n")[1:]
        for i,l in enumerate(lines):
            sline = l.split()
            if sline[0] == 'NAME':
                name = sline[1]
                if name.count('*') > 0:
                    name = name.replace('*','s')
                dic['NAME'] = name
            elif sline[0] == 'GROUP':
                dic['GROUP'] = sline[1]
            elif sline[0] == 'CELL':
                dic['CELL'] = sline[1:7]
            elif sline[0] == 'NODE':
                nodes.append(list(map(float,sline[3:])))
            elif sline[0] == 'EDGE':
                edges.append(list(map(float,sline[1:])))
            elif sline[0] == 'END':
                break
        dic['nodes'] = nodes
        dic['edges'] = edges
        if name in list(self._nets.keys()):
            self._nets[name]['cgd'] = dic
        else:
            self._nets[name] = {'cgd':dic}
        return
   
    def write_cgd(self, names, fname):
        def write_entry(f, name):
            try:
                cgd = self._nets[name]['cgd']
            except KeyError:
                return
            f.write('CRYSTAL\n')
            f.write('  NAME %s\n'  % name)
            f.write('  GROUP %s\n' % cgd['GROUP'])
            f.write('  CELL %s\n' % string.join(cgd['CELL']))
            for i in cgd['nodes']:
                f.write('  NODE %s\n' % string.join(i))
            for i in cgd['edges']:
                f.write('  EDGE %s\n' % string.join(i))
            f.write('END\n\n')
            return

        f = open(fname, 'w')
        for i,n in enumerate(names):
            write_entry(f, n)
        f.close()
        return

    def read_2dall(self, fname="2dall.txt"):
        txt = open(fname,'r').read().split('start')[1:-1]
        for i,t in enumerate(txt):
            self.parse_2dall(t)
        return

    def parse_2dall(self, txt):
        ndic = {}
#        print(txt)
        lines = txt.split('\n')[1:]
        # jump over line cotaining the id
        lines.pop(0)
        # get the netname
        name = lines.pop(0).split()[0]
        #if name in ['mtd-a', 'mtg-a']:
        #    return
        if name.count('*') > 0:
            name = name.replace('*','s')
            
        ndic['name'] = name
        #ndic['embed_type'] = lines.pop(0)

        nsymbols = int(lines.pop(0).split('!')[0])
        ndic['symbols'] = [lines.pop(0) for i in range(nsymbols)]

        nnames = int(lines.pop(0).split('!')[0])
        ndic['knownas'] = [lines.pop(0) for i in range(nnames)]
        nnames = int(lines.pop(0).split('!')[0])
        ndic['knownas'] += [lines.pop(0) for i in range(nnames)]
        nkeys = int(lines.pop(0).split('!')[0])
        ndic['keywords'] = [lines.pop(0) for i in range(nkeys)]
        # no reference in 2d
        #nrefs = int(lines.pop(0).split('!')[0])
        #ndic['refs'] = [lines.pop(0) for i in range(nrefs)]
        # no sg_number, new wallpaper group name
        ndic['wg_name'] = lines.pop(0).split()
        ndic['sg_name'] = lines.pop(0).split()
        ndic['cell'] = numpy.array(list(map(float,lines.pop(0).split())))
        # new in 2d
        nkinds = lines.pop(0).split()
        ndic['nverts'] = int(nkinds[0])
        ndic['nedges'] = int(nkinds[0])
        ndic['nfaces'] = int(nkinds[0])
        #self.make_cellvec()
        
        # not in 2d
        #nverts = int(lines.pop(0))
        nverts = ndic['nverts']
        
        ndic['symbolic']   = []
        # not in 2d
        #ndic['wyckoff']     = []
        #ndic['symmetry']    = []
        #ndic['order']       = []
        ndic['node']        = []
        ndic['node_coordination'] = []
        ndic['cs']          = []
        ndic['vs']          = []

        for i in range(nverts):
            # switched in 2d
            #ndic['node_coordination'].append(int(lines.pop(0).split()[-1]))
            #ndic['node'].append(map(float,lines.pop(0).split()))
            ndic['node_coordination'].append(lines.pop(0).split()[-1])
            try: # mtd-a and mtg-a are malformed
                ndic['node_coordination'][-1] = int(ndic['node_coordination'][-1])
            except ValueError:
                # assumption, same as the one before the last
                ndic['node_coordination'][-1] = int(ndic['node_coordination'][-2])
            ndic['node'].append(list(map(float,lines.pop(0).split())))
            ndic['symbolic'].append(lines.pop(0).split()[0])
            ndic['cs'].append(map(int, lines.pop(0).split())[:-1])
            ndic['vs'].append(lines.pop(0).split()[0])
            # not in 2d
            #ndic['wyckoff'].append(lines.pop(0).split()[0])
            #ndic['symmetry'].append(lines.pop(0).split()[0])
            #ndic['order'].append(int(lines.pop(0)))
    
        # not in 2d
        #nedges = int(lines.pop(0))
        nedges = ndic['nedges']
        
        # not in 2d
        #ndic['center_symbolic']   = []
        #ndic['center_wyckoff']    = []
        #ndic['center_symmetry']   = []
        #ndic['edge_center']       = []
        #for i in range(nedges):
        #    temp = lines.pop(0)
        #    ndic['edge_center'].append(map(float,lines.pop(0).split()))
        #    ndic['center_symbolic'].append(lines.pop(0))
        #    ndic['center_wyckoff'].append(lines.pop(0))
        #    ndic['center_symmetry'].append(lines.pop(0))
        # jump over the next 5 lines
        # read coord seqences and vertex symbols
        # not in 2d
        #for i in range(5): lines.pop(0)
        #for i in range(nverts):
        #    ndic['cs'].append(map(int, lines.pop(0).split())[:-1])
        #for i in range(nverts):
        #    ndic['vs'].append(lines.pop(0).split()[0])
        # put ndic into the overall _nets dictionaray
        if list(self._nets.keys()).count(name) == 0:
            self._nets[name] = ndic
        else:
            self._nets[name].update(ndic)
        return 

    def read_3dall(self, fname="3dall.txt"):
        txt = open(fname,'r').read().split('start')[1:-1]
        for i,t in enumerate(txt):
            self.parse_3dall(t)
        return

    def parse_3dall(self, txt):
        ndic = {}
#        print(txt)
        lines = txt.split('\n')[1:]
        # jump over line cotaining the id
        lines.pop(0)
        # get the netname
        name = lines.pop(0).split()[0]
        if name.count('*') > 0:
            name = name.replace('*','s')
        
        ndic['name'] = name
        ndic['embed_type'] = lines.pop(0)

        nsymbols = int(lines.pop(0).split('!')[0])
        ndic['symbols'] = [lines.pop(0) for i in range(nsymbols)]

        nnames = int(lines.pop(0).split('!')[0])
        ndic['knownas'] = [lines.pop(0) for i in range(nnames)]
        nnames = int(lines.pop(0).split('!')[0])
        ndic['knownas'] += [lines.pop(0) for i in range(nnames)]
        nkeys = int(lines.pop(0).split('!')[0])
        ndic['keywords'] = [lines.pop(0) for i in range(nkeys)]
        nrefs = int(lines.pop(0).split('!')[0])
        ndic['refs'] = [lines.pop(0) for i in range(nrefs)]
        t = lines.pop(0).split()
        ndic['sg_name'] = t[0]
        ndic['sg_number'] = t[1]
        ndic['cell'] = numpy.array(list(map(float,lines.pop(0).split())))
        #self.make_cellvec()
        
        nverts = int(lines.pop(0))
        
        ndic['symbolic']   = []
        ndic['wyckoff']     = []
        ndic['symmetry']    = []
        ndic['order']       = []
        ndic['node']        = []
        ndic['node_coordination'] = []
        ndic['cs']          = []
        ndic['vs']          = []

        for i in range(nverts):
            ndic['node_coordination'].append(int(lines.pop(0).split()[-1]))
            ndic['node'].append(list(map(float,lines.pop(0).split())))
            ndic['symbolic'].append(lines.pop(0).split()[0])
            ndic['wyckoff'].append(lines.pop(0).split()[0])
            ndic['symmetry'].append(lines.pop(0).split()[0])
            ndic['order'].append(int(lines.pop(0)))
    
        nedges = int(lines.pop(0))
        
        ndic['center_symbolic']   = []
        ndic['center_wyckoff']    = []
        ndic['center_symmetry']   = []
        ndic['edge_center']       = []
        for i in range(nedges):
            temp = lines.pop(0)
            ndic['edge_center'].append(list(map(float,lines.pop(0).split())))
            ndic['center_symbolic'].append(lines.pop(0))
            ndic['center_wyckoff'].append(lines.pop(0))
            ndic['center_symmetry'].append(lines.pop(0))
        # jump over the next 5 lines
        # read coord seqences and vertex symbols
        for i in range(5): lines.pop(0)
        for i in range(nverts):
            ndic['cs'].append(map(int, lines.pop(0).split())[:-1])
        for i in range(nverts):
            ndic['vs'].append(lines.pop(0).split()[0])
        # put ndic into the overall _nets dictionaray
        if list(self._nets.keys()).count(name) == 0:
            self._nets[name] = ndic
        else:
            self._nets[name].update(ndic)
        return 
