import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
#fig = plt.figure()
import numpy

def plot_cell(cell,show=True):
    fig = plt.figure(figsize=plt.figaspect(1.0)*1.5)
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    #ax = self.add_cell(ax)
    def axplt(ax,xx,yy):
        ax.plot([xx[0],yy[0]],[xx[1],yy[1]],[xx[2],yy[2]],color='black',linewidth=3)
    x,y,z=cell[:,0],cell[:,1],cell[:,2]
    zero = numpy.zeros(3)
    axplt(ax,zero,x)
    axplt(ax,zero,y)
    axplt(ax,zero,z)
    #axplt(ax,,)
    axplt(ax,x,x+y)
    axplt(ax,y,x+y)
    axplt(ax,z+x,x+y+z)
    axplt(ax,x+y,x+y+z)
    axplt(ax,z+y,x+y+z)
    axplt(ax,z,z+x)
    axplt(ax,z,z+y)
    axplt(ax,x,x+z)
    axplt(ax,y,y+z)     
    if show: plt.show()
    return

class plotter(object):
    
    def __init__(self,mol):
        self.mol = mol
        
        
        
    def add_cell(self,ax):
        mol = self.mol
        def axplt(ax,xx,yy):
            ax.plot([xx[0],yy[0]],[xx[1],yy[1]],[xx[2],yy[2]],color='black',linewidth=3)
            
        cell = mol.get_cell()
        x,y,z=cell[:,0],cell[:,1],cell[:,2]
        zero = numpy.zeros(3)
        axplt(ax,zero,x)
        axplt(ax,zero,y)
        axplt(ax,zero,z)
        #axplt(ax,,)
        axplt(ax,x,x+y)
        axplt(ax,y,x+y)
        axplt(ax,z+x,x+y+z)
        axplt(ax,x+y,x+y+z)
        axplt(ax,z+y,x+y+z)
        axplt(ax,z,z+x)
        axplt(ax,z,z+y)
        axplt(ax,x,x+z)
        axplt(ax,y,y+z)
        
        
        return ax
        
    def plot(self,scell=False,bonds=False,labels=False,skip=False):
        mol = self.mol
        col = ['r','g','b','m','c','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k']+['k']*200
        fig = plt.figure(figsize=plt.figaspect(1.0)*1.5)
        #ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        ax = self.add_cell(ax)
        atd = {}
        for i,aa in enumerate(list(set(mol.atypes))):
            atd.update({aa:col[i]})
        print(atd)
        if bonds:
            for i in range(mol.natoms):
                conn = mol.conn[i]
                for j in range(len(conn)):
                    try: 
                        mol.pconn
                    except:
                        mol.pconn = False
                    if mol.pconn:
                        if numpy.sum(numpy.abs(mol.pconn[i][j])) == 0:
                            ax.plot([mol.xyz[i][0],mol.xyz[conn[j]][0]],[mol.xyz[i][1],mol.xyz[conn[j]][1]],[mol.xyz[i][2],mol.xyz[conn[j]][2]],color='black')
                        else:
                            xyznew = mol.get_image(mol.xyz[conn[j]],mol.get_distvec([i][j])[2])
                            ax.scatter(xyznew[0],xyznew[1],xyznew[2],color='orange')
                            ax.plot([mol.xyz[i][0],xyznew[0]],[mol.xyz[i][1],xyznew[1]],[mol.xyz[i][2],xyznew[2]],color='green')
                    else:
                        ax.plot([mol.xyz[i][0],mol.xyz[conn[j]][0]],[mol.xyz[i][1],mol.xyz[conn[j]][1]],[mol.xyz[i][2],mol.xyz[conn[j]][2]],color=atd[mol.atypes[i]])

        if labels:
            for i in range(mol.natoms):
                label = str(i)+'-'+str(mol.atypes[i]) +'-'+str(len(mol.conn[i]))
                ax.text(mol.xyz[i][0], mol.xyz[i][1], mol.xyz[i][2]+0.005, label, color='k',fontsize=9)
        if scell:
            xyz3 = mol.make_333(out=True)
            xyz3 =  numpy.array(xyz3)
            ax.scatter(xyz3[:,0],xyz3[:,1],xyz3[:,2],color='r',alpha=0.5)
        xyz=numpy.array(mol.xyz)
        for i,xx in enumerate(xyz):
            if skip:
                if i % skip != 0:
                    continue
            ax.scatter(xx[0],xx[1],xx[2],marker='.',color=atd[mol.atypes[i]])
        minbound = numpy.min([numpy.min(xyz[:,0]),numpy.min(xyz[:,1]),numpy.min(xyz[:,2])])
        maxbound = numpy.max([numpy.max(xyz[:,0]),numpy.max(xyz[:,1]),numpy.max(xyz[:,2])])
        ax.auto_scale_xyz([0.0, maxbound], [0.0, maxbound], [0.0, maxbound])
        #ax.scatter(xyz1[:,0],xyz1[:,1],xyz1[:,2],color='k')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.show()
        
    def write_vmd_bondindices(self,filename='vmd.tcl',maxlength = 2.0,radius=0.2,color='Name'):
        m = self.mol
        from molsys.util import unit_cell
        m.set_ctab_from_conn()
        f = open(filename,'w')     
        cellparams = unit_cell.abc_from_vectors(m.get_cell())
        origin = numpy.min(m.xyz,axis=0)
        #import pdb; pdb.set_trace()
        #text = 'pbc set { %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f} -all\n' % tuple(cellparams)
        #text += 'pbc box -center origin -shiftcenter {%12.6f %12.6f %12.6f}\n' % tuple(origin)
        #text += 'pbc wrap\n'
        #text += 'pbc box_draw\n'
        #print(text)
        #f.write(text)
        for i,c in enumerate(m.ctab):
            text = 'mol color %s\nmol representation DynamicBonds %8.6f %8.6f 30.000000\n' % (color,maxlength,radius)
            text+= 'mol selection index %i %i\nmol material Opaque\nmol addrep 0\n' % (c[0],c[1])
            print(text)
            f.write(text)

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return
        
