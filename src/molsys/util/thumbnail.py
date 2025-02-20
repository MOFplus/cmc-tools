"""
    thumbnail

    takes a mol object and generates a png file

    prerequisites: needs vmd and imagemagick "convert" to be installed

    please help to improve options
"""


import tempfile
import subprocess
import os

def thumbnail(mol, size=200, scale=1.3, transparent=True, fname=None, debug=False, own_bonds=False, with_sd=False, sdf="sd.plt"):
    """
    generate a thumbnail from a mol object
    by default a png is returned.

    sdf : the path to the spin density plt file

    we generate the tcl commands on the fly in order to change stuff there
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if fname is None:
            fname = os.getcwd() + "/" + mol.name + ".png"
        else:
            fname = os.getcwd() + "/" + fname
        # files
        xyzf = tmpdir+"/mol.xyz"
        tclf = tmpdir+"/vmd.tcl"
        tgaf = tmpdir+"/vmd.tga"
        # store structure as xyz
        mol.write(xyzf)
        # write vmd commands
        f = open(tclf, "w")
        vmd_settings = """
        color Display Background white
        color Name C black
        axes location Off
        mol modstyle 0 0 CPK 0.70000 0.300000 12.000000 12.000000
        scale by %f
        """ % scale

        # should we use the bonding from the mol object?
        if own_bonds:
            vmd_settings += "\n" + 'mol color %s\nmol representation DynamicBonds %8.6f %8.6f 30.000000\n' % ('Name',2.0,0.2)
            for i,c in enumerate(mol.ctab):
                vmd_settings += 'mol selection index %i %i\nmol material Opaque\nmol addrep 0\n' % (c[0],c[1])

        # to plot the spin density
        if with_sd:
            assert os.path.isfile(sdf), "The spin density plot file %s exists." %sdf
            vmd_settings += """mol addfile %s
        mol addrep 0
        mol modstyle 1 0 isosurface 0.005 0 0 0 1 1
        mol modmaterial 1 0 Transparent
        """ %(sdf)

        # add additional stuff here ... optional
        f.write(vmd_settings)
        f.write("render TachyonInternal %s\n" % tgaf)
        f.write("exit\n")
        f.close()
        # now run vmd
        vmdoutf = open(tmpdir+"/vmd.out", "w")
        subprocess.run(["vmd", "-xyz", xyzf, "-size", str(size), str(size), "-e", tclf, "-dispdev", "text"], stdout=vmdoutf)
        # now convert to png
        convert = ["convert"]
        if transparent:
            convert += ["-transparent", "white"]
        convert += [tgaf, fname]
        subprocess.run(convert)
        if debug==True:
            print ("In DEBUG mode")
            print ("go with another shell to tempdir %s to see intermediate files")
            input ("press ENTER to end an delete everything")
    return

