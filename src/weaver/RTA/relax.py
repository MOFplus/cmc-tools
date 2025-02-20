import sys
import pydlpoly
import molsys
from molsys.util import ff2pydlpoly

# input staff
name = sys.argv[1]
resfile = sys.argv[2]
ff = sys.argv[3]
tresh_lbfgs = float(sys.argv[4])
tresh_lat   = float(sys.argv[5])
lat_iter    = int(sys.argv[6])

# setup and assignment
m = molsys.mol.fromFile(name+'.mfpx')
at = molsys.util.atomtyper(m)
at()
frag = molsys.util.fragmentizer()
frag(m)
m.addon("ff")
m.ff.assign_params(ff)
wm = ff2pydlpoly.wrapper(m)
pd = pydlpoly.pydlpoly(name)
pd.setup(web = wm, local = True, bcond = 3)

# optimization
pd.MIN_lbfgs(tresh_lbfgs)
pd.LATMIN_sd(tresh_lat, tresh_lbfgs,lat_maxiter= lat_iter)

# write output
m.xyz = pd.get_xyz()
m.set_cell(pd.get_cell())
m.write(name+"_final.mfpx")
e = pd.get_energy_contribs()
c = pd.get_cell()

e = e.tolist()
ne = len(e)

f = open(resfile, "w")
# get last element of e (etot to the front)
f.write(((ne+9)*"%12.8f "+"\n") % tuple([e[-1]]+e[:-1]+c.ravel().tolist()))
f.close()

pd.end()
