import pylmps

"""
simple example to test expot

"""
# WARNING: This does NOT run within a function, nor does it run within pytest
#   the reason is the simple way in which the function for the callbackl is searched in the python globale namespace

pl = pylmps.pylmps("CuPW")
# instantiate the external potential expot_test for atoms #20 and #30, refdist = 4.0 and force const = 1.0
ep = pylmps.expot_test(20, 30, 4.0, 1.0)
# we need to register the objects callback as a global function
callback = ep.callback
# now add the expot object together with the name of the global callback
pl.add_external_potential(ep, "callback")
pl.setup(local=True, ff="file")
pl.MIN(0.01)
# TBI check for total energy
pl.MD_init("pyfix", T=300.0, startup=True)
pl.MD_run(5000)
# TBI check for energy conservation
pl.end()



