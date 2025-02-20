import weaver

# a weaver.framework instance has to be instanciated with a name
f = weaver.framework('HKUST-1')

# first ingredient: Topology. given as topo type mfpx file
f.read_topo('tbo.mfpx')

# second ingredient: building blocks, given as bb type mfpx files
f.assign_bb('0', 'btc.mfpx')
f.assign_bb('1', 'CuPW.mfpx')

# since nets usually come with edge length == 1, we need to scale 
f.autoscale_net(fiddle_factor=1.5) # c-c bond formed 
#f.scale_net(6.0) 

# iterative rotational optimization with 10 trials per vertex
f.scan_orientations(10)
f.write_orientations('HKUST-1.orients')

# construct MOF structure from topo+BBs+orientation
f.generate_framework() 
#f.generate_framework(autoscale=True) # uncomment for autoscaling

# write the MOF structure as mfpx file
f.framework.write('HKUST-1.mfpx')
# write the MOF structure as tinker xyz file
f.framework.write('HKUST-1.txyz')
