import weaver


f = weaver.framework('hkust-1')
f.read_topo('tbo.mfpx')

f.assign_bb('0','btc.mfpx')
f.assign_bb('1','CuPW.mfpx')

f.scale_net(6.0)

f.scan_orientations(10)

f.generate_framework([0]*f.net.natoms)


f.framework.write('HKUST-1.mfpx',ftype='mfpx')
