import pylmps
import molsys

ref_energies = {
            "vdW" :    125.6517552 ,
        "Coulomb" :  -7828.0856961 ,
        "CoulPBC" :  -8839.1068750 ,
           "bond" :     47.3757530 , 
          "angle" :    282.4041314 ,
            "oop" :      6.9431505 , 
        "torsion" :      7.4500266 ,
}
   


def test_opt():
    pl = pylmps.pylmps("hkust1")
    m = molsys.mol.from_file("hkust1.mfpx")
    m.addon("ff")
    m.ff.read("hkust1")
    pl.setup(mol=m, local=True)
    pl.MIN(0.1)
    energies = pl.get_energy_contribs()
    pl.end()

    for e in ref_energies.keys():
        assert abs(ref_energies[e]-energies[e])<1.0e-6

if __name__=="__main__":
    test_opt()

