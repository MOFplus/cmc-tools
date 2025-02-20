# Example for C/H/O parameters
# The parameters were taken from the ReaxFF module in lammps:

#! at1; at2;  De(sigma);   De(pi);   De(pipi);   p(be1);    p(bo5);  13corr;   p(bo6),  p(ovun1);           p(be2);  p(bo3);  p(bo4);   n.u.;  p(bo1);  p(bo2)
#  1     1   156.5953    100.0397     80.0000     -0.8157  -0.4591   1.0000  37.7369   0.4235               0.4527   -0.1000   9.2605   1.0000  -0.0750   6.8316   1.0000   0.0000
#  1     2   170.2316      0.0000      0.0000     -0.5931   0.0000   1.0000   6.0000   0.7140               5.2267    1.0000   0.0000   1.0000  -0.0500   6.8315   0.0000   0.0000
#  2     2   156.0973      0.0000      0.0000     -0.1377   0.0000   1.0000   6.0000   0.8240               2.9907    1.0000   0.0000   1.0000  -0.0593   4.8358   0.0000   0.0000
#  1     3   160.4802    105.1693     23.3059     -0.3873  -0.1613   1.0000  10.8851   1.0000               0.5341   -0.3174   7.0303   1.0000  -0.1463   5.2913   0.0000   0.0000
#  3     3    60.1463    176.6202     51.1430     -0.2802  -0.1244   1.0000  29.6439   0.9114               0.2441   -0.1239   7.6487   1.0000  -0.1302   6.2919   1.0000   0.0000
#  2     3   180.4373      0.0000      0.0000     -0.8074   0.0000   1.0000   6.0000   0.5514               1.2490    1.0000   0.0000   1.0000  -0.0657   5.0451   0.0000   0.0000



# atomID; ro(sigma); Val;    atom mass; Rvdw;      Dij;    gamma;   ro(pi);  Val(e), alfa;    gamma(w);  Val(angle);  p(ovun5);  n.u.;    chiEEM;  etaEEM;    n.u.;   ro(pipi) ;p(lp2); Heat increment; p(boc4);  p(boc3); p(boc5),    n.u.;    n.u.;  p(ovun2); p(val3);  n.u.;   Val(boc); p(val5);n.u.;n.u.;n.u.
# C        1.3825   4.0000   12.0000    1.9133   0.1853   0.9000   1.1359   4.0000   9.7602   2.1346    4.0000        33.2433   79.5548   5.8678   7.0000   0.0000   1.2104    0.0000  199.0303         8.6991   34.7289   13.3894   0.8563   0.0000   -2.8983   2.5000   1.0564   4.0000   2.9663   0.0000   0.0000   0.0000
# H        0.7853   1.0000    1.0080    1.5904   0.0419   1.0206  -0.1000   1.0000   9.3557   5.0518    1.0000         0.0000  121.1250   5.3200   7.4366   1.0000   -0.1000   0.0000   62.4879         1.9771    3.3517    0.7571   1.0698   0.0000  -15.7683   2.1488   1.0338   1.0000   2.8793   0.0000   0.0000   0.0000
# O        1.2477   2.0000   15.9990    1.9236   0.0904   1.0503   1.0863   6.0000   10.2127   7.7719   4.0000        36.9573  116.0768   8.5000   8.9989   2.0000   0.9088    1.0003   60.8726         20.4140   3.3754    0.2702   0.9745   0.0000  -3.6141   2.7025   1.0493   4.0000   2.9225   0.0000   0.0000   0.0000

import numpy as np
import os

class reaxparam:

    def __init__(self, reaxff='cho'):

       if "REAXFF_FILES" in os.environ:
           reaxff_filepath = os.environ["REAXFF_FILES"]
       else:
           print("The variable REAXFF_FILES is not in the operating system environment. Make sure lammps is avaiable.")
           exit()

       f_reaxff = os.path.join(reaxff_filepath,'ffield.reax.'+reaxff)
       assert os.path.isfile(f_reaxff), "The file %s exists." %f_reaxff
      
       f = open(f_reaxff,'r')
       lines = f.readlines()
       
       self.num_to_atom_type = {}
       self.atom_type_to_num = {}
       # parameters for overcoordination correction
       self.pboc4 = []
       self.pboc3 = []
       self.pboc5 = []
       # equilibrium distances for atom types
       self.r_s   = []
       self.r_pi  = []
       self.r_pi2 = []
       # valency of the atoms (needed to correct for over coordination)
       self.valency = []
       self.valency_val = []
       
       for i,line in enumerate(lines):
       
           if "Nr of atoms" in line:
              natoms = int(line.split("!")[0])
              nrows = 4
              for n in range(natoms):
                 values = []
                 for nrow in range(nrows):
                     values += lines[i+nrows*(n+1)+nrow].split()
                 atom_type = values[0].lower()
                 self.num_to_atom_type[n] = atom_type
                 self.atom_type_to_num[atom_type] = n
                 self.pboc4.append(float(values[20]))
                 self.pboc3.append(float(values[21]))
                 self.pboc5.append(float(values[22]))
                 self.r_s.append(float(values[1]))
                 self.r_pi.append(float(values[7]))
                 self.r_pi2.append(float(values[17]))
                 self.valency.append(float(values[2]))
                 self.valency_val.append(float(values[28]))
           if "Nr of bonds" in line:
              # exponents for calculating uncorrected bond order
              self.pbo1 = np.full([natoms, natoms], None)
              self.pbo2 = np.full([natoms, natoms], None)
              self.pbo3 = np.full([natoms, natoms], None)
              self.pbo4 = np.full([natoms, natoms], None)
              self.pbo5 = np.full([natoms, natoms], None)
              self.pbo6 = np.full([natoms, natoms], None)
              self.v13cor = np.full([natoms, natoms], None)
              self.ovc = np.full([natoms, natoms], None)
       
              nbonds = int(line.split("!")[0])
              nrows = 2
              for n in range(nbonds):
                 values = []
                 for nrow in range(nrows):
                     values += lines[i+nrows*(n+1)+nrow].split()
                 at1 = int(values[0]) - 1
                 at2 = int(values[1]) - 1
                 self.pbo1[at1,at2] = float(values[14])
                 self.pbo1[at2,at1] = float(values[14])
                 self.pbo2[at1,at2] = float(values[15])
                 self.pbo2[at2,at1] = float(values[15])
                 self.pbo3[at1,at2] = float(values[11])
                 self.pbo3[at2,at1] = float(values[11])
                 self.pbo4[at1,at2] = float(values[12])
                 self.pbo4[at2,at1] = float(values[12])
                 self.pbo5[at1,at2] = float(values[6])
                 self.pbo5[at2,at1] = float(values[6])
                 self.pbo6[at1,at2] = float(values[8])
                 self.pbo6[at2,at1] = float(values[8])
                 self.v13cor[at1,at2] = float(values[7])
                 self.v13cor[at2,at1] = float(values[7])
                 self.ovc[at1,at2] = float(values[16])
                 self.ovc[at2,at1] = float(values[16])
