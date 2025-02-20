"""
This module contains Turbomole related tools.
"""
import sys
import os
import re
import numpy
import scipy
import tempfile
import time
import shutil
import molsys
from   molsys.util.constants import angstrom
import matplotlib.pyplot as plt
import graph_tool
from graph_tool import Graph, GraphView
import json

class DefineEndedAbnormallyError(Exception):
    def __init__(self, message=None, errors=None):
        # Calling the base class with the parameter it needs
        super().__init__(message)
        self.errors = errors
        return

class SymmetryAssignmentChangedtheStructureError(Exception):
    def __init__(self, message=None, errors=None):
        # Calling the base class with the parameter it needs
        super().__init__(message)
        self.errors = errors
        return


class GeneralTools:
    def __init__(self, path=os.getcwd()):
        if not os.path.isdir(path):
            raise FileNotFoundError("The directory %s does not exist." % path)
        else:
            self.path    = os.path.abspath(path)
            self.maindir = os.getcwd()
        self.nalpha = 0
        self.nbeta = 0
        return
        
    def get_TM_version(self):
        ''' Returns the Turbomole version'''
        TURBODIR = os.popen('echo $TURBODIR').read().strip()
        README = os.path.join(TURBODIR,'README')
        if not os.path.isfile(README):
            raise FileNotFoundError("The file %s does not exist." %README)
        else:
            with open(README) as f:
                for line in f:
                    if "INSTALLATION OF TURBOMOLE" in line:
                        version_tmp = line.strip().split("INSTALLATION OF TURBOMOLE")[1]
                        version = float(version_tmp.split()[0])
        return version

    def invoke_define(self, define_in_name='define.in', define_out_name = 'define.out', coord_name='coord'):
        os.chdir(self.path)
        coord_path = os.path.join(self.path,coord_name)
        define_in_path = os.path.join(self.path,define_in_name)
        if not os.path.isfile(coord_path):
            raise FileNotFoundError("Please provide a coord file for invoking define.")
        if not os.path.isfile(define_in_path):
            raise FileNotFoundError("Please provide an input file for invoking define.")
        out_path = os.path.join(self.path, define_out_name)
        err_path = os.path.join(self.path, "errormessage")

        # 1. invoke define
        # NOTE Before TM7.6 there might be a bug for linear molecules during ired: "Attention! Not enough linearly independent coordinates" 
        os.system('define < %s > %s 2> %s' %(define_in_path, out_path, err_path))

        # 2. check if define was succesful
        with open(err_path) as err:
            for line in err:
                if "define ended normally" not in line:
                    raise DefineEndedAbnormallyError("Define ended abnormally. Check the define input under %s." % define_in_path)
                else:
                    os.remove(err_path)
        os.chdir(self.maindir)
        return

    def write_coord_from_mol(self, mol, coord_name = 'coord'):
        os.chdir(self.path)
        mol.write(coord_name,ftype="turbo")
        os.chdir(self.maindir)
        return   

    def coord_to_mol(self, coord_name="coord"):
        os.chdir(self.path)
        mol = molsys.mol.from_file(coord_name,ftype="turbo")
        os.chdir(self.maindir)
        return mol

    def add_to_coord(self, coord_name="coord", text=""):
        coord_path = os.path.join(self.path,coord_name)
        newlines = ''
        if not os.path.isfile(coord_path):
            raise FileNotFoundError("Please provide a coord file.")
        with open(coord_path) as coord:
           for line in coord:
               if '$end' in line:
                   newlines += '%s\n$end' %text
               else:
                   newlines += line
        f = open(coord_path,'w')
        f.write(newlines)
        f.close()
        return



    def change_basis_set(self, basis_set, blow_up = True, ref_control_file = "../control",  title = "", delete_old_aux_bas = True, ri = True):
        """
        Args:
            blow_up : True  -> Express molecular orbitals of a previous calculation in the new basis
                      False -> TODO $atomdens to generate start MOs
            ref_control_file : Path to the control file of the reference calculation
            ri               : Was the resolution of identity approximation used?
        """
        mol = self.coord_to_mol()
        n_el_type = len(set(mol.elems))
        define_in_path = os.path.join(self.path,'define.in')
        f=open(define_in_path,'w')
        f.write('%s\n' %title)
        f.write('n\n') # do not change geometry data
        f.write('y\n') # yes, we want to change ATOMIC ATTRIBUTE DATA (BASES,CHARGES,MASSES,ECPS)
        f.write('bb all %s\n' %basis_set)
        f.write('*\n') # terminate this section and write data to control
        if delete_old_aux_bas:
            for i in range(n_el_type):
                f.write('d\n') # delete the old auxiliary basis set for each atom type
        if blow_up:
            f.write('use %s\n' %ref_control_file) #  SUPPLY MO INFORMATION USING DATA FROM <ref_control_file>
        else:
            print('Not implemented yet.')
            sys.exit()
        control_path = os.path.join(self.path,"control")
        control = open(control_path,'r')
        lines = control.readlines()
        f.write('n\n') # DO YOU WANT TO DELETE DATA GROUPS LIKE
        f.write('n\n') # we do not want to write natural orbitals
        if ri:
            f.write('ri\n')
            for i in range(n_el_type):
                f.write('y\n')
            f.write('q\n') 
        f.write('q')
        f.close() 
        self.invoke_define()
        return

    def change_dft_functional(self, functional, title = ""):
        define_in_path = os.path.join(self.path,'define.in')
        f=open(define_in_path,'w')
        f.write("%s\n" %title)
        f.write('n\n') # do not change geometry data
        f.write('n\n') # do not change atomic attribute data
        f.write('n\n') # do not change molecular orbital data
        f.write('n\n') # DO YOU WANT TO DELETE DATA GROUPS LIKE
        f.write('n\n') # we do not want to write natural orbitals 
        f.write('dft\n')
        f.write('func %s\n' %functional)
        f.write('q\n')
        f.write('q')
        f.close()
        self.invoke_define()
        return
 

    def define_rohf(self, basis_set, M, scfiterlimit = 300, charge = 0, title = ""):
        TMversion = self.get_TM_version()
        mol  = self.coord_to_mol()
        n_el = mol.get_n_el(charge)
        define_in_path = os.path.join(self.path,'define.in')
        f=open(define_in_path,'w')
        f.write('\n') # IF YOU WANT TO READ DEFAULT-DATA FROM ANOTHER control-TYPE FILE -> NO
        f.write('%s\n' %title)
        f.write('a coord\n')
        f.write('*\n')
        f.write('no\n')
        f.write('bb all %s\n' %basis_set)
        f.write('*\n') # terminate this section and write data to control
        f.write('eht\n')
        f.write('y\n')
        f.write('%d\n' %charge)
        f.write('n\n') # don't accept the occupation
        # M = nalpha - nbeta + 1
        # the number of closed shell orbitals = (number of electrons - number of singly occupied orbitals)/2
        n_somo = (M-1)
        n_c  = (n_el-n_somo)/2
        print('number of closed shell orbitals:',n_c,'; number of singly occupied molecular orbitals:',n_somo)
        f.write("c 1-%d\n" %n_c)
        if TMversion <= 7.6:
            for i in range(int(n_somo)):
                mo = n_c+1+i
                f.write("o %d\n" %(mo))
                f.write('1\n') # THE OPEN-SHELL OCCUPATION NUMBER PER MO
                f.write('y\n') 
            f.write('*\n')
            if n_somo < 3:
                f.write('y\n')
            else:
                f.write('1 2\n') # The Roothaan parameters could not be provided. Assign them as a=1, b=2 (high spin case)
        else:
            if n_somo == 1:
                f.write('oh %d\n' %(n_c+1))
            elif n_somo > 1:
                f.write('oh %d-%d\n' %(n_c+1,n_c+n_somo))
            f.write('*\n')
        f.write('\n')
        f.write('scf\n')
        f.write('iter\n')
        f.write('%d\n' %scfiterlimit)
        f.write('\n\n')
        f.write('q\n')
        f.close()
        self.invoke_define()
        return

    def define_canonical_ccsd(self, model, max_mem=500, F12=True, title=''):
        define_in_path = os.path.join(self.path,'define.in')
        f=open(define_in_path,'w')
        f.write("%s\n" %title)
        f.write('n\n') # do not change geometry data
        f.write('n\n') # do not change atomic attribute data
        f.write('n\n') # do not change molecular orbital data
        f.write('n\n') # DO YOU WANT TO DELETE DATA GROUPS LIKE
        f.write('n\n') # we do not want to write natural orbitals 
        f.write('cc\n')
        f.write('freeze\n') # set frozen occupied/virtual orbital options
        f.write('*\n')
        f.write('cbas\n') # assign auxiliary basis sets
        f.write('*\n')
        f.write('memory %d\n' %max_mem) # set maximum core memory per_core
        if F12:
            f.write('f12\n') # F12 approximation
            f.write('*\n')
            f.write('cabs\n') # complementary auxiliary basis set
            f.write('*\n')
            f.write('jkbas\n') # auxiliary basis for Fock matrices
            f.write('*\n')
        f.write('ricc2\n') 
        f.write('model %s\n' %model) 
        f.write('*\n')
        f.write('*\n')
        f.write('q\n')
        f.close()
        self.invoke_define()
        return


    def define_pnoccsd(self, pnoccsd, max_mem=500, F12=True, title=''):
        define_in_path = os.path.join(self.path,'define.in')
        f=open(define_in_path,'w')
        f.write("%s\n" %title)
        f.write('n\n') # do not change geometry data
        f.write('n\n') # do not change atomic attribute data
        f.write('n\n') # do not change molecular orbital data
        f.write('n\n') # DO YOU WANT TO DELETE DATA GROUPS LIKE
        f.write('n\n') # we do not want to write natural orbitals 
        f.write('pnocc\n')
        f.write('freeze\n') # set frozen occupied/virtual orbital options
        f.write('*\n')
        f.write('cbas\n') # assign auxiliary basis sets
        f.write('*\n')
        f.write('memory %d\n' %max_mem) # set maximum core memory per_core
        if F12:
            f.write('f12\n') # F12 approximation
            f.write('*\n')
            f.write('cabs\n') # complementary auxiliary basis set
            f.write('*\n')
            f.write('jkbas\n') # auxiliary basis for Fock matrices
            f.write('*\n')
        f.write('*\n')
        f.write('q\n')
        f.close()
        self.invoke_define()
        # Now add pnoccd to the control file
        newlines = ''
        control_path = os.path.join(self.path,'control')
        with open(control_path) as control:
           for line in control:
               if '$end' in line:
                   newlines += '$pnoccsd\n%s\n$end'%pnoccsd
               else:
                   newlines += line
        f = open(control_path,'w')
        f.write(newlines)
        f.close()
        return


    def read_charge_from_control(self):
        c = None
        control_path = os.path.join(self.path,"control")
        if not os.path.isfile(control_path):
            raise FileNotFoundError("There is no control file in the directory %s." % self.path)
        with open(control_path) as control:
            for line in control:
                if line.startswith("$charge"):
                    charge = float(next(control).split()[0])
        c = round(charge)
        return c

    def read_ssquare_from_control(self):
        ssquare = None
        control_path = os.path.join(self.path,"control")
        if not os.path.isfile(control_path):
            raise FileNotFoundError("There is no control file in the directory %s." % self.path)
        with open(control_path) as control:
            for line in control:
                if line.startswith("$ssquare"):
                    ssquare = float(next(control).split()[0])
        return ssquare
  

    def get_nalpha_and_nbeta_from_ridft_output(self, ridft_out_name='ridft.out'):
        """ Read the number of alpha and beta electrons from the ridft output file. 
            
            Args:
                ridft_out_name: the name of the ridft output file.

            Returns:
                nalpha: Number of occupied alpha shells
                nbeta: Number of occupied beta shells
        """
        ridft_path = os.path.join(self.path, ridft_out_name)
        if not os.path.isfile(ridft_path):
            raise FileNotFoundError("There is no ridft output file named %s in the directory %s." %(ridft_out_name, self.path))

        with open(ridft_path) as ridft_out:
            for line in ridft_out:
                # Get the number of alpha and beta shell occupations
                if line.startswith('   sum'):
                    self.nalpha = float(line.split()[1])
                    self.nbeta = float(line.split()[2])
        return self.nalpha, self.nbeta


    def calculate_spin_multiplicity_from(self, nalpha, nbeta):
        """ Calculate the spin multiplicity from the number of alpha and beta electrons

            Args:
                nalpha: Number of occupied alpha shells
                nbeta: Number of occupied beta shells

            Returns
                M: The minimum possible multiplicity that the system can get
                   M = 2ms+1 = 2*(nalpha-nbeta)*(1/2)+1 = (nalpha-nbeta)+1
        """
        M = abs(nalpha-nbeta)+1
        return M


    def _get_n_mos(self, split_line):
        """ Reads the total number of occupied electrons in a line under the data group
            $alpha shells or $beta shells

            Args:
                split_line: list of strings: The striped and split line under $alpha shells or $beta shells data group

            Returns:
                n_mos: The total number of occupied electrons
        """
        #  a   1-65       ( 1 )
        if '-' in split_line[1] and ',' not in split_line[1]:
            mos = split_line[1].split('-')
            n_mos = int(mos[1])-int(mos[0])+1
        #  a   65,66      ( 1 )
        elif '-' not in split_line[1] and ',' in split_line[1]:
            mos = split_line[1].split(',')
            n_mos = len(mos)
        #  a   1-64,65    ( 1 )
        elif '-' in split_line[1] and ',' in split_line[1]:
            n_mos = 0
            for mos_range in split_line[1].split(','):
                if '-' in mos_range:
                    mos = mos_range.split('-')
                    n_mos += int(mos[1])-int(mos[0])+1
                else:
                    n_mos += 1
        #  a   66         ( 1 )
        elif '-' not in split_line[1] and ',' not in split_line[1]:
            n_mos = 1
        return n_mos

    def get_M_from_control(self):
        """ Reads the control file and calculates M

            Returns
                M: The minimum possible multiplicity that the system can get
                   M = 2ms+1 = 2*(nalpha-nbeta)*(1/2)+1 = (nalpha-nbeta)+1
        """
        control_path = os.path.join(self.path,"control")
        lines = []
        with open(control_path,'r') as control:
            for line in control:
                lines.append(line)
    
        for i,line in enumerate(lines):
            if '$alpha shells' in line:
                j = 1
                while '$' not in lines[i+j]:
                    split_line = lines[i+j].strip().split()
                    n_mos = self._get_n_mos(split_line)
                    occ = float(split_line[3])
                    self.nalpha += n_mos*occ
                    j += 1
            if '$beta shells' in line:
                j = 1
                while '$' not in lines[i+j]:
                    split_line = lines[i+j].strip().split()
                    n_mos = self._get_n_mos(split_line)
                    occ = float(split_line[3])
                    self.nbeta += n_mos*occ
                    j += 1
        M = abs(self.nalpha-self.nbeta)+1
        return M

    
    def round_fractional_occupation(self, THRESHOLD = 1.0E-7):
        """ Rounds the fractional occupations in the control file under $alpha shells and $beta shells
            and writes a new control file. 
 
            Args:
                THRESHOLD: The threshold for rounding the occupation of the molecular orbitals

        """
        control_path = os.path.join(self.path,'control')
        if not os.path.isfile(control_path):
            raise FileNotFoundError("There is no control file in the directory %s." % self.path)

        # 1. Read the lines of the control file
        lines = []
        with open(control_path,'r') as control:
            for line in control:
                lines.append(line)
        new_lines = lines

        # 2. If the difference between the occupation and the rounded occupation is less than the threshold, round the occupation number
        for i,line in enumerate(lines):
            if '$alpha shells' in line:
                j = 1
                while '$' not in lines[i+j]:
                    split_line = lines[i+j].strip().split()
                    occ = float(split_line[3])
                    if abs(occ-round(occ)) > THRESHOLD:
                        new_lines[i+j] = ' %s       %s                                     ( %s )\n' %(split_line[0],split_line[1],str(round(occ)))
                    j += 1
            if '$beta shells' in line:
                j = 1
                while '$' not in lines[i+j]:
                    split_line = lines[i+j].strip().split()
                    occ = float(split_line[3])
                    if abs(occ-round(occ)) > THRESHOLD:
                        new_lines[i+j] = ' %s       %s                                     ( %s )\n' %(split_line[0],split_line[1],str(round(occ)))
                    j += 1

        # 3. Rewrite the control file with the round occupation numbers
        os.remove(control_path)
        with open(control_path,'w') as new_control:
            for line in new_lines:
                new_control.write(line)
        return


    def for_c1_sym_change_multiplicity_in_control_by(self, N, nalpha, nbeta):
        """ Changes the multiplicity by N, by changing the occupation of the alpha and beta electrons in the control file. 

            Args:
                N: How much lower of higher the multiplicity wants to be changed
                nalpha: Number of occupied alpha shells
                nbeta: Number of occupied beta shells
        """
        if N % 2 != 0:
            print('Provide an even number.')
        else:
            control_path = os.path.join(self.path,'control')
            if not os.path.isfile(control_path):
                raise FileNotFoundError("There is no control file in the directory %s." % self.path)

            # 1. Get the line number where alpha shells are written
            with open(control_path,'r') as control:
                for i,line in enumerate(control):
                    if '$alpha shells' in line:
                       line_alpha = i
            
            # 2. Determine the new occupations
            if nalpha >= nbeta:
                new_alpha = nalpha+N/2
                new_beta  = nbeta-N/2
            elif nalpha < nbeta:
                new_alpha = nalpha-N/2
                new_beta  = nbeta+N/2
            
            # 3. Remove the old occupations from the control file
            self.kdg('alpha shells')
            self.kdg('beta shells')

            # 4. Add the new occupations to the control file
            newlines = ''
            with open(control_path) as control:
               for i,line in enumerate(control):
                   if i == line_alpha:
                       newlines += '$alpha shells\n'
                       newlines += ' a       1-%d   ( 1 )\n' %(new_alpha)
                       newlines += '$beta shells\n' 
                       newlines += ' a       1-%d   ( 1 )\n' %(new_beta)
                   newlines += line
            f = open(control_path,'w')
            f.write(newlines)
            f.close()
        return


    def make_tmole_dft_input(self, elems, xyz, M, max_mem, title, lot, genprep = 0, scf_dsta = 1.0, fermi = True, nue = False, new_ired = True):
        """Creates a tmole input called 'turbo.in' with c1 symmetry.

        Args:
            elems   : the list of elements
            xyz     : numpy array of shape (len(elems),3)
            M       : the (initial) spin multiplicity
            max_mem : Maximum memory per core to use in the calculations.
            title   : title of the job
            lot     : The QM level of theory, must be string, and according to Tmole manual
            genprep : 0 -> perform calculation, 1 -> only prepare the input
            scf_dsta: start value for the SCF damping.
            fermi   : Boolean for Fermi smearing
            nue     : To perform Fermi smearing with a restricted multiplicity, look at the Turbomole manual for further details
            new_ired: To use the new version of the redundant coordinates code
        """
        turbo_in_path = os.path.join(self.path,"turbo.in")
        c = xyz*angstrom
        f = open(turbo_in_path,"w")
        f.write("%title\n")
        f.write("%s\n" % title)
        f.write("%method\n")
        # NOTE For some reason tmole does not modify the memory...
        f.write("ENRGY :: %s [gen_mult = %d, gen_symm = c1, scf_dsta = %f, scf_msil = 1000, scf_rico = %d, for_maxc = %d, genprep = %d]\n" %
                (lot, M, scf_dsta, 0.3*max_mem, 0.7*max_mem, genprep))
        f.write("%coord\n")
        for i in range(len(elems)):
            f.write("  %19.14f %19.14f %19.14f   %-2s\n" %
                    (c[i,0],c[i,1], c[i,2], elems[i]))
        f.write("%add_control_commands\n")
        f.write("$disp3\n")
        if new_ired:
            f.write('$redund_inp\n')
            f.write('   new_version=1\n')
        if fermi:
            if nue:
                f.write("$fermi tmstrt=300.00 tmend= 50.00 tmfac=0.900 hlcrt=1.0E-01 stop=1.0E-03 nue=%d\n" %M)
            else:
                f.write("$fermi tmstrt=300.00 tmend= 50.00 tmfac=0.900 hlcrt=1.0E-01 stop=1.0E-03\n")
        f.write("ADD END\n")
        f.write("%end\n")
        f.close()
        return


    def run_tmole(self):
        os.chdir(self.path)
        os.system("tmole &>/dev/null")
        os.chdir(self.maindir)
        return


    def check_scf_converged(self, scf = 'ridft', scf_out_name = None):
        """ Reads the ridft/dscf output and checks if ridft/dscf converged.

            Returns
                converged: True/False 
        """
        converged = True
        if scf_out_name == None:
            scf_out_name = scf + '.out'
        scfout = open(os.path.join(self.path, scf_out_name),'r')
        for line in scfout:
            if 'ATTENTION: %s did not converge!' %scf in line:
                converged = False
            elif '%s ended abnormally' %scf in line:
                converged = False
        lst = os.listdir(self.path)
        for f in lst:
            f_path = os.path.join(self.path, f)
            if "diff_dft_oper" in f: os.remove(f_path)
            if "diff_densmat"  in f: os.remove(f_path)
            if "diff_fockmat"  in f: os.remove(f_path)
            if "diff_errvec"   in f: os.remove(f_path)
        return converged


    def get_energy_from_scf_out(self, scf_out_name = 'ridft.out'):
        """ Reads the ridft/dscf output and checks if ridft/dscf converged.

            Returns
                SPE: Single point energy in Hartree
        """
        scfout = open(os.path.join(self.path, scf_out_name),'r')
        for line in scfout:
           l = line.split()
           if  'total' in l and 'energy' in l and '=' in l:
               SPE = float(l[4])
        return SPE

    def get_ccsd_energy(self, F12=True, Tstar = True, model='ccsd(t)', ccsd_out_name="pnoccsd.out", n_el=3):
       """ Reads the ccsdf12 and returns the energy.

           Args:
               F12  : Explicitly correlated method?
               Tstar: Do you want to read the T* approximation? (only for F12 calculations)
               model: ccsd, ccsd(t), or ccsd(t0)
               ccsd_out_name: Name of the output file; e.g. 'ccsdf12.out' or 'pnoccsd.out'
               n_el : The number of electrons

           Returns
               the single point energy of the requested method in Hartree
       """
       ccsdout = open(os.path.join(self.path, ccsd_out_name),'r')
       for line in ccsdout:
          if F12:
              if 'Final CCSD(F12*) energy        ' in line:
                  CCSDenergy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(F12*)(T0) energy  ' in line:
                  CCSDT0energy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(F12*)(T0*) energy  ' in line:
                  CCSDT0starenergy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(F12*)(T) energy  ' in line:
                  CCSDTenergy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(F12*)(T*) energy  ' in line:
                  CCSDTstarenergy = float(line.rstrip('\n').split()[5])
          else:
              if 'Final CCSD energy         ' in line:
                  CCSDenergy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(T0) energy   ' in line:
                  CCSDT0energy = float(line.rstrip('\n').split()[5])
              elif 'Final CCSD(T) energy    ' in line:
                  CCSDTenergy = float(line.rstrip('\n').split()[5])

       if model.lower() == 'ccsd':
           return CCSDenergy

       elif model.lower() == 'ccsd(t0)':
           if n_el == 2:
               return CCSDenergy
           else:
               if Tstar:
                   return CCSDT0starenergy
               else:
                   return CCSDT0energy

       elif model.lower() == 'ccsd(t)':
           if n_el == 2:
               return CCSDenergy
           else:
               if Tstar:
                   return CCSDTstarenergy
               else:
                   return CCSDTenergy

       else:
           return None

    def get_energy_from_aoforce_out(self, aoforce_out_name = 'aoforce.out'):
        """ Reads the aoforce output and reads the energy.

            Returns
                SPE: Single point energy in Hartree
                ZPE: Zero point vibrational energy in Hartree
        """
        aoforce_path = os.path.join(self.path, aoforce_out_name)
        with open(aoforce_path) as aoforce:
            for line in aoforce:
                if '  zero point VIBRATIONAL energy  ' in line:
                    ZPE = float(line.split()[6]) # The zero point vibrational energy in Hartree
                if 'SCF-energy' in line:
                    SPE = float(line.split()[3]) # Energy in Hartree
        return SPE, ZPE


    def ridft(self):
        SPE = None
        os.chdir(self.path)
        os.system("ridft > ridft.out")
        converged = self.check_scf_converged("ridft")
        if converged:
            SPE =  self.get_energy_from_scf_out("ridft.out")
        else:
            print("ridft did not converge")
        os.chdir(self.maindir)
        return SPE

    def dscf(self):
        SPE = None
        os.chdir(self.path)
        os.system("dscf > dscf.out")
        converged = self.check_scf_converged("dscf")
        if converged:
            SPE =  self.get_energy_from_scf_out("dscf.out")
        else:
            print("dscf did not converge")
        os.chdir(self.maindir)
        return SPE


    def change_scfdamp(self, start = 1.0, step = 0.05, end = 0.1):
        """ Changes the start value for SCF damping. 
            
            Args:
                dsta: The start value for SCF damping
        """
        control_path = os.path.join(self.path,'control')
        if not os.path.isfile(control_path):
            raise FileNotFoundError("There is no control file in the directory %s." % self.path)
    
        newlines = []
        with open(control_path,'r') as control:
            for i,line in enumerate(control):
                if '$scfdamp' in line:
                    newline = line.split()[0]+' start=%f step=%f min=%f \n' %(start,step,end)
                    newlines.append(newline)
                else:
                    newlines.append(line)
   
        os.remove(control_path)
        with open(control_path,'w') as new_control:
            for line in newlines:
                new_control.write(line)
        return


    def kdg(self, dg_name=""):
        """ Removes the data group named <dg_name>.
         
            Args:
                dg_name: The data group to be removed in the control file. 
                         e.g. $rij --> dg_name = "rij"
        """
        os.chdir(self.path)
        os.system("kdg %s" % dg_name)
        print("The data group %s is removed from the control file." %dg_name)
        os.chdir(self.maindir)
        return


    def similaritycheck_from_mol(self, mol_1, mol_2):
         """ Runs the fortran script similaritycheck and compares similarity of the two structures.

             Args:
                 mol_1, mol_2: Mol objects to be compared

             Returns:
                 is_similar: True/False
         """
         xyz_1 = 'mol_1.xyz'
         mol_1.write(xyz_1)
         xyz_2 = 'mol_2.xyz'
         mol_2.write(xyz_2)
         is_similar = self.similaritycheck_from_xyz(xyz_1, xyz_2)
         os.remove(xyz_1)
         os.remove(xyz_2)
         return is_similar


    def similaritycheck_from_xyz(self, xyz_1, xyz_2, tolstr = 0.25):
        """ Runs the similaritycheck program of Turbomole and compares similarity of the two structures.

            Args:
                xyz_1, xyz_2: The xyz paths of the molecules to be compared
                tolstr      : The RMSD threshold

            Returns:
                is_similar: True/False
        """
        out = os.popen('''echo "'%s' '%s'" | similaritycheck -t %f''' %(xyz_1, xyz_2, tolstr)).read()
        is_similar = False
        if out.split()[-1] == 'T':
           is_similar = True
        if os.path.isfile('ISSIMILAR'):
           os.remove('ISSIMILAR')
        return is_similar


class GeometryTools:

    def add_noise(mol, active_atoms = [], upto = 0.05):
        """ Adds a noise to the structure using a uniform distribution.

        First shifts the noise to the origin by subtracting 0.5,
        then divides it by 10 to get a noise up to 0.05 Angstrom by default.

        No noise is added to the active atoms.

        Args:
            mol         : mol object to whom the noise will be added
            active_atoms: List of atoms to whom the noise will not be added
                          Note! indexing is 1,2,3,... instead of 0,1,2,...
            upto        : The maximum noise added to each coordinate in Angstrom

        Returns:
            xyz : Numpy array of the noise added coordinates (units in Angstrom)
        """
        if active_atoms != []:
            noise_ini = (numpy.random.rand(mol.natoms-len(active_atoms),3)-0.5)*upto
            noise = GeometryTools._make_noise_without_active_atoms(mol, noise_ini, active_atoms) 
        else:
            noise = (numpy.random.rand(mol.natoms,3)-0.5)*upto
        # add the noise array to the coordinates array
        new_xyz = mol.xyz + noise
        return new_xyz

    def _make_noise_without_active_atoms(mol, noise_ini, active_atoms):
        """ For the case of active atoms (in transition states), makes 
        sure that there is no noise added to the coordinates of these atoms.

        Args:
            noise_ini: The numpy array which includes noise for all of the atoms of the molecule
            active_atoms: List of indices of atoms to whom the noise will not be added
                          Note! indexing is 1,2,3,... instead of 0,1,2,...

        Returns:
            noise: The numpy array which includes zeros for the active_atoms and the values from noise_ini for the rest                        
        """
        noise_list = noise_ini.tolist()
        noise_to_add = []
        j = 0
        for i in range(mol.natoms):
            if i+1 in active_atoms:
                noise_to_add.append([0.0,0.0,0.0])
            else:
                noise_to_add.append(noise_list[j])
                j += 1
        noise = numpy.array(noise_to_add)
        return noise

    def get_point_group_from_mol(mol):
        """ Invokes define in a temporary folder to get the assigned symmetry.

        Args:
            mol: Mol object

        Returns:
            point_group: string of the detected point group
        """
        # 1. Create a tmp directory
        curdir = os.getcwd()
        tmp_path = os.path.join(curdir,"tmp")
        os.mkdir(tmp_path)
        GeneralTools(tmp_path).write_coord_from_mol(mol)
        # 2. Write a define file 
        GeometryTools._write_define_check_symmetry(tmp_path)
        # 3. Invoke define 
        os.chdir(tmp_path)
        os.system("define < define.in > define.out")
        os.chdir(curdir)
        # 4. Read the assigned point group
        point_group = GeometryTools.get_point_group_from_control(tmp_path)
        # 5. Remove the tmp directory
        shutil.rmtree(tmp_path)
        return point_group

    def get_point_group_from_coord(path=os.getcwd(), coord_name='coord'):
        """ Invokes define in a temporary folder to get the assigned symmetry.
        Args:
            path: directory where coord file is located
            coord_name: Name of the coord file

        Returns:
            point_group: string of the detected point group
        """
        mol = GeneralTools(path).coord_to_mol(coord_name)
        point_group = GeometryTools.get_point_group_from_mol(mol)
        return point_group

    def _write_define_check_symmetry(path=os.getcwd()):
        """ Writes a 'define.in' file to get the detected point group. """
        define_in_path = os.path.join(path,'define.in')
        f=open(define_in_path,'w')
        f.write('\n\na coord\ndesy\n*\nno\nqq')
        f.close()
        return
    
    def get_point_group_from_control(path=os.getcwd()):
        """ Reads the point group from control file. 
        Args:
            path: directory where contol file is located

        Returns:
            point_group: string of the point group  read from control file
        """
        with open(os.path.join(path,'control')) as control:
             for lines in control:
                 if 'symmetry' in lines:
                     point_group = lines.strip().split()[1]
        return point_group


    def _write_define_new_point_group(new_point_group='c1', path=os.getcwd()):
        # 1. Get the number of alpha and beta electrons
        nalpha, nbeta = GeneralTools(path).get_nalpha_and_nbeta_from_ridft_output()
        # 2. Get the charge
        charge = GeneralTools(path).read_charge_from_control()
        # 3. Write a define input
        define_in_path = os.path.join(path,"define.in")
        f = open(define_in_path,"w")
        f.write("\n")
        f.write("y\n")
        f.write("desy \n")
        f.write("sy %s\n" % new_point_group)
        f.write("ired\n")
        f.write("*\n")
        f.write("\n")
        f.write("eht\n")
        f.write("\n")
        f.write("%d\n" % charge)
        f.write("n\n")
        f.write("u %d\n" % abs(nalpha-nbeta))
        f.write("*\n\n\nq")
        f.close()
        return


    def _get_natoms_from_control(path=os.getcwd()):
        if not os.path.isdir(path):
            raise FileNotFoundError("The directory %s does not exist." % path)
        with open(os.path.join(path,'control'),'r') as control:
            for lines in control:
                if 'natoms' in lines:
                    natoms = int(lines.strip().split('=')[-1])
        return natoms


    def change_point_group(path=os.getcwd(), point_group='c1'):
        """ Invokes define to assign a point group. 
            Note! If an abelian point group is given, it assigns an abelian subgroup to avoid degenaricies as this needs manual assignment.

        Args:
            path: the directory where the Turbomole input files are located
            point_group: The desired point group to be assigned

        Returns:
            assigned_point_group: The assigned point group

        """
        non_abelian_point_groups = {  'o':'d2',  'oh':'d2h',  'td':'c2v',  'th':'s6',    't':'d2',
                                'd2d':'c2v','d3d':'s6',  'd4d':'s8',  'd5d':'s10', 'd6d':'s12','d7d':'s14','d8d':'s16',
                                'd3h':'c2v','d4h':'c4h', 'd5h':'c2v', 'd6h':'c2h', 'd7h':'c2v', 'd8h':'c2h',
                                'c3v':'cs', 'c4v':'c2v', 'c5v':'c5',  'c6v':'c2v', 'c7v':'c7', 'c8v':'c8',
                                 'd3':'c3',  'd4':'c4',   'd5':'c5',   'd6':'c6',   'd7':'c7',  'd8':'c8'}
        if point_group in non_abelian_point_groups:
            point_group_to_assign = non_abelian_point_groups[point_group] 
        else:
            point_group_to_assign = point_group
        # 1. Get the number of atoms
        natoms = GeometryTools._get_natoms_from_control(path)
        # 2. Write the define.in file
        GeometryTools._write_define_new_point_group(point_group_to_assign, path)
        # 3.  define
        GeneralTools(path).invoke_define(define_out_name = "define-sym.out")
        # 4. Check if the number of atoms is still the same
        newnatoms = GeometryTools._get_natoms_from_control(path)
        if natoms != newnatoms:
            raise SymmetryAssignmentChangedtheStructureError("The structure is does not follow the  point group %s symmetry operations. Therefore, while trying to change the symmetry group, new atoms are added to enforce it." % new_point_group)
        assigned_point_group = point_group_to_assign
        return assigned_point_group

        


class OptimizationTools:
    """Methods for the optimization of QM species with ridft.
    
    Args:
        path    : the directory where the QM calculations will be performed
        lot     : level of theory
                  for DFT: "ri-u<DFT functional>/<Basis Set>
                  Note! Only unrestricted DFT and with rij approximation is implemented.
        max_mem : Maximum memory to be used on each processor in MB
        gcart            : converge maximum norm of cartesian gradient up to 10^-<gcart> atomic units
    """
    def __init__(self, path=os.getcwd(), lot = 'ri-utpss/SVP', max_mem = 500, gcart = 3, reaxff = 'cho', verbose = False):
        if not os.path.isdir(path):
            raise FileNotFoundError("The directory %s does not exist." % path)
        else:
            self.verbose = verbose
            self.path = os.path.abspath(path)
            self.maindir = os.getcwd()
            self.lot = lot # e.g. ri-utpss/SVP, ri-utpssh/TZVP
            self.max_mem = max_mem
            self.gcart = gcart
            self.reaxff = reaxff
        return

    ### CHANGES WITH DEFINE #########################################################

    def freeze_atoms(self, active_atoms, path = ''):
        """ writes a new define file to fix active atoms in internal coordinates. """
        if path == '': path = self.path
        a_a = ''
        for i in active_atoms:
            a_a += str(i)
            if i != active_atoms[-1]:
               a_a += ',' 
        define_in_path = os.path.join(path,'define.in')
        f = open(define_in_path, 'w')
        f.write(' \n')
        f.write('y\n')
        # adds the letter f to the coord file next to the active atoms
        f.write('fix %s\n' %(a_a))
        # defines internal coordinates, also taking into account of the active atoms
        f.write('ired\n')
        # removes the letter f in the coord file, so that only internal coordinates are frozen
        f.write('fix none\n')
        # leaves the geometry menu
        f.write('*\n')
        # exits define
        f.write('qq\n')
        f.close()
        os.chdir(self.path)
        os.system('define < %s > %s' %(define_in_path, os.path.join(path,'define.out')))
        os.chdir(self.maindir)
        return

    def ired_and_itvc_1(self, rmax = 3e-2, path = ''):
        """writes a new define file to define internal redundant coordinates,
           changes the itvc to 1 (for TS optimization), 
           and changes the coordinates to the redundant internal coordinates.

        Args:
            rmax: Maximum thrust radius
            path: the directory where the Turbomole input files are located
        """
        if path == '': path = self.path
        define_in_path = os.path.join(path,'define.in')
        f = open(define_in_path, 'w')
        f.write(' \n')
        f.write('y\n')
        f.write('ired\n')
        f.write('*\n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write('stp\n')
        f.write('itvc\n')
        f.write('1\n')
        f.write('rmax %e\n' %rmax)
        f.write('*\n')
        f.write('q\n')
        f.close()
        GeneralTools(path).invoke_define()
        return

    def change_rmax(self, rmax = 3e-2, path = ''):
        """ to change the maximum thrust radius for the geometry optimizations. 

        Args:
            rmax: Maximum thrust radius
            path: the directory where the Turbomole input files are located
        """
        if path == '': path = self.path
        define_in_path = os.path.join(path,'define.in')
        f = open(define_in_path, 'w')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write('stp\n')
        f.write('rmax %e\n' %rmax)
        f.write('*\n')
        f.write('q\n')
        f.close()
        GeneralTools(path).invoke_define()
        return


    def ired(self, path = ''):
        """writes a new define file to define internal redundant coordinates,
           and changes the coordinates to the redundant internal coordinates.
        """
        if path == '': path = self.path
        define_in_path = os.path.join(path,'define.in')
        f = open(define_in_path, 'w')
        f.write(' \n')
        f.write('y\n')
        f.write('ired\n')
        f.write('*\n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        f.write('q\n')
        f.close()
        GeneralTools(path).invoke_define()
        return

    def check_imaginary_frequency(self, path = ''):
        if path == '': path = self.path
        f_vibspec = os.path.join(path,"vibspectrum")
        with open(f_vibspec) as vibspec:
           linenum = 0
           inum = 0 # number of imaginary frequencies
           imfreq = []
           for lines in vibspec:
               linenum += 1
               if (linenum > 3 and 'end' not in lines):
                   s = lines.rstrip('\n').split(None,3)
                   v = float(s[2])
                   if (v < 0):
                       inum += 1
                       imfreq.append(float(v))
        vibspec.close()
        if self.verbose: print('The number of imaginary frequencies is ', inum,'and they are/it is', imfreq)
        return inum, imfreq

    def disturb_and_reoptimize(self, natoms=None):
        os.chdir(self.path)
        # 1. Copy necessary files to a file new directory called disturb
        os.system("cpc disturb")
        path_tmp = os.path.join(self.path,"disturb")
        OT = OptimizationTools(path_tmp, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        GT = GeneralTools(path_tmp)
        if natoms == None:
            mol = GT.coord_to_mol() 
            natoms = mol.natoms
        os.chdir(path_tmp)
        # 2. Disturb the coordinates along the first normal vibration with the default T value 298 K.
        os.system('echo -e "1\n\n" | vibration')
        # 3. Read the new coordinates from the control file
        newcoords = os.popen("sdg newcoord | tail -%s" %natoms).read()
        # 4. Write new coord file
        f = open("coord", "w")
        f.write("$coord\n"+newcoords+"$end")
        f.close()
        # 5. Define internal redundant coordinates
        OT.ired()
        # 6. Geometry optimization
        converged = OT.jobex()
        if not converged:
            print("Geometry optimization failed.")
            sys.exit()
        # 7. Calculate Hessian
        OT.aoforce()
        inum, imfreq = OT.check_imaginary_frequency()
        if inum == 0:
            print("The minimum is found!")
            # 8. Remove the old files
            old_files = [ os.path.join(self.path,f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path,f)) ]
            for f in old_files:
               os.remove(f)
            # 9. Move the tmp dir to the main one
            for f in os.listdir(path_tmp):
                shutil.move(os.path.join(path_tmp,f), os.path.join(self.path,f))
            os.rmdir(path_tmp)
        else:
            print("This is not a minimum!")
            sys.exit()
        os.chdir(self.maindir)
        return


    ### CALLING TURBOMOLE BINARIES #########################################################

    def jobex(self, ts=False, cycles=150, gcart=None):
        """ Runs jobex.

        Args:
           ts    : Eigenvector following? 
           cycles: Maximum number of iterations
           gcart : Threshold for convergence, larger -> tighter

        Returns:
           converged: Did the geometry optimization converge?
        """
        if gcart == None: gcart = self.gcart
        converged = False
        os.chdir(self.path)
        if ts:
            os.system("jobex -ri -c %d -trans -gcart %d > jobex.out" %(cycles,gcart))
        else:
            os.system("jobex -ri -c %d -gcart %d > jobex.out" %(cycles,gcart))
        f_path = os.path.join(self.path,"GEO_OPT_CONVERGED")
        if os.path.isfile(f_path):
                converged = True
                os.remove(f_path)
        os.chdir(self.maindir)
        return converged


    def t2x(self):
        os.chdir(self.path)
        os.system("t2x coord > coord.xyz")
        os.chdir(self.maindir)
        return


    def aoforce(self):
        os.chdir(self.path)
        os.system("aoforce > aoforce.out")
        if os.path.isfile("dh"):
            os.remove("dh")
        os.chdir(self.maindir)
        return


    def IRC(self, cycle = 150):
        os.chdir(self.path)
        os.system("DRC -i -c %d > IRC.out" %cycle)
        os.chdir(self.maindir)
        return


    def pnoccsd(self):
        os.chdir(self.path)
        os.system("pnoccsd > pnoccsd.out")
        os.chdir(self.maindir)
        return

    def ccsdf12(self):
        os.chdir(self.path)
        os.system("ccsdf12 > ccsdf12.out")
        os.chdir(self.maindir)
        return

    def _clean_ccsd(self, ccsd_path):
        lst = os.listdir(ccsd_path)
        for f in lst:
            f_path = os.path.join(ccsd_path, f)
            if f in ["wherefrom","statistics"] or f.startswith("PNO") or f.startswith("CC") or f.startswith("RIR12"):
                os.remove(f_path)
        return

    def add_dg_to_control(self, dg="arh"):
        os.chdir(self.path)
        os.system("kdg end")
        os.system('''echo "\$%s" >> control''' %dg)
        os.system('''echo "\$end" >> control''')
        os.chdir(self.maindir)
        return

    ### USING mol OBJECT #########################################################

    def make_molecular_graph(self, mol, by_bo = True):
        # 1) Detect the connectivity information by bond order
        if by_bo:
            mol.detect_conn_by_bo(reaxff = self.reaxff)
        else:
            mol.detect_conn()
        # 2) Make the molecular graph
        if  not hasattr(mol, "graph"):
            mol.addon("graph")
        mol.graph.make_graph()
        mg = mol.graph.molg
        return mg

    def separate_molecules(self, mol, by_bo = False):
       """ Separates the molecules according to their molecular graph.

       By default the connectivity is determined without the bo.

       Returns:
           mols: a dictionary of mol objects of the separated molecules
           labels: list of labels for individual fragments
       """
       mg = self.make_molecular_graph(mol = mol, by_bo = by_bo)
       # 1) Label the components of the molecular graph to which each vertex in the the graph belongs
       from graph_tool.topology import label_components
       labels = label_components(mg)[0].a.tolist()

       # 2) Number of molecules
       n_mols = len(set(labels))

       # 3) Now create mol objects with the separated molecules and append them into a list
       mols = []
       for i in set(labels):
           n_atoms = labels.count(i)
           mol_str = '%d\n\n' %n_atoms
           counter = 0
           for j,label in enumerate(labels):
               if i == label:
                   mol_str += '%s %5.10f %5.10f %5.10f' %(mol.elems[j], mol.xyz[j,0], mol.xyz[j,1], mol.xyz[j,2])
                   counter += 1
                   if counter != n_atoms:
                       mol_str += '\n'
           mol_tmp = molsys.mol.from_string(mol_str, 'xyz')
           mol_tmp.detect_conn_by_bo(reaxff = self.reaxff)
           mols.append(mol_tmp)
       return mols, labels


    def get_max_M(self, mol):
        """ Calculates the number of core + valence MOs (nMOs)
        and from there calculates the maximum multiplicity that molecule possibly have.

        Returns:
            Max_M: the maximum multiplicity that molecule can possibly have
        """
        # The dictionary of the minimal number of atomic orbitals
        # i.e. core + valence shells
        dic_nAOs = {'h':1,'he':1, 'c':5, 'n':5, 'o':5, 'f':5, 'ne':5, 's':9, 'cl':9, 'ar':9}
        nMOs = 0
        for t in set(mol.elems):
           assert t in dic_nAOs, 'The element %s is in the dictionary.' %t.capitalize()
           amount = mol.elems.count(t)
           nMOs += dic_nAOs[t]*amount
        nel = mol.get_n_el()
        alphashells = nMOs
        betashells = nel - nMOs
        NumberofMaxUnpairedElectrons = alphashells - betashells
        Max_M = NumberofMaxUnpairedElectrons + 1
        return Max_M


    ### TOOLS TO SET UP CALCULATIONS #########################################################

    def generate_MOs(self, mol, title = '', add_noise = True, upto = 0.05, active_atoms = [], fermi = True, pre_defined_M = False, lowest_energy_M = False, M = None) :
        """ To generate molecular orbitals
     
        Args:
            mol             : Mol object
            title           : Title of the calculation
            add_noise       : Do you want to add a noise within <upto> Angstrom?
            upto            : The maximum limit of the noise in Angstrom
            active_atoms    : List of atoms where the noise should not be applied 
                              Note! with atom indicing starting with 1) --> 1,2,3,...
            fermi           : True  -> You want to apply Fermi smearing to obtain the ground state multiplicity.
                                       It will start with M = 2 for the systems with even number of electrons
                                                          M = 3 for the systems with off number of electrons
                              False -> You want start with extended Hückel theory. -> You must specify an <M>.
            pre_define_M    : True  -> It will apply Fermi smearing but then will modify occupations according to the value <M> provided.
            lowest_energy_M : True  -> Apply Fermi smearing and look at multiplicities which are +-2 different than the value obtained from Fermi smearing.
            M               : |2 ms + 1| = |(nalpha-nbeta) + 1|

        Returns:
            energy          : The energy of the optimized MOs, in Hartree
        """
        GT = GeneralTools(self.path)
        atom = False
        if add_noise:
            xyz = GeometryTools.add_noise(mol, active_atoms = active_atoms, upto = upto)
        else:
            xyz = mol.xyz

        if mol.natoms == 1:
            atom = True

        #############################
        # CASES WITH FERMI SMEARING #
        #############################
        if fermi:

            # 1. Determine the start multiplicity
            nel = mol.get_n_el()
            if (nel % 2) == 0:
                M_start = 3
            else:
                M_start = 2

            # 2. Perform single point calculation with Fermi smeatring
            GT.make_tmole_dft_input(
                    elems = mol.elems, 
                    xyz = xyz, 
                    M = M_start, 
                    max_mem = self.max_mem, 
                    title = title, 
                    lot = self.lot, 
                    scf_dsta = 1.0, # SCF start damping
                    fermi = True, # True for Fermi smearing 
                    nue = False) # True to enforce a multiplicity in the Fermi smearing
            GT.run_tmole()
            converged = GT.check_scf_converged()


            # 3. If SCF did not converge, increase the SCF start damping to 2.0 instead of 1.0.
            if not converged:
                print('The SCF calculation for Fermi smearing with start damping 1.0 did not converge. Increasing it to 2.0.' )
                for f in os.listdir(self.path):
                    os.remove(os.path.join(self.path,f))
                GT.make_tmole_dft_input(
                        elems = mol.elems,  
                        xyz = xyz,
                        M = M_start,
                        max_mem = self.max_mem,
                        title = title,
                        lot = self.lot,
                        scf_dsta = 2.0, # SCF start damping
                        fermi = True, # True for Fermi smearing 
                        nue = False) # True to enforce a multiplicity in the Fermi smearing
                GT.run_tmole()
                converged = GT.check_scf_converged()
                if not converged:
                    print('WARNING! The SCF calculation for Fermi smearing with start damping 2.0 did not converge. Increasing it to 10.0.' )
                    for f in os.listdir(self.path):
                        os.remove(os.path.join(self.path,f))
                    GT.make_tmole_dft_input(
                            elems = mol.elems,
                            xyz = xyz,
                            M = M_start,
                            max_mem = self.max_mem,
                            title = title,
                            lot = self.lot,
                            scf_dsta = 10.0, # SCF start damping
                            fermi = True, # True for Fermi smearing 
                            nue = False) # True to enforce a multiplicity in the Fermi smearing
                    GT.run_tmole()
                    converged = GT.check_scf_converged()
                    if not converged:
                        print('The SCF calculation with Fermi smearing did not converge also with start damping 10.0.')
                        sys.exit()
                    else:
                        # decrease the start damping to 2.0
                        print('The SCF calculation converged with start damping 10.0. Removing Fermi option, decreasing SCF start damping to 2.0 and re-performing the SCF calculation.')
                        GT.kdg("fermi")
                        GT.change_scfdamp(start=2.0)
                        GT.ridft()
                        converged = GT.check_scf_converged()
                        if not converged:
                            print('The SCF calculation with start damping 2.0 did not converge.')
                            sys.exit()
                else:
                    print('The SCF calculation converged with start damping 2.0. Removing Fermi option (if not removed yet).')
                    GT.kdg("fermi")
                # decrease the start damping to 1.0
                print('Decreasing SCF start damping to 1.0 and re-performing the SCF calculation.')
                GT.change_scfdamp(start=1.0)
                energy = GT.ridft()
                converged = GT.check_scf_converged()
                if not converged:
                    print('The SCF calculation with start damping 1.0 did not converge.')
                    sys.exit()

            # 4. Remove the data group $fermi from the control file
            GT.kdg("fermi")

            # 5. Add to the control file the Augmented Roothan Hall solver keyword
            self.add_dg_to_control("arh")

            # 6. If there are partial occupations round them to integers
            GT.round_fractional_occupation()
            energy = GT.ridft()
            print("The energy of the structure is %f Hartree." %energy)

            # 7. Now get the spin multiplicity from Fermi
            nalpha, nbeta = GT.get_nalpha_and_nbeta_from_ridft_output()
            M_fermi = GT.calculate_spin_multiplicity_from(nalpha, nbeta)

            ### TS ###
            if pre_defined_M:
                assert M != None, "You must provide a multiplicity for a TS!"
 
                # 8. If the multiplicity changes in the Fermi smearing, change the multiplicity such that the desired multiplicity is used.
                if M_fermi != M:
                    print("The spin multiplicity after Fermi smearing is %d, it will be changed to %d." %(M_fermi, M))
                    GT.for_c1_sym_change_multiplicity_in_control_by(M-M_fermi, nalpha, nbeta)   
                    energy = GT.ridft()
                    converged = GT.check_scf_converged()
                    if not converged:
                        print('The SCF calculation did not converge starting with orbitals from Fermi and multiplicity %d.' %M)
                        sys.exit()

            ### Equilibrium Structure ###
            if lowest_energy_M:
                # 7. Now calculate two lower spin multiplicities
                dict_energies = {energy:M_fermi}
                if nel > 1:
                    new_dirs = []
                    if M_fermi-2 > 0:
                        m2_path = os.path.join(self.path,'M_%d' %(M_fermi-2))
                        new_dirs.append(m2_path)
                        os.chdir(self.path)
                        os.system('cpc %s' %m2_path)
                        os.chdir(m2_path)
                        GT_m2 = GeneralTools(m2_path)
                        GT_m2.for_c1_sym_change_multiplicity_in_control_by(-2, nalpha, nbeta)
                        energy_m2 = GT_m2.ridft()
                        converged = GT_m2.check_scf_converged()
                        if converged:      
                            dict_energies[energy_m2] = M_fermi-2
                        else:
                            print('The SCF calculation with multiplicity %d did not converge. It is not added to the dictionary.' %(M_fermi-2))
                        os.chdir(self.maindir)
                    # The maximum possible multiplicity that the molecule can have
                    M_max = self.get_max_M(mol)
                    if M_fermi < M_max:
                        p2_path = os.path.join(self.path,'M_%d' %(M_fermi+2))
                        new_dirs.append(p2_path)
                        os.chdir(self.path)
                        os.system('cpc %s' %p2_path)
                        os.chdir(p2_path)
                        GT_p2 = GeneralTools(p2_path)
                        GT_p2.for_c1_sym_change_multiplicity_in_control_by(+2, nalpha, nbeta)
                        energy_p2 = GT_p2.ridft()
                        converged = GT_p2.check_scf_converged()
                        if converged:
                            dict_energies[energy_p2] = M_fermi+2
                        else:
                            print('The SCF calculation with multiplicity %d did not converge. It is not added to the dictionary.' %(M_fermi+2))
                        os.chdir(self.maindir)
                    M_final = dict_energies[min(dict_energies)]
                    print('The dictionary of energies and multiplicities:')
                    print(dict_energies)
                    if M_final != M_fermi:
                        print('The multiplicity %d results in lower energy than %d.' %(M_final, M_fermi))
                        path_min = os.path.join(self.path,'M_%d' %(M_final))
                        path_fermi = os.path.join(self.path,'M_%d' %(M_fermi))
                        os.mkdir(path_fermi)
                        # move the files the path where the original Fermi smearing was done to a new directory.
                        for fname in os.listdir(self.path):
                            f = os.path.join(self.path,fname)
                            if os.path.isfile(f) and fname not in ['submit.py','submit.out']:
                                shutil.move(f, path_fermi)
                        # move the files of the multiplicity which gives the lowest energy to a higher directory.
                        for fname in os.listdir(path_min):
                            f = os.path.join(path_min,fname)
                            shutil.move(f, self.path)
                        shutil.rmtree(path_min)
                        print('The calculations will proceed using multiplicity %d.' % M_final )
                    else:
                        print('The multiplicity %d will be used.' %M_final)

        #################################
        # CASES WITHOUT FERMI SMEARING  #
        #################################
        # For cases without Fermi smearing starting from extented Hueckel Theory guess for a given multiplicity
        else:
            assert M != None, "You must provide a multiplicity without Fermi smearing!"
            GT.make_tmole_dft_input(
                    elems   = mol.elems,
                    xyz     = xyz,
                    M       = M,
                    max_mem = self.max_mem,
                    title   = title,
                    lot     = self.lot,
                    fermi   = False, # True for Fermi smearing
                    nue     = False) # True to enforce a multiplicity in the Fermi smearing
            GT.run_tmole()
            converged = GT.check_scf_converged()
            # If SCF did not converge, increase the SCF start damping to 2.0 instead of 1.0.
            if not converged:
                print('The SCF calculation with start damping 1.0 did not converge. Increasing it to 2.0.' )
                for f in self.path:
                    os.remove(f)
                GT.make_tmole_dft_input(
                        elems = mol.elems,
                        xyz = xyz,
                        M = M,
                        max_mem = self.max_mem,
                        title = title,
                        lot = self.lot,
                        scf_dsta = 2.0, # SCF start damping
                        fermi = False, # True for Fermi smearing 
                        nue = False) # True to enforce a multiplicity in the Fermi smearing
                GT.run_tmole()
                converged = GT.check_scf_converged()
                if not converged:
                    print('The SCF calculation did not converge also with start damping 2.0.')
                    sys.exit()
                else:
                    print('The SCF calculation converged with start damping 2.0. Decreasing it back to 1.0 and re-performing the SCF calculation.')
                    GT.change_scfdamp(start=1.0)
                    energy = GT.ridft()
                    converged = GT.check_scf_converged()
                    if not converged:
                        print('The SCF calculation with start damping 1.0 did not converge.')
                        sys.exit()
            energy = GT.get_energy_from_scf_out("ridft.out")

        converged = GT.check_scf_converged()
        if not converged:
            print('The ridft calculation did not converge.')
            sys.exit()
        return energy


    ### TOOLS TO CHECK THE END POINTS FROM IRC CALCULATIONS ################################################

    def list_of_list_rbonds(self, rbonds):
        """ findR returns the reactive bonds as a list, this returns them into a list of list
        Args:
            rbonds: list of reactive bonds; e.g. [0,2,5,7]

        Returns:
            bonds: list of list of reactive bonds; e.g. [[0,2],[5,7]]
        """
        bonds = []
        bond_tmp = []
        for i, index in enumerate(rbonds):
            if i%2 == 0:
               bond_tmp.append(index)
            else:
               bond_tmp.append(index)
               bonds.append(sorted(bond_tmp))
               bond_tmp = []
        return bonds

    def _match_wrt_ts(self, mol_spec, mol_ts, label, n, rbonds, atom_ids_dict):
        """ This method matches the indices of the mol_spec (an equilibrium species) with that of the mol_ts.
 
            The redundant species are written to the database => The indices in the atom_ids_dict cannot be 
            directly used to re-order the species. Therefore, the indices of the species should be mapped on 
            the indices of an "extracted species" from the transition state structure by using molecular graph isomorphism.
        
        Args:
            mol_spec, mol_ts : The mol objects, with detected connectivities
            label            : 'educt' or 'product'
            n                : index which shows which educt/product it is
            atoms_ids_dict   : The dictionary which holds the list of atom_ids from the ReaxFF trajectory
                               The keys of the dictionary should be as "label_n"; 
                               e.g. label = "educt", n = 1 => The corresponding key is "educt_1"
            rbonds           : list of reactive bonds; e.g. [0,2,5,7]
            
        Returns:
            ratom     : The index of the reactive atom on the species
            rbond_btw : The reactive bond which is between the two "educts" or between the two "products".
            vts2vspec : The dictionary to map from transition state structure to the species indices.

        """
        # 1. Convert the rbonds to a list of lists
        rbonds = self.list_of_list_rbonds(rbonds)

        # 2. Extract the "equilibrium" species from the TS structure and create a mol object
        counter = 0
        natoms = len(mol_spec.elems)
        mol_str = '%d\n\n' %natoms
        ReaxFF2match = {} # dictionary to match the indices of the ReaxFF optimized TS with the DFT optimized equilibrium species.
        match2ReaxFF = {}           
        # Loop over the indices of the ReaxFF optimized TS and its atom indices from the MD simulation
        for i_ReaxFF, i_fromMD in enumerate(atom_ids_dict['ts']):
            # Loop over the indices of the "equilibrium" species from the MD simulation
            for j in atom_ids_dict['%s_%d' %(label,n)]:

                # Match the indices of TS from MD simulation to those of the "equilibrium" species from MD simulation
                if i_fromMD == j:
                    # Mapping of the ReaxFF optimized TS indices to the extracted equilibrium structure
                    ReaxFF2match[i_ReaxFF] = counter
                    match2ReaxFF[counter] = i_ReaxFF
                    mol_str += '%s %5.10f %5.10f %5.10f\n' %(mol_ts.elems[i_ReaxFF], mol_ts.xyz[i_ReaxFF,0], mol_ts.xyz[i_ReaxFF,1], mol_ts.xyz[i_ReaxFF,2])
                    counter += 1

        # 3. Create a mol object for the extracted equilibrium structure from the TS structure.
        mol_match = molsys.mol.from_string(mol_str, 'xyz')
        mol_match.detect_conn_by_bo(reaxff = self.reaxff)

        # 4. Check if a reactive bond is within the extracted species
        # As the "equilibrium" species is extracted from the TS, the bond might not be connected.
        # This will cause a non-isomorphism; and therefore, the matching of the indices will fail.
        # To avoid this, assure that the reactive bonds are artificially connected.
        ratom = None
        rbond_btw = []
        for rb in rbonds:
            if rb[0] not in mol_ts.conn[rb[1]]:
                mol_ts.conn[rb[0]].append(rb[1])
                mol_ts.conn[rb[1]].append(rb[0])
            if set(ReaxFF2match).intersection(rb) == set(rb):
                print('The reactive bond %d---%d belongs to the %s_%d.' %(rb[0]+1,rb[1]+1,label,n))
                i0 = ReaxFF2match[rb[0]]
                i1 = ReaxFF2match[rb[1]]
                if i0 not in mol_match.conn[i1]:
                    mol_match.conn[i0].append(i1)
                    mol_match.conn[i1].append(i0)
            elif set(ReaxFF2match).intersection(rb) != set():
                 ratom = list(set(ReaxFF2match).intersection(rb))[0]
                 rbond_btw = rb
                 print('The reactive bond %d---%d is between the two %ss and %d belongs to the %s_%d.' %(rb[0]+1,rb[1]+1,label,ratom+1,label,n))
        if n==2 and ratom == None:
            print('There is no reactive bond between the two molecules! Exiting...')
            sys.exit()

        # 5. Make molecular graphs of the DFT opt species and extracted one
        if not hasattr(mol_spec, "graph"):
            mol_spec.addon("graph")
        mol_spec.graph.make_graph()
        mg_spec = mol_spec.graph.molg
        mol_spec.graph.plot_graph('mg_%s_%d'  %(label, n))
        mol_spec.write("mol_%s_%d.xyz" %(label, n))

        if  not hasattr(mol_match, "graph"):
            mol_match.addon("graph")
        mol_match.graph.make_graph()
        mg_match = mol_match.graph.molg
        mol_match.graph.plot_graph('mg_match')
        mol_match.write("mol_match.xyz")

        # 6. Now compare the molecular graphs of the DFT optimized species and the "extracted" equilibrium structure to get the matching indices.
        # a) compare the graph
        # isomorphism is buggy ---> first indices are mapped ---> then vertex types are compared => prone to fail
        #is_equal, isomap = graph_tool.topology.isomorphism(mg_spec,mg_match, vertex_inv1=mg_spec.vp.type, vertex_inv2=mg_match.vp.type, isomap=True)
        # Therefore, as a workaround use subgraph_isomorphism...
        masterg = Graph(mg_match)
        masterg.add_vertex()
        vertex_maps = graph_tool.topology.subgraph_isomorphism(mg_spec, masterg, max_n=0, vertex_label=(mg_spec.vp.type,masterg.vp.type), edge_label=None, induced=False, subgraph=True, generator=False) 
        is_equal = len(vertex_maps) > 0
        if is_equal == False:
            print('The graph of the %s_%d is not isomorphic to the graph of the fragment of %s_%d in the transition state. Exiting...' %(label,n,label,n))
            sys.exit()
        isomap = vertex_maps[0]
        # b) get matching indices
        vts2vspec = {}
        print("------------------")
        print(" Matching indices")
        print("------------------")
        print("  spec   |   ts   ")
        print("------------------")
        for vspec, vmatch in zip(mg_spec.vertices(), isomap):
            vts = match2ReaxFF[vmatch]
            print("  %2s%3d  |  %2s%3d " %(mol_spec.elems[int(vspec)].capitalize(), int(vspec)+1, mol_ts.elems[vts].capitalize(), vts+1))
            vts2vspec[vts]  = int(vspec)
        return ratom, rbond_btw, vts2vspec


    def reorder_wrt_ts(self, QM_path, ts_path, label, n, rbonds, atom_ids_dict):
        """ Re-orders by reading xyz files of the optimized molecules under the directories educt_1, educt_2, ..., product_1, product_2, ...,
            according to the order of the transition state.
            
            TODO Generalise this method: probably instead of creating the mol object here, it could make more sense to 
                 give mol objects as an argument, and instead of creating the key of the dictionary as "<label>_<n>"
                 one can give as an argument the keys of the atom_ids_dict dictionary.

        Args:
            QM_path          : Where the calculations are being performed. Actually, this is the same as self.path. 
                               I don't know why I wrote it like this. 
            ts_path          : The path to the xyz file of the transition state.
            label            : 'educt' or 'product'
            n                : index which shows which educt/product it is
            rbonds           : list of reactive bonds; e.g. [0,2,5,7]
            atoms_ids_dict   : The dictionary which holds the list of atom_ids from the ReaxFF trajectory

        Returns:
            mol_ordered      : Mol object of the re-ordered molecule
        """
        # 1. Make a mol object for the species
        path_spec = os.path.join(QM_path,"%s_%d" %(label,n))
        mol_spec = GeneralTools(path_spec).coord_to_mol()
        mol_spec.detect_conn_by_bo(reaxff = self.reaxff)

        # 2. Make a mol object for the TS
        mol_ts = molsys.mol.from_file(ts_path)
        mol_ts.detect_conn_by_bo(reaxff = self.reaxff)

        # 3. Get the matching indices
        ratom, rbond_btw, vts2vspec = self._match_wrt_ts(mol_spec, mol_ts, label, n, rbonds, atom_ids_dict)

        # 4. Order the species wrt TS
        mol_str = '%d\n\n' %mol_ts.natoms
        for vts in sorted(vts2vspec):
            vspec = vts2vspec[vts]
            atom = mol_spec.elems[vspec]
            x = mol_spec.xyz[vspec][0]
            y = mol_spec.xyz[vspec][1]
            z = mol_spec.xyz[vspec][2]
            mol_str += '%s %5.6f %5.6f %5.6f\n' %(atom,x,y,z)
        mol_ordered = molsys.mol.from_string(mol_str,'xyz')
        return mol_ordered


    def make_rxn_complex(self, rbonds, atom_ids_dict, label, n_eq, QM_path, ts_path, distance = 3.0):
        """ This method adds the 2nd species based on the internal coordinates of the reference TS; e.g. ReaxFF optimized TS,
            at a distance (3.0 Angstrom by default) to the 1st species using Z-matrices.
            
            TODO Again, probably giving the mol objects as an input might be a good idea. Maybe also as a dictionary with
                 the same keys as in atoms_ids_dict. 
                 n_eq is an unnecessary argument.
                 There is space for improvement.

        Args:
            rbonds           : list of reactive bonds; e.g. [0,2,5,7]
            atoms_ids_dict   : The dictionary which holds the list of atom_ids from the ReaxFF trajectory
            label            : 'educt' or 'product'
            n_eq             : the number of equilibrium species with the <label> 
                               -> This is kind of a stupid argument, it can only handle n_eq = 2, where was my mind?...
            QM_path          : Where the calculations are being performed. Actually, this is the same as self.path. 
                               I don't know why I wrote it like this. 
            ts_path          : The path to the xyz file of the transition state.
            distance         : The distance between the 1st and 2nd species

        Returns:
            mol_complex      : Mol object of the created reaction complex

        """
        try:
           from molsys.addon import zmat
        except:
           print("The module zmat from molsys.addon could not be imported. Be sure chemcoord is available.")
        if n_eq > 3:
            print('Cannot handle more than two species. Exiting...')
            sys.exit()
        elif n_eq < 2:
            print('You cannot create a complex with a single molecule. Exiting...')
            sys.exit()

        # 1. Make a mol object for the TS
        mol_ts = molsys.mol.from_file(ts_path)
        mol_ts.detect_conn_by_bo(reaxff = self.reaxff)

        # 2. Loop over the equilibrium species
        mols_opt = []
        for n in range(1, n_eq+1):
           # a) Create a mol object of the DFT optimized species
           path_opt = os.path.join(QM_path,"%s_%d" %(label,n))
           mol_opt = GeneralTools(path_opt).coord_to_mol()
           mol_opt.detect_conn_by_bo(reaxff = self.reaxff)
           mols_opt.append(mol_opt)

        ns = [1,2]
        if len(mols_opt[0].elems) < len(mols_opt[1].elems):
           mols_opt[0],mols_opt[1] = mols_opt[1],mols_opt[0]
           ns = [2,1]

        print(mols_opt)

        i_mol = 0
        for n, mol_opt in zip(ns,mols_opt):
           i_mol += 1
           print('\n================')
           print('    %s_%d' %(label,n))
           print('================')
           natoms = len(mol_opt.elems)
           print('natoms',natoms)

           # 3. Get the matching indices
           ratom, rbond_btw, vts2vopt = self._match_wrt_ts(mol_opt, mol_ts, label, n, rbonds, atom_ids_dict)
           iopt2its = {}
           vopt2vts = {}
           for vts in vts2vopt:
               vopt = vts2vopt[vts]
               vopt2vts[vopt] = vts
               iopt2its[vopt+1] = vts+1

           if i_mol == 1: vts2vopt_1 = vts2vopt

           # 4. Make the Z-matrix of the optimized species
           # 1st species
           if i_mol == 1:
               # 5.a) Build construction table for the 1st species
               xyz_1 = zmat(mol_opt).cart
               # b) Form the Z-matrix of the 1st species (to replace the internal coordinates later)
               zmat_opt_1 = xyz_1.get_zmat()
               # c) Change indices to that of the TS using iopt2its dictionary
               const_table_1 = xyz_1.get_construction_table()
               const_table_1 = const_table_1.replace(iopt2its)
               new_index = [iopt2its[iopt] for iopt in const_table_1.index]
               const_table_1.index = new_index
               if self.verbose: print('zmat_opt_1',zmat_opt_1)

           #  2nd species
           elif i_mol == 2:
               # 5.a) Build the construction table for the 2nd species
               xyz_2 = zmat(mol_opt).cart
               const_table_2 = xyz_2.get_construction_table()
               # b) Make sure that the Z-matrix of the 2nd species starts with the reacting atom on the 2nd species
               first_atom_idx = const_table_2.index[0]
               ratom_idx = vts2vopt[ratom]+1
               if first_atom_idx != ratom_idx:
                   print('The Z-matrix will be modified to have the reacting atom %d on %s_%d as the first.' %(ratom+1,label,n))
                   for i, idx in enumerate(const_table_2.index):
                       if idx == ratom_idx:
                          ratom_pos = i
                   new_index = list(const_table_2.index)
                   new_index[0] = ratom_idx
                   new_index[ratom_pos] = first_atom_idx
                   idx2newidx = {}
                   for i, idx in enumerate(new_index):
                       idx2newidx[const_table_2.index[i]] = idx
                   const_table_2.index = new_index
                   const_table_2 = const_table_2.replace(idx2newidx)
               # c) Form the Z-matrix of the 2nd species (to replace the internal coordinates later)
               zmat_opt_2 = xyz_2.get_zmat(const_table_2)
               # d) Change indices to that of the TS
               const_table_2 = const_table_2.replace(iopt2its)
               new_index = [iopt2its[iopt] for iopt in const_table_2.index]
               const_table_2.index = new_index
               if self.verbose: print('zmat_opt_2',zmat_opt_2)

               # 6. Build the construction table of the complex/transition state

               # a) Append the construction table of the 2nd species to that of the 1st species.
               const_table = const_table_1.append(const_table_2)

               # b) Replace the empty references based on the connectivity of the TS structure.

               # 1st atom of the 2nd molecule
               # ----------------------------

               # BOND:
               ratom_on_mol1 = list(set(rbond_btw)-{ratom})[0]
               const_table.loc[ratom+1,'b'] = ratom_on_mol1+1 
               
               # ANGLE:
               for i in mol_ts.conn[ratom_on_mol1]:
                    # check if the connected one is a terminal atom, because if so, assigning the atom which will define the dihedral is difficult...
                    terminal_atom = False
#                    print('i',i)
#                    print('mol_ts.conn', mol_ts.conn)
#                    print('mol_ts.conn[i]',mol_ts.conn[i])
                    if len(mol_ts.conn[i]) == 1: terminal_atom = True 
#                    print('terminal_atom',terminal_atom)
                    if i != ratom_on_mol1  and i != ratom and not terminal_atom: 
                        a = i
               try:
                    const_table.loc[ratom+1,'a'] = a + 1
               except:
                    print('The connected atom to the reacting atom %d different than the other reacting atom %d, and which is not a terminal atom could not be assigned.' %(ratom_on_mol1+1, ratom+1))
                    sys.exit()
              
               # DIHEDRAL:
               for i in mol_ts.conn[a]:
                    if i != ratom_on_mol1 and i != ratom and i != a: 
                        d = i
               try:
                    const_table.loc[ratom+1,'d'] = d + 1
               except:
                    print('The connected atom to the atom %d, different than the reactive atom %d could not be assigned.' %(a+1,ratom+1))
                    sys.exit()

               # 2nd atom of the 2nd molecule
               # ----------------------------
               if natoms >= 2:
                   idx = const_table_2.index[1]
                   const_table.loc[idx,'a'] = a + 1
                   const_table.loc[idx,'d'] = d + 1

               # 3rd atom of the 2nd molecule
               # ----------------------------
               if natoms >= 3:
                   idx = const_table_2.index[2]
                   const_table.loc[idx,'d'] = d + 1

               # 7. Construct the Z-matrix of the TS using the construction table created
               xyz_ts = zmat(mol_ts).cart
               zmat_complex = xyz_ts.get_zmat(const_table)
               if self.verbose: print('zmat_ts',zmat_complex)

               # 8. Replace all of the coordinates for the 1st molecule
               for i in const_table_1.index:
                   if zmat_complex.loc[i,'atom'] != zmat_opt_1.loc[vts2vopt_1[i-1]+1,'atom']:
                      print('Something went wrong with matching the atom indices. Exiting...')
                      sys.exit()
                   zmat_complex.safe_loc[i,'bond'] = zmat_opt_1.loc[vts2vopt_1[i-1]+1,'bond']
                   zmat_complex.safe_loc[i,'angle'] = zmat_opt_1.loc[vts2vopt_1[i-1]+1,'angle']
                   zmat_complex.safe_loc[i,'dihedral'] = zmat_opt_1.loc[vts2vopt_1[i-1]+1,'dihedral']
               # 9. Replace the coordinates independent coordinates of the 2nd molecule
               for i in const_table_2.index:
                   if zmat_complex.safe_loc[i,'atom'] != zmat_opt_2.loc[vts2vopt[i-1]+1,'atom']:
                      print('Something went wrong with matching the atom indices. Exiting...')
                      sys.exit()
                   # first atom
                   if i == const_table_2.index[0]:
                       zmat_complex.safe_loc[i,'bond'] = distance
                   # second atom
                   elif i == const_table_2.index[1]:
                       zmat_complex.safe_loc[i,'bond'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'bond']
                   # third atom
                   elif i == const_table_2.index[2]:
                       zmat_complex.safe_loc[i,'bond'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'bond']
                       zmat_complex.safe_loc[i,'angle'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'angle']
                   # rest of the 2nd molecule
                   else:
                       zmat_complex.safe_loc[i,'bond'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'bond']
                       zmat_complex.safe_loc[i,'angle'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'angle']
                       zmat_complex.safe_loc[i,'dihedral'] = zmat_opt_2.loc[vts2vopt[i-1]+1,'dihedral']

               if self.verbose: print('zmat_complex',zmat_complex)

               # 10. Convert the Z-matrix of the complex back to the carte
               xyz_complex = zmat_complex.get_cartesian()
               print('The complex is succesfully created.')

               # 11. Now make a mol object which has the same order as in the ts
               mol_str = '%d\n\n' %mol_ts.natoms
               for i in range(mol_ts.natoms):
                   atom = xyz_complex.loc[i+1, 'atom']
                   x = xyz_complex.loc[i+1, 'x']
                   y = xyz_complex.loc[i+1, 'y']
                   z = xyz_complex.loc[i+1, 'z']
                   mol_str += '%s %5.6f %5.6f %5.6f\n' %(atom,x,y,z)
               mol_complex = molsys.mol.from_string(mol_str,'xyz')
        return mol_complex 


    def find_end_point_from_IRC(self, IRC_path = '', displaced = 'minus'):
        """ Optimizes the end points of intrinsic reaction coordinates (IRC) as found in the directories 
        'displaced_minus' and 'displaced_plus', then separates the molecules and returns the 
        corresponding mol objects.
        
        Args:
            IRC_path  : The path to the directory where the results from the IRC calculation is.
            displaced : Which side do you want to optimize, plus or minus?
            gcart     : converge maximum norm of cartesian gradient up to 10^-<gcart> atomic units

        Returns:
            mols      : The list of mol objects from the geometry optimized end point (which are not individually optimized)
            QM_path   : The path to the directory where this calculation is performed

        """
        if IRC_path == '': IRC_path = self.path

        # 1. Make a sub directory for the optimization
        path = os.path.abspath(os.path.join(IRC_path, 'displaced_%s' %displaced))
        os.chdir(path)
        os.system('cpc %s' %displaced)
        QM_path = os.path.join(path, displaced)

        OT = OptimizationTools(QM_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        GT = GeneralTools(QM_path)

        # 2. Remove the gradient to not to use the TS
        GT.kdg('grad')
        grad_file = os.path.join(QM_path,'gradient')
        if os.path.isfile(grad_file): os.remove(grad_file)

        # 3. Define the internal coordinates
        OT.ired()

        # 4. Optimize the geometry
        converged = OT.jobex()
        if not converged:
            print('The geometry optimization of the end point of IRC has failed.')
            sys.exit()

        # 5. Get the molecules at the end points
        mol = GT.coord_to_mol()
        mols, labels = self.separate_molecules(mol)

        # 6. Add the $frag data group to the coordinate file 
        if len(mols) > 1:
            OT.add_frag_to_coord(labels=labels)
        
        # 7. Go to the main directory
        os.chdir(self.maindir)

        return mols, QM_path

    def add_frag_to_coord(self, coord_dir='', labels=[]):
        if coord_dir == '': coord_dir = self.path
        to_coord = '$frag\n'
        for i,frag in enumerate(labels):
            to_coord += '       atom  %d  fragment  %d\n' %(i+1,frag+1)
        newlines = ''
        coord_path = os.path.join(coord_dir, 'coord')
        with open(coord_path) as coord:
            for line in coord:
                 if '$end' in line:
                     newlines += to_coord
                     newlines += '$end'
                 else:
                     newlines += line
        print(newlines)
        f = open(coord_path,'w')
        f.write(newlines)
        f.close()
        return

    def compare_mols(self, mols_1, mols_2, mode = 'mg'):
        """ Compares the list of reference molecules (mols_1) with the list of molecules to compare (mols_2).
        
        Args:
             mols_1, mols_2: List of mol objects of the molecules to be compared
             mode          : Comparison mode:
                             'mg' -> Compare using molecular graph isomorphism
                             'similaritycheck' -> Compare based on RMSD distances (https://doi.org/10.1002/jcc.21925)

        Returns:
             mols_similar: True/False
             index_dict  : If they are similar, the mapping of the similar molecules.
                           e.g.,
                           mols_1  mols_2
                           ---------------
                             0 ----> 1
                             1 ----> 2
                             2 ----> 0
                           index_dict = {0:1,1:2,2:0}
        """
        mols_similar = True # similarity of the list of molecules
        index_dict = {}
        n_mols_1 = len(mols_1)
        n_mols_2  = len(mols_2)        
        if n_mols_1 != n_mols_2:
            mols_similar = False
        else:
            # Loop over the reference molecules
            for i,mol_1 in enumerate(mols_1):

                mol_similar = False # similarity of individual molecules

                if mode == 'mg':
                    mg_1 = self.make_molecular_graph(mol_1)

                # Loop over the molecules to compare
                for j,mol_2 in enumerate(mols_2):

                    if mode == 'mg':
                        mg_2 = self.make_molecular_graph(mol_2)
                        is_equal = molsys.addon.graph.is_equal(mg_1, mg_2, use_fast_check=False)[0]
                    elif mode == 'similaritycheck':
                        is_equal = GeneralTools().similaritycheck_from_mol(mol_1, mol_2)
                    else:
                        print('Please specify a valid comparison method!')
                        sys.exit()

                    if is_equal:
                        index_dict[i] = j

                    mol_similar = mol_similar or is_equal

                # if all of the reference molecules were similar to one of the molecules to compare, it is True
                mols_similar = mols_similar and mol_similar

        return mols_similar, index_dict


    def check_end_points(self, mols_minus, mols_plus, mols_ed, mols_prod, mode = 'mg'):
        """
        Compares the molecular graphs of the output of the IRC calculation to those of reference structures.
        Basically this is used to check if the molecular graph of the reaction has changed or not.

        Args:
            mols_minus    : List of mol objects created by separating the molecules from IRC output, displaced_minus
            mols_plus     : List of mol objects created by separating the molecules from IRC output, displaced_plus
            mols_ed       : List of mol objects of the reference educts   (e.g. from ReaxFF optimized structures)
            mols_prod     : List of mol objects of the reference products (e.g. from ReaxFF optimized structures)
            mode          : The comparison mode: 
                            'mg'              -> molecular graph, 
                            'similaritycheck' -> mapping the structures in 3D and calculating RMSD (https://doi.org/10.1002/jcc.21925)

        Returns:
            is_similar    : True/False 
                            mode = 'mg' -> Did the reaction graph change? 
                            mode = 'similaritycheck' -> Are the structures the same at the end points of the reaction?
            match         : if is_similar: the mapping of plus and minus paths to the educts and products
                            e.g., match = {'minus':'educt', 'plus':'product'}
            index_dict    : the mapping of the molecules on the irc end points to the reference educts and products
                            e.g., index_dict = {'minus':{0:1,1:0}, 'plus':{0:0,1:1}}
        """
        is_similar = False
        match = {}
        index_dict = {}
        n_mol_minus  = len(mols_minus)
        n_mol_plus   = len(mols_plus)
        n_mol_educts = len(mols_ed)
        n_mol_products = len(mols_prod)
        if (n_mol_minus == n_mol_educts and n_mol_plus == n_mol_products) or (n_mol_minus == n_mol_products and n_mol_plus == n_mol_educts):

            # 1. Compare the educts with minus and plus from IRC
            minus_ed_is_similar, index_dict_minus_ed  = self.compare_mols(mols_minus, mols_ed, mode = mode)
            plus_ed_is_similar , index_dict_plus_ed   = self.compare_mols(mols_plus,  mols_ed, mode = mode)

            # 2. Compare the products with minus and plus from IRC
            minus_prod_is_similar, index_dict_minus_prod = self.compare_mols(mols_minus, mols_prod, mode = mode)
            plus_prod_is_similar , index_dict_plus_prod  = self.compare_mols(mols_plus,  mols_prod, mode = mode)

            if self.verbose:
                print("minus_ed_is_similar,minus_prod_is_similar,plus_ed_is_similar,plus_prod_is_similar", minus_ed_is_similar,minus_prod_is_similar,plus_ed_is_similar,plus_prod_is_similar)
                case1 = minus_ed_is_similar and plus_prod_is_similar
                case2 = plus_ed_is_similar and minus_prod_is_similar
                print('minus_ed_is_similar and plus_prod_is_similar =', case1)
                print('plus_ed_is_similar and minus_prod_is_similar =', case2)

            if (minus_ed_is_similar and plus_prod_is_similar):
                is_similar = True
                match['minus']      = 'educt'
                index_dict['minus'] =  index_dict_minus_ed
                match['plus']       = 'product'
                index_dict['plus']  =  index_dict_plus_prod
            elif (plus_ed_is_similar and minus_prod_is_similar):
                is_similar = True
                match['minus']      = 'product'
                index_dict['minus'] =  index_dict_minus_prod
                match['plus']       = 'educt'
                index_dict['plus']  =  index_dict_plus_ed
        return is_similar, match, index_dict


    ### TOOLS TO OPTIMIZE REACTION PATH AND ANALYSE THEM ######################################

    def get_peaks(self, path, plot = False):
        """Gets the peaks from the woelfling output

        Args:
            path : string  : The path to where the woelfling calculation have been performed.
            plot : boolean : True if you want to plot the energy profile
        
        Returns:
            peaks_woelf : List of peaks based on indexing -> 1,2,3,...
                          so that one can directly go into directory 'rechnung_<peak>' for further calculations
            barrierless : True/False

        """
        barrierless = False

        f_woelfling_out = os.path.join(path,"woelfling_current.out")
        if not os.path.isfile(f_woelfling_out):
            print('No woelfling calculations have been performed under this directory.')
            sys.exit()
        else:
            x = [] # structure number
            y = [] # energy profile
            with open(f_woelfling_out) as woelfling_out:
               energy_profile = {}
               for lines in woelfling_out:
                   if 'structure ' in lines:
                       line = lines.strip().split()
                       energy   = float(line[5])
                       struc_no = int(line[1])
                       y.append(energy)
                       x.append(struc_no)
                       energy_profile[struc_no] = energy
                       
            if plot:
                plt.plot(x,y)
                plt.ylabel('Energy Profile (Hartree)')
                plt.savefig(os.path.join(path,'energy_profile.pdf'), format='pdf')


            from scipy.signal import find_peaks
            peaks, _ = find_peaks(y) # This function takes a 1-D array and finds all local maxima by simple comparison of neighboring values. 
            peaks_woelf = []
            if len(peaks) == 0:
                barrierless = True
            elif len(peaks) >  1: 
                print('WARNING!!! There are multiple local maxima!') 
                print('The local maxima:')
                for peak in peaks:
                    print(peak+1)
                    peaks_woelf.append(peak+1)
            else:
                peaks_woelf.append(peaks[0]+1)

        return peaks_woelf, barrierless

    def add_woelfling_to_control(self, control_path = 'control', ninter = 24, ncoord = 2, maxit = 40):
        """
        Args:
            control_path : the path to the control file
            ninter       : the number of structures to be used for discretization of the path
            ncoord       : the number of input structures provided
            maxit        : the number of cycles to run
        """
        newlines = ''
        with open(control_path) as control:
           for line in control:
               if '$end' in line:
                   newlines += '$woelfling\n ninter  %d\n riter   0\n ncoord  %d\n align   0\n maxit   %d\n dlst    3.0\n thr     1.0E-4\n method  q\n$end' %(ninter, ncoord, maxit)
               else:
                   newlines += line
        f = open(control_path,'w')
        f.write(newlines)
        f.close()
        return

    def woelfling_workflow(self, woelfling_path = '', ninter = 24, ncoord = 2, maxit = 40):
        """workflow to perform a TS search using woelfling
        Args:
            woelfling_path : the path to the directory of where the woelfling calculation will be performed
            ninter         : the number of structures to be used for discretization of the path
            ncoord         : the number of input structures provided
            maxit          : the number of cycles to run
 
        Returns:
            barrierless : True/False barrierless reaction?
            ts_paths    : The list of paths to the directories where the input for the further calculations
                          to find the transition state structures (ts_workflow: aoforce, jobex -trans, aoforce, IRC, ...) are.
        """
        if woelfling_path == '': woelfling_path = self.path
        assert os.path.isfile(os.path.join(woelfling_path,'coords')), 'Please provide coords file!'
        control_path = os.path.join(woelfling_path,'control')
        assert os.path.isfile(control_path), 'Please provide a control file!'

        # 1. Add woelfling group to the control file
        self.add_woelfling_to_control(control_path = control_path, ninter = ninter, ncoord = ncoord, maxit = maxit)

        # 2. Perform the woelfling calculation
        if not os.path.isdir(woelfling_path):
            os.mkdir(woelfling_path)
        os.chdir(woelfling_path)
        # NOTE Before TM7.6 the energy of the initial and final point is calculated in each iteration. 
        # Starting with already converged orbitals might cause convergence problems in the SCF cycle with ARH solver.
        os.system('woelfling-job -ri > woelfling.out')
        with open('woelfling.out') as out:
            for line in out:
                if 'dscf_problem' in line or 'scf energy ended abnormally' in line:
                    if os.path.isfile(os.path.join(woelfling_path,'path-20.xyz')):
                        print('The SCF did not converge in one of the structures of woelfling calculation. But there were at least 20 iterations. So, a TS will be searched...')
                    else:
                        self._clean_woelfling(woelfling_path)
                        print('The SCF did not converge in one of the structures of woelfling calculation. Exiting...')
                        sys.exit()

        # 3. If exists, get the peaks on the woelfling energy profile to use later as a TS start guess
        try:
            peaks, barrierless = self.get_peaks(path = woelfling_path, plot = True)
        except:
            peaks, barrierless = self.get_peaks(path = woelfling_path, plot = False)

        ts_paths = []
        if not barrierless:
            for peak in peaks:
                # Get into the corresponding directory and calculate its Hessian
                peak_path = os.path.join(woelfling_path, 'rechnung-%d' %(peak))
                ts_path = os.path.join(peak_path, 'ts')
                os.chdir(peak_path)
                os.system('cpc %s' %ts_path)
                self.ired(path = ts_path)
                ts_paths.append(ts_path)
                os.chdir(self.maindir)

        # Clean the directories 
        self._clean_woelfling(woelfling_path)

        return barrierless, ts_paths


    def _clean_woelfling(self, woelfling_path):
        """ Clean the mos from the woelfling and rechnung-* directories """
        lst = os.listdir(woelfling_path)
        for f in lst:
            f_path = os.path.join(woelfling_path, f)
            #if f in ["alpha", "beta", "mos", "wherefrom","statistics"]:
            if f in ["wherefrom","statistics"]:
                os.remove(f_path)
            if "rechnung" in f:
                lst_rechnung_X = os.listdir(f_path)
                for f_rechnung_X in lst_rechnung_X:
                    if f_rechnung_X in ["alpha", "beta", "mos", "wherefrom","statistics"]:
                        os.remove(os.path.join(f_path, f_rechnung_X))
        return


    def ts_pre_optimization(self, mol, M, rbonds, gcart = 3, add_noise = True, upto = 0.05):
        """ Generates MOs by applying Fermi smearing and changing the occupations to the assigned multiplicity. 
            Then fixes the the internal coordinates between the atoms involved in bond-order change and optimizes the structure.
            Sets the itvc = 1 after the pre-optimization.

        Args:
            mol       : the mol object of the structure to be pre-optimized
            M         : the multiplicity of the TS/reaction
            rbonds    : the list of reactive bonds; e.g. [0,1,3,7]
            gcart     : converge maximum norm of cartesian gradient up to 10^-<gcart> in a.u.
            add_noise : Do you want to add a noise to the structure before the optimization?
            upto      : The maximum noise to be added to the structure in Angstrom

        Returns:
            converged : True/False: Did the geometry optimization converge?
            QM_path   : The path to the directory where the pre-optimization is performed.

        """
        # indexing of active_atoms goes as 1,2,3,... whereas that of rbonds goes as 0,1,2,...
        active_atoms = []
        for i in set(rbonds):
            active_atoms.append(i+1)

        # create a QM path for the pre-optimization
        QM_path = os.path.join(self.path, 'ts_M%d' %M)
        os.mkdir(QM_path)
        OT = OptimizationTools(QM_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        GT = GeneralTools(QM_path)

        # Add noise to the structure and perform the single point energy calculation
        energy = OT.generate_MOs(mol = mol, pre_defined_M = True, fermi = True, M = M, active_atoms = active_atoms, add_noise = add_noise, upto = 0.05)

        # TODO Fix this in tmole... Tmole does not change maxcor
        OT.change_maxcor()

        # freeze the active atoms
        OT.freeze_atoms(active_atoms)

        # pre-optimization
        converged = OT.jobex(gcart = gcart)

        if converged:
            # remove the gradient left from pre-optimization
            GT.kdg('grad')
            grad_file = os.path.join(QM_path, 'gradient')
            if os.path.isfile(grad_file):  os.remove(grad_file)


        # define internal coordinates without constraints and set itvc to 1.
        GT.kdg('intdef')
        GT.kdg('redundant')
        OT.ired_and_itvc_1()

        return converged, QM_path


    def _optimize_irc_end_points(self, M, displaced = '', mols_QM_ref = [], match = {}, index_dict = {}, irc_mols = {}, irc_path = {}, is_similar = False):
        """ Optimizes the end points of the IRC calculation

        Args:
            M              : Multiplicity of the reaction/transition state
            displaced      : 'minus' or 'plus'?
            mols_QM_ref    : the list of mols objects of the reference QM species: used to compare multiplicities, because at the 
                             educts_and_products workflow, multiplicity attribute is added to the mol object.
                             -> If reaction graph is similar; then the multiplicity of the reference species 
                                and the optimised species are compared. If the multiplicities differ, then this is printed.
            match          : the mapping of plus and minus paths to the educts and products
                             e.g., match = {'minus':'educt', 'plus':'product'}
            index_dict     : the mapping of the molecules on the irc end points to the reference educts and products
                             e.g., index_dict = {'minus':{0:1,1:0}, 'plus':{0:0,1:1}}
            irc_mols       : the list of (separated) mol objects from the irc end points
            irc_path       : the path to the directory where the irc calculation has been performed
            gcart          : Threshold for geometry optimization, converge maximum norm of cartesian gradient up to 10^(-gcart) atomic units.
            is_similar     : True/False 

        Returns:
            QM_paths_dict  : Dictionary of the path to the directories where the optimized end points are.

        """
        QM_paths_dict = {}
        if is_similar:
            label = match['%s' %displaced]   # 'educt' or 'product'
            irc2ref  = index_dict['%s' %displaced] # dictionary which matches of e.g., QM_reference educts to the corresponding IRC species
            if self.verbose:
                print('_optimize_irc_end_points')
                print('------------------------')
                print('index_dict', index_dict)
                print('label', label)
                print('irc2ref', irc2ref)
        else:
            if displaced == 'minus':
                label = 'educt'
            elif displaced == 'plus':
                label = 'product'
        # Get the mols and the path to the end points
        mols      = irc_mols[displaced]  # the separated educt mol objects from the irc end points, those we want to optimize now
        path      = irc_path[displaced] # the pathway to the .../displaced_minus/minus or .../displaced_plus/plus directories 

        OT = OptimizationTools(path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        GT = GeneralTools(path)

        # If there is only one molecule just calculate just the Hessian 
        if len(mols) == 1:
            mol_opt = mols[0]
            mol_opt.multiplicity = M # The multiplicity is the same as the TS, as the same orbitals are used with constant occupations
            OT.aoforce()
            inum, imfreq = OT.check_imaginary_frequency()
            if inum != 0:
                OT.disturb_and_reoptimize(natoms=mol_opt.natoms)
            else:
                key = '%s_1' %(label)
                QM_paths_dict[key] = path

        # If there is more than one, then optimize them separately starting only from the coordinates.
        else:
            print("The number of molecules is not 1...")
            OT.add_dg_to_control("frag file=coord")
            OT.proper_frag()
            for i,mol in enumerate(mols):
                key = '%s_%d' %(label, i+1)
                QM_path = os.path.join(path,key)
                os.mkdir(QM_path)
                OT.prepare_frag_input(i+1,QM_path)
                OT_opt = OptimizationTools(QM_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
                OT_opt.jobex()
                OT_opt.aoforce()
                inum, imfreq = OT_opt.check_imaginary_frequency()
                if inum != 0:
                    OT_opt.disturb_and_reoptimize(natoms=mols[i].natoms)
                else:
                    QM_paths_dict[key] = QM_path

        return QM_paths_dict


    # TODO The path to the proper should be changed to link that set from the TURBODIR
    def proper_frag(self, proper_path="proper"):
        """ produces for each fragment 5 files: frag1.xyz, control1, coord1, alpha1, beta1
        """
        # TODO Delete
        # from here
        # This part is not necessary but added as a temporary solution due to a minor bug in the release version of Turbomole 7.6.
        control_path = os.path.join(self.path,"control")
        atomsdg = False
        newatomsdg = 'atoms\n'
        basisadded = False
        jbasadded  = False
        for line in open(control_path,'r'):
            if '$atoms' in line:
                atomsdg = True
            elif '$' in line:
                atomsdg =  False
            elif atomsdg:
                if 'basis' in line and not basisadded:
                    newatomsdg += '    basis = %s    ' %line.split()[2]
                    basisadded = True
                elif 'jbas' in line and not jbasadded:
                    newatomsdg += '\ \n    jbas  = %s    ' %line.split()[2]
                    jbasadded = True
        GT = GeneralTools(self.path)
        GT.kdg('atoms')
        self.add_dg_to_control(newatomsdg)
        # until here ---
        os.chdir(self.path)
        proper_in = os.path.join(self.path,'proper.in')
        f=open(proper_in,'w')
        f.write('mos\nfrag\nq')
        f.close()
        os.system('%s < proper.in > proper_frag.out' %proper_path)
        os.chdir(self.maindir) 
        return

    def prepare_frag_input(self, i, QM_path):
        os.chdir(self.path)
        OT_QM = OptimizationTools(QM_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        # step 1: Move the input files created by proper for that fragment
        for f in ['control','coord','alpha','beta']:
            shutil.move('%s%d' %(f,i), os.path.join(QM_path,f))
        newlines = ''
        control_path = os.path.join(QM_path,'control') 
        with open(control_path) as control:
            for line in control:
                line_added = False
                for f in ['coord','alpha','beta']:
                    if '%s%d' %(f,i) in line:
                        sline =  line.split('%s%d' %(f,i))
                        newlines += sline[0]+f+sline[-1]
                        line_added = True
                        break
                if not line_added:
                    newlines += line
        f = open(control_path,'w')
        f.write(newlines)
        f.close()

        # step 2: Copy the data existing groups from the reference calculation to the new one
        control_ref = os.path.join(self.path,'control')
        with open(control_ref) as control:
            text = control.read()
        datagroups = text.split('$')
        for dg in datagroups:
            for s in ['ri','disp','dft','scfiterlimit','maxcor']:
                if dg.startswith(s):
                    OT_QM.add_dg_to_control(dg.strip())
   
        # step 3: Define the internal redundant coordinates
        define_in = os.path.join(QM_path, 'define.in')
        f = open(define_in, 'w')
        f.write('\n')      # title
        f.write('y\n')
        f.write('ired\n')  # assign internal redundant coordinates
        f.write('*\n')    
        f.write('qq\n')    # exit define
        f.close()
        os.chdir(QM_path)
        os.system('define < define.in > define.out')
        os.chdir(self.maindir)
        return


    ### THE MAIN WORKFLOWS #########################################################

    def ts_workflow(self, QM_path_ts, mols_ed_QM_ref, mols_prod_QM_ref, M = None, mode = 'mg'):
        """ The workflow for the TS. 

            NOTE! under data group $statpt in the control file, itrvec should be set to 1 before using this method!

        Args:
            QM_path_ts       : the path to the directory where the transition state calculation will be performed
            mols_ed_QM_ref   : the list of mol objects of the reference educts (should have multiplicity as an attribute)
            mols_prod_QM_ref : the list of mol objects of the reference products (should have multiplicity as an attribute)
            M                : multiplicity of the transition state
            mode             : Comparison mode to compare the reaction graph
                               'mg' -> Compare using molecular graph isomorphism
                               'similaritycheck' -> Compare based on RMSD distances (https://doi.org/10.1002/jcc.21925)

        Returns:
            converged        : True/False: Did the geometry optimization converge and there is only one imaginary frequency?
            is_similar       : True/False: Did the reaction graph change?
            QM_paths_dict    : the dictionary of paths to the species connecting the educts and the products
            reason           : string : Why it did not converge?
        """
        QM_paths_dict  = {}
        converged = False
        is_similar = False
        reason = ''
        OT = OptimizationTools(QM_path_ts, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)

        # 1. Calculate the Hessian
        OT.aoforce()
        
        # 2. Check the number of imaginary frequencies
        inum, imfreq = OT.check_imaginary_frequency()

        if inum == 0:        
            reason += 'No imaginary frequency at the start structure.'
        else:
            if inum > 1:
                reason += 'There are more than one imaginary frequencies at the start structure. But it will try to optimize.'
                print(reason)

            # 3. Optimize the TS with eigenvector following
            converged = OT.jobex(ts=True)
 
            if not converged:
               reason += 'The transition state optimization did not converge.'

            else:
               # 4. Calculate Hessian and check if it is a saddle point
               OT.aoforce()
               inum, imfreq = OT.check_imaginary_frequency()

               if inum == 1: # if a saddle point
                  print('There is only one imaginary frequency. The intrinsic reaction coordinate is being calculated.')

                  # 5. Perform intrinsic reaction coordinate calculation
                  OT.IRC()
                  irc_mols = {}
                  irc_path = {}

                  # 6. Optimize the end points (without separating the molecules)
                  irc_mols['minus'], irc_path['minus'] = OT.find_end_point_from_IRC(displaced = 'minus')
                  irc_mols['plus' ], irc_path['plus' ] = OT.find_end_point_from_IRC(displaced = 'plus') 

                  # 7. Compare the end points with the educts and products
                  is_similar, match, index_dict = OT.check_end_points(irc_mols['minus'], irc_mols['plus'], mols_ed_QM_ref, mols_prod_QM_ref, mode = mode)

                  if not is_similar: reason += "This transition state does not connect the reference educts and products."

                  # 8. Optimize the end points of the separated molecules
                  QM_paths_dict_min = self._optimize_irc_end_points(M = M, displaced = 'minus', mols_QM_ref = mols_ed_QM_ref,   match = match, index_dict = index_dict, irc_mols = irc_mols, irc_path = irc_path, is_similar = is_similar)
                  QM_paths_dict_plus = self._optimize_irc_end_points(M = M, displaced = 'plus',  mols_QM_ref = mols_prod_QM_ref, match = match, index_dict = index_dict, irc_mols = irc_mols, irc_path = irc_path, is_similar = is_similar)

                  QM_paths_dict = {**QM_paths_dict_min, **QM_paths_dict_plus}
                  QM_paths_dict['ts'] = QM_path_ts

               elif inum == 0:
                  converged = False
                  reason += 'There are no imaginary frequencies.'

               elif inum > 1:
                  converged = False
                  reason += 'There are still more than one imaginary frequencies.'
        OT.t2x()

        return converged, is_similar, QM_paths_dict, reason


    def change_maxcor(self, frac_max_mem = 0.7):
        os.chdir(self.path)
        os.system("kdg maxcor")
        os.system("kdg end")
        os.system('''echo "\$maxcor    %d MiB  per_core" >> control''' %(frac_max_mem*self.max_mem))
        os.system('''echo "\$end" >> control''')
        os.chdir(self.maindir)
        return


    def educts_and_products_workflow(self, mols, add_noise = True, upto = 0.05, label = 'eq_spec'):
        """ Optimizes the equilibrium species by first determining the ground state multiplicity through comparison of energies by {M_Fermi, M_Fermi+-2}.
            
            This method is meant to be called from the reaction_workflow method. But can also be called separately.

            TODO 1: Add an option to start with eht orbitals.
            TODO 2: Add an option to start with superposition of atomic densities.

            Args:
                mols      : The mol objects of the structures to be optimised.
                add_noise : True/False: Should a noise be added to the structures?
                upto      : How much maximum noise should be added to the structures? (in Angstrom)
                label     : Label to name the directories; e.g., 'product', 'educt', etc...

            Returns:
                multiplicities : the list of multiplicities of the optimised species
                QM_paths       : the list of paths to the directories of where the calculations are done
                mols_opt       : the mol objects of the optimised species
        """
        multiplicities = []
        QM_paths       = []
        mols_opt        = []
        for i, mol_ini in enumerate(mols):
            QM_path = os.path.join(self.path, '%s_%d' %(label,i+1))
            os.mkdir(QM_path)
            OT = OptimizationTools(QM_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
            GT = GeneralTools(QM_path)
            atom = False
            if mol_ini.natoms == 1: atom = True

            # 1. Make molecular graph of the reference structure
            self.make_molecular_graph(mol_ini)
            mg_ini = mol_ini.graph.molg            

            # 2. Add noise to the structure and perform the single point energy calculation
            energy = OT.generate_MOs(mol = mol_ini, add_noise = add_noise, upto = upto, lowest_energy_M = True)

            # TODO This should be fixed in tmole... It doesn't modify the memory correctly
            OT.change_maxcor() 

            # 3. Get the multiplicity
            nalpha, nbeta = GeneralTools(QM_path).get_nalpha_and_nbeta_from_ridft_output()
            M = abs(nalpha-nbeta)+1

            if atom:
                mol_opt = GT.coord_to_mol()
            else:
                # 4. Perform the geometry optimization
                converged = OT.jobex()
                if not converged:
                    print('The geometry optimization did not converge for %s_%d.' %(label,i+1))
                    exit()
 
                # 5. Make mol object after the geometry optimization
                mol_opt = GT.coord_to_mol()

                # 6. Check if the molecule stays as a single molecule, e.g., the awkward H-O-H-O-H-O complexes of ReaxFF...
                mols, labels =  self.separate_molecules(mol_opt)
                if len(mols) != 1:
                    print('The optimised structure has more than a single molecule. This cannot be handled automatically.')
                    exit()

                # 7. Compare the graph of the initial and optimized structure 
                self.make_molecular_graph(mol_opt)
                mg_opt = mol_opt.graph.molg
                equal = molsys.addon.graph.is_equal(mg_ini, mg_opt)[0]

                # 8. If the graph is different, try to increase multiplicity by two
                if not equal:
                    print('The graph changes. The program will try multiplicity %d.' %(M+2))
                    path_tmp = os.path.join(QM_path, 'M_%d' %(M+2))
                    if os.path.isdir(path_tmp):
                        OT_tmp = OptimizationTools(path_tmp, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
                        GT_tmp = GeneralTools(path_tmp)
                        converged = GT_tmp.check_scf_converged()
                        if not converged:
                             print('The are no converged SCF orbitals with multiplicity %d.' %(M+2))
                             exit()
                        converged = OT_tmp.jobex()
                        if not converged:
                            print('The geometry optimization with multiplicity %d has failed.' %(M+2))
                            exit()
                        mol_tmp = GT_tmp.coord_to_mol()
                        self.make_molecular_graph(mol_tmp)
                        mg_tmp = mol_tmp.graph.molg
                        equal = molsys.addon.graph.is_equal(mg_ini, mg_tmp)[0]
                        if not equal:
                             print('The graph still changes with the multiplicity %d.' %(M+2))
                             exit()
                        else:
                             print('The graph does not change with the multiplicity %d.' %(M+2))
                             mol_opt = mol_tmp
                             files = os.listdir(QM_path)
                             os.mkdir(os.path.join(QM_path,'M_%d' %M))
                             for f in files:
                                 f_to_move = os.path.join(QM_path,f)
                                 if os.path.isfile(f_to_move) and f not in ['submit.py','submit.out']:
                                     shutil.move(f_to_move, os.path.join(QM_path,'M_%d' %M))
                             for f in os.listdir(path_tmp):
                                 shutil.move(os.path.join(path_tmp, f), QM_path)
                             shutil.rmtree(path_tmp)
                             M = M+2

           # 9. Return the final multiplicities and the QM paths of each structure as a list
            OT.t2x()
            mol_opt.multiplicity = M
            multiplicities.append(M)
            QM_paths.append(QM_path)
            mols_opt.append(mol_opt)

        return multiplicities, QM_paths, mols_opt

    def write_coords(self, coords_path, mol_ini, mol_fin):
        """ Writes a coords file which includes concatanated coord files of the two mol objects."""
        coords = open(coords_path,'w')
        coords.write(mol_ini.to_string(ftype="turbo"))
        coords.write('\n')
        coords.write(mol_fin.to_string(ftype="turbo"))
        coords.close()
        return


    def get_M_range(self, M):
        """ Returns range of minimal multiplicities correcponding to the coupling of two spins.

         Derivation/Explanation
        -----------------------------------------------------------------------------------------------------
         ms = (n_alpha-n_beta)*(1/2): minor spin quantum number, which is the z-component of the spin angular momentum
        
         When the spins couple:
         ms total = (ms_1-ms_2), (ms_1-ms_2)+1, ..., ms_1+ms_2
        
         e.g., |O + O -> O2
            ---------------
            m_s|1   1    1
               
         ms_1 = 1, ms_2 = 1 => ms_total = 2,1,0
        
         M = 2*ms+1 (The minimal multiplicity that the system can have)
         ms_1 = (M_1 - 1)/2
         ms_2 = (M_2 - 1)/2
         M = 2*((ms_1-ms_2)+1, 2*((ms_1-ms_2)+1)+1, ..., 2*(ms_1+ms_2+1)+1
         M = (M_1-M_2)+1, (M_1-M_2+2)+1, ..., (M_1+M_2-2)+1
         M = M_1-M_2+1, M_1-M_2+3, ..., M_1+M_2-1
             \_______/                  \_______/
               M_min                      M_max
        -----------------------------------------------------------------------------------------------------

        Args:
            M: list of multiplicities of the two molecules whose spins will couple
               e.g., [3,5]
        Returns: 
            M_range: e.g., [3,5,7]

        """        
        assert len(M) == 2, "This function can only couple two spins"

        M_max = int(    M[0] + M[1] -  1)
        M_min = int(abs(M[0] - M[1]) + 1)
        M_range = range(M_min, M_max+2, 2)
        return M_range

    def reaction_multiplicity(self, M_ed, M_prod):
        """ Determines the multiplicity of the reaction

        Args:
            M_ed   : list of multiplicities of educts
            M_prod : list of multiplciities of products

        Returns:
            spin_not_conserved : True/False : Do the multiplicities of educts and products do not violate for the conservation of spin? 
            M_list : If the multiplicity of the educts and products would result in a spin forbidden reaction returns the union of set of multiplicities.
                     Otherwise, returns their intersection.
        """
        n_ed = len(M_ed)
        n_prod = len(M_prod)
        print('n_ed = ', n_ed, 'n_prod = ', n_prod)
        spin_not_conserved = False
        M_list = []
        if n_ed == 1 and n_prod == 1:
            if M_ed[0] != M_prod[0]:
                spin_not_conserved = True
                M_list = [M_ed[0], M_prod[0]]
            else:
                M_list = M_ed
        elif n_ed == 2 and n_prod == 1:
            M_ed_range = self.get_M_range(M_ed)
            M_int = list(set.intersection(set(M_prod),set(M_ed_range)))
            M_uni = list(set.union(set(M_prod),set(M_ed_range)))
            if M_int == []:
                spin_not_conserved = True
                M_list = M_uni
            else:
                M_list = M_int
        elif n_prod == 2 and n_ed == 1:
            M_prod_range = self.get_M_range(M_prod)
            M_int = list(set.intersection(set(M_ed),set(M_prod_range)))
            M_uni = list(set.union(set(M_ed),set(M_prod_range)))
            if M_int == []:
                spin_not_conserved = True
                M_list = M_uni
            else:
                M_list = M_int
        elif n_prod == 2 and n_ed == 2:
            M_ed_range = self.get_M_range(M_ed)
            M_prod_range = self.get_M_range(M_prod)
            M_int = list(set.intersection(set(M_prod_range),set(M_ed_range)))
            M_uni = list(set.union(set(M_prod_range),set(M_ed_range)))
            if M_int == []:
                spin_not_conserved = True
                M_list = M_uni
            else:
                M_list = M_int
        else:
            print("This function cannot couple more than two spins.")
            sys.exit()
        return spin_not_conserved, M_list



    def make_r_info_dict(self, QM_paths_dict, uni, change, source, origin = None, barrierless = False):
        info = {}
        info['reaction'] = {}
        info['opt_spec'] = {}

        info['reaction']['uni']    = uni # unimolecular?
        info['reaction']['change'] = change # did the reaction graph change?
        info['reaction']['source'] = source # e.g. woelfling/ReaxFF/?...
        info['reaction']['origin'] = origin # reactionID of the reference reaction
        info['reaction']['barrierless'] = barrierless

        print(QM_paths_dict)

        M_ed = []
        M_prod = []
        for i,spec in enumerate(QM_paths_dict):
            info['opt_spec'][i] = {}
            spec_info = info['opt_spec'][i]
            path = QM_paths_dict[spec]
            relpath = os.path.relpath(path, self.path) # Relative path; otherwise, e.g., '/scratch/oyoender_120981/etc'
            spec_info['path'] = relpath
            GT = GeneralTools(path)
            M = GT.get_M_from_control()
            if 'educt' in spec:
                M_ed.append(M)
                spec_info['itype'] = -1
            if 'ts' in spec:
                M_ts = M 
                spec_info['itype'] =  0
            if 'product' in spec: 
                M_prod.append(M)
                spec_info['itype'] =  1
            mol = GT.coord_to_mol()
            spec_info['info'] = ''
            if mol.natoms == 1:
                spec_info['energy'] = GT.get_energy_from_scf_out("ridft.out")
                spec_info['info'] += 'ZPE = 0.0;'
            else:
                spec_info['energy'], ZPE = GT.get_energy_from_aoforce_out()
                spec_info['info'] += 'ZPE = %3.7f;' %ZPE
            spec_info['lot']  = self.lot
            spec_info['info'] += 'M = %d;' %M
            spec_info['info'] += 'ssquare = %3.3f;' %GT.read_ssquare_from_control()

        spin_not_conserved, M_list = self.reaction_multiplicity(M_ed, M_prod)
        if not barrierless:
            if list(set.intersection(set(M_list),{M_ts})) == []: spin_not_conserved = True
        if spin_not_conserved:
            info['reaction']['source'] += ':failed'
        return info
            
    def write_json(self, info_dict, name = 'r'):
        f_json = open(os.path.join(self.path, '%s.json' %name), 'w')
        f_json.write(json.dumps(info_dict, indent = 2))
        f_json.close()
        return

    def reaction_workflow(self, rbonds = [], path_ref_educts = [], path_ref_products = [], path_ref_ts = '', atom_ids_dict = {}, start_lot = '', reactionID = 0, mode = 'mg'):
        """ This method considers the reaction event and optimizes the species accordingly.
        All of the input variables are retrieved from the RDB database, but should also work if one would like to just provide some reference structures...

        Args:
            rbonds           : list of integers : The indices of the reactive bonds in the TS structure. The atom indexing is like in python, 0,1,2,...
            path_ref_educts  : list of strings  : The list of paths to the reference educts cartesian coordinate files.
            path_ref_products: list of strings  : The list of paths to the reference products cartesian coordinate files.
            path_ref_ts      : string           : The path to the TS cartesian coordinate files.
            atom_ids_dict    : dictionary which holds the atom ids matching the ReaxFF optimized structures to that of the trajectory.
            start_lot        : the level of theory of the reference structures; e.g. ReaxFF
            reactionID       : the reactionID of the reference reaction in the database

        Returns:
            Nothing, but writes the "r.json" file which contains information of the reactions for which a TS and connecting minima could be found.
        """
        info_dict = {} # dictionary which holds the information about reaction, this will be dumped into a json file
        counter = 0    # counter for the succesfully assigned rxns 

        QM_path = self.path

        GT = GeneralTools(QM_path)

        n_ed   = len(path_ref_educts)
        n_prod = len(path_ref_products)

        # 1. Optimize the educts and the products
        mols_ini_ed   = []
        mols_ini_prod = []
        for path_ref in path_ref_educts:
            mol_ini = molsys.mol.from_file(path_ref) 
            mol_ini.detect_conn_by_bo(reaxff = self.reaxff)
            mols_ini_ed.append(mol_ini)
        for path_ref in path_ref_products:
            mol_ini = molsys.mol.from_file(path_ref)
            mol_ini.detect_conn_by_bo(reaxff = self.reaxff)
            mols_ini_prod.append(mol_ini)
        M_ed,   QM_paths_ed  , mols_opt_ed    = self.educts_and_products_workflow(mols = mols_ini_ed,   label = 'educt', add_noise = True)
        M_prod, QM_paths_prod, mols_opt_prod  = self.educts_and_products_workflow(mols = mols_ini_prod, label = 'product',  add_noise = True)

        uni = False
        if n_ed == 1 and n_prod == 1: uni = True

        # 2. Determine the multiplicity of the reaction
        spin_not_conserved, M_list = self.reaction_multiplicity(M_ed, M_prod)
        skip_woelfling = False
        if spin_not_conserved : skip_woelfling = True

        print("=========== TS BY ONLY EIGENVECTOR FOLLOWING ============")
        # 3. Pre-optimize the TS contraining the atoms of the reactive bonds
        mol_ini_ts = molsys.mol.from_file(path_ref_ts)
        mol_ini_ts.detect_conn_by_bo(reaxff = self.reaxff)
        for M in M_list:
            converged, QM_path_ts = self.ts_pre_optimization(mol = mol_ini_ts, M = M, rbonds = rbonds, gcart = 3, add_noise = True)

            # ( initiate the woelfling calculation using the orbitals from the pre-optimized TS
            try_woelfling = False
            woelfling_path = os.path.join(self.path,'woelfling_M%d' %M)
            os.chdir(QM_path_ts)
            os.system("cpc %s" %woelfling_path)
            os.chdir(self.maindir)
            # )

            if not converged:
                # -> If not converged set up a woelfling calculation.
                print('The TS pre-optimization did not converge.') 
                try_woelfling = True
            else:
                # 4. Optimize the TS
                # * Calculate the Hessian of the pre-optimized structure.
                # * If the number of imaginary frequencies are not zero, try to optimize the geometry of the transition state with eigenvector following.
                # * If the geometry optimization is converged, perform intrinsic reaction coordinate (IRC) calculation and optimize the end points.
                # * Compare the end points with from the IRC with the educts end products under QM_paths_ed and QM_paths_prod
                #    NOTE: This could also be done as described in  https://doi.org/10.1002/jcc.21925
                #          but at the moment done based on the comparison of the molecular graph
                converged, is_similar, QM_paths_dict, reason = self.ts_workflow(QM_path_ts = QM_path_ts, mols_ed_QM_ref = mols_opt_ed, mols_prod_QM_ref = mols_opt_prod, M = M, mode = mode)
                print(reason)
                # * If not converged then set up a woelfling calculation.
                if not converged:
                    try_woelfling = True
                # * If converged, then it will be added to the database.
                else:
                    r_info = self.make_r_info_dict(QM_paths_dict = QM_paths_dict, uni = uni, change = not is_similar, source = start_lot+':ts', origin=reactionID)
                    info_dict[counter] = r_info
                    self.write_json(info_dict)
                    counter += 1
                    if not is_similar:
                        print('A TS is found but does not connect the original minima.')
                        try_woelfling = True

            # 5. Perform the woelfling calculation
            if try_woelfling and not skip_woelfling:
                print("============ WOELFLING CALCULATION =============")
                # the coords file
                coords_path = os.path.join(woelfling_path,'coords')
                if uni:
                    mol_ed = self.reorder_wrt_ts(QM_path, path_ref_ts, 'educt', 1, rbonds, atom_ids_dict)
                    mol_prod = self.reorder_wrt_ts(QM_path, path_ref_ts, 'product', 1, rbonds, atom_ids_dict)
                    is_similar = GT.similaritycheck_from_mol(mol_ed, mol_prod)
                    if is_similar:
                        print('The educt and the product is the same. A woelfling calculation is not possible.')
                        sys.exit()
                    self.write_coords(coords_path, mol_ed, mol_prod)
                elif n_ed > 1 and n_prod == 1:
                    mol_ed_complex = self.make_rxn_complex(rbonds, atom_ids_dict, 'educt', n_ed, self.path, path_ref_ts)
                    mol_prod = self.reorder_wrt_ts(QM_path, path_ref_ts, 'product', 1, rbonds, atom_ids_dict)
                    self.write_coords(coords_path, mol_ed_complex, mol_prod)
                elif n_ed > 1 and n_prod > 1:
                    mol_ed_complex = self.make_rxn_complex(rbonds, atom_ids_dict, 'educt', n_ed, self.path, path_ref_ts)
                    mol_prod_complex = self.make_rxn_complex(rbonds, atom_ids_dict, 'product', n_prod, self.path, path_ref_ts)
                    self.write_coords(coords_path, mol_ed_complex, mol_prod_complex)
                elif n_ed == 1 and n_prod > 1:
                    mol_ed = self.reorder_wrt_ts(QM_path, path_ref_ts, 'educt', 1, rbonds, atom_ids_dict)
                    mol_prod_complex = self.make_rxn_complex(rbonds, atom_ids_dict, 'product', n_prod, self.path, path_ref_ts)
                    self.write_coords(coords_path, mol_ed, mol_prod_complex)

                # 6. Run woelfling and check if it is barrierless, or which one is the highest peak
                barrierless, ts_paths = self.woelfling_workflow(woelfling_path)

                if barrierless:
                    QM_paths_dict = {} 
                    print('The reaction is barrierless.')
                    for i,QM_path_ed in enumerate(QM_paths_ed):
                        QM_paths_dict['educt_%d' %i] = QM_path_ed
                        OT = OptimizationTools(QM_path_ed, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
                        OT.aoforce()
                        inum, imfreq = OT.check_imaginary_frequency()
                        if inum != 0:
                            OT.disturb_and_reoptimize()
                    for i,QM_path_prod in enumerate(QM_paths_prod):
                        QM_paths_dict['product_%d' %i] = QM_path_prod
                        OT = OptimizationTools(QM_path_prod, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
                        OT.aoforce()
                        inum, imfreq = OT.check_imaginary_frequency()
                        if inum != 0:
                            OT.disturb_and_reoptimize()
                    r_info = self.make_r_info_dict(QM_paths_dict = QM_paths_dict, uni = uni, change = False, source = start_lot+':woelfling', origin=reactionID,  barrierless = True)
                    info_dict[counter] = r_info
                    counter += 1
                    self.write_json(info_dict)


                else:
                    for ts_path in ts_paths:
                        converged, is_similar, QM_paths_dict, reason = self.ts_workflow(QM_path_ts = ts_path, mols_ed_QM_ref = mols_opt_ed, mols_prod_QM_ref = mols_opt_prod, M = M, mode = 'mg')
                        print(reason)
                        if not converged:
                            print('The TS guess from woelfling calculation did not converge.')
                        else:
                            r_info = self.make_r_info_dict(QM_paths_dict = QM_paths_dict, uni = uni, change = not is_similar, source = start_lot+':woelfling', origin=reactionID)
                            info_dict[counter] = r_info
                            counter += 1
                            self.write_json(info_dict)

            else:
                shutil.rmtree(woelfling_path)
        return



    def dft_re_optimization(self, specID, ts = False):
        """ This method re-optimizes the structures of previous DFT calculations with the new functional and basis set.

        Args:
            opt_speciesID: the opt_species id of the reference reaction from the database
            ts           : is it a ts or not?
        """
        change_molg = False 
        ospec_info = {}
 

        # 1. Make molecular graph of the reference structure
        GT = GeneralTools(self.path)
        mol_ini = GT.coord_to_mol()
        mg_ini = self.make_molecular_graph(mol_ini)

        # 2. Single point calculation
        energy = GT.ridft()
        converged = GT.check_scf_converged()
        if not converged:
            print("ridft did not converge.")
            sys.exit()

        if mol_ini.natoms != 1:

            # 3. Geometry optimization
            converged = self.jobex(ts=ts)
            mol_opt = GT.coord_to_mol()
            mg_opt = self.make_molecular_graph(mol_opt)
            change_molg = not molsys.addon.graph.is_equal(mg_ini, mg_opt, use_fast_check=False)[0]

            if not converged:
                print("Geometry optimization did not converge.")

            # 4. Calculate the Hessian
            self.aoforce()
            energy, ZPE = GT.get_energy_from_aoforce_out()

            inum, imfreq = self.check_imaginary_frequency()
            if ts:
                if inum != 1:
                    print("This is not a saddle point.")
                    sys.exit()
            else:
                if inum != 0:
                    self.disturb_and_reoptimize(natoms=mol_opt.natoms)
        else: # atom
            ZPE = 0 

        wherefrom  = os.path.join(self.path,'wherefrom')
        statistics = os.path.join(self.path,'statistics')
        if os.path.isfile(wherefrom): os.remove(wherefrom)
        if os.path.isfile(statistics): os.remove(statistics)

        ospec_info['lot']    = self.lot
        ospec_info['energy'] = energy
        ospec_info['specID'] = specID
        ospec_info['info']   = 'ZPE = %3.7f;M = %d;ssquare = %3.3f;' %(ZPE,GT.get_M_from_control(),GT.read_ssquare_from_control())
        ospec_info['change_molg'] = change_molg
        self.write_json(info_dict = ospec_info, name = 'ospec')
        return


    def replace_uhfmo_xxx_with_scfmo(self, xxx, ref_mo_file, mo_file):
        f1 = open(ref_mo_file, 'r')
        f2 = open(mo_file, 'w')
        for line in f1:
            f2.write(line.replace('uhfmo_%s' %xxx, 'scfmo'))
        f1.close()
        f2.close()
        return

    def replace_cso_xxx_with_scfmo(self, xxx, ref_mo_file, mo_file):
        f1 = open(ref_mo_file, 'r')
        f2 = open(mo_file, 'w')
        for line in f1:
            f2.write(line.replace('cso_%s' %xxx, 'scfmo'))
        f1.close()
        f2.close()
        return

    def get_csos(self, proper_path="proper"):
        os.chdir(self.path)
        f = open("proper.in", "w")
        f.write("mos\ncsos symao\nq")
        f.close()
        os.system("%s < proper.in > proper_csos.out" %proper_path)
        os.remove("proper.in")
        os.chdir(self.maindir)
        return
        

    def refine_ccsd(self, ospecID, pnoccsd="", proper_path="proper"):
        # 1. Get the model and basis set from level of theory (self.lot)
        # example formats for lot:
        # "PNO-ROHF-CCSD(T)(F12*)/cc-pVTZ" -> Uses pnoccsd code
        # "PNO-CCSD(T)(F12*)/cc-pVTZ"      -> Takes the orbitals from the reference DFT calculation
        # "ROHF-CCSD(T)(F12*)/cc-pVTZ"     -> Uses canonical code ccsdf12
        # "CCSD(T)(F12*)/cc-pVTZ"          -> Takes the orbitals from the reference DFT calculation
        Tstar = False
        if "pno-" in self.lot.lower() and "rohf" in self.lot.lower():
            model_basis_set = re.split("rohf-", self.lot, flags=re.IGNORECASE)[-1]
            pno = True
            rohf = True
        elif "pno-" in self.lot.lower() and "rohf" not in self.lot.lower():
            model_basis_set = re.split("pno-", self.lot, flags=re.IGNORECASE)[-1]
            pno = True
            rohf = False
        elif "pno-" not in self.lot.lower() and "rohf" in self.lot.lower():
            model_basis_set = re.split("rohf-", self.lot, flags=re.IGNORECASE)[-1]
            pno = False
            rohf = True
        elif "pno-" not in self.lot.lower() and "rohf" not in self.lot.lower():
            model_basis_set = self.lot.lower()
            pno = False
            rohf = False
        basis_set = model_basis_set.split("/")[-1]
        model_tmp = model_basis_set.split("/")[0]
        # check if F12 approximation is used
        if "(f12*)" in model_basis_set.lower():
            F12 = True
            if "-" in model_tmp:
                model = ''.join(re.split("\(f12\*\)", model_tmp, flags=re.IGNORECASE)).lower().split("-")[-1]
                if '*' in model:
                    model = ''.join(model.split('*'))
                    Tstar = True
            else:
                model = ''.join(re.split("\(f12\*\)", model_tmp, flags=re.IGNORECASE)).lower()
                if '*' in model:
                    model = ''.join(model.split('*'))
                    Tstar = True
        elif "f12" in model_basis_set.lower():
            print("Only the submission of the calculations with CCSD(F12*) approximation is automated. For other F12 approximations, you need to modify the code.")
            sys.exit()
        else:
            print("No F12 approximation will be used.")
            F12 = False
            model = model_tmp.lower()
        # make sure the the model matches in the pnoccsd data group and the specified lot.
        if pno:
            assert model in [s.strip() for s in pnoccsd.split('\n')], "The model %s specified in the level of theory (lot) should be consistent with that specified in the pnoccsd option:" %model + pnoccsd

        # cpc to a new directory
        os.chdir(self.path)
        dir_name = "".join(".".join(".".join("".join(self.lot.split("*")).split("/")).split("(")).split(")"))
        ccsd_path = os.path.join(self.path,dir_name)
        os.mkdir(ccsd_path)
        OT = OptimizationTools(ccsd_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
        GT = GeneralTools(ccsd_path)
        GT_ref = GeneralTools(self.path)
        # 1. Set up the HF calculation input
        if rohf:
            shutil.copy(os.path.join(self.path,"coord"),os.path.join(ccsd_path,"coord"))
            M = GT_ref.get_M_from_control()
            GT.define_rohf(basis_set=basis_set, M=M)
            # Start with the DFT orbitals of the highest occupied one (alpha/beta)
            tmp_path = os.path.join(self.path,"tmp")
            os.system("cpc tmp")
            GT_tmp =  GeneralTools(tmp_path)
            OT_tmp =  OptimizationTools(tmp_path, lot = self.lot, max_mem = self.max_mem, gcart = self.gcart, reaxff = self.reaxff)
            # / blow up the basis set so that the number of atomic orbitals are correct
            GT_tmp.change_basis_set(basis_set=basis_set, ref_control_file = "../control",  title = self.lot, ri=False)
            GT_tmp.kdg("rij")
            GT_tmp.dscf()
            converged = GT_tmp.check_scf_converged("dscf")
            if not converged:
                print("The scf calculation did not converge with the new basis set.")
            mo_file = os.path.join(ccsd_path,"mos")
            os.remove(mo_file)
            # / get the corresponding spin orbitals
            OT_tmp.get_csos()
            print('number of alpha electrons:', GT_ref.nalpha,'; number of beta electrons:', GT_ref.nbeta)
            # / copy the blown up alpha or beta mo file to mos
            if GT_ref.nalpha >= GT_ref.nbeta:
                ref_mo_file = os.path.join(tmp_path,"cso_alpha")
                self.replace_cso_xxx_with_scfmo("alpha", ref_mo_file, mo_file)
            #    ref_mo_file = os.path.join(tmp_path,"alpha")
            #    self.replace_uhfmo_xxx_with_scfmo("alpha", ref_mo_file, mo_file)
            else:
                ref_mo_file = os.path.join(tmp_path,"cso_beta")
                self.replace_cso_xxx_with_scfmo("beta", ref_mo_file, mo_file)
                #ref_mo_file = os.path.join(tmp_path,"beta")
                #self.replace_uhfmo_xxx_with_scfmo("beta", ref_mo_file, mo_file)            
        else:
            # Start with the DFT orbitals
            os.system("cpc %s" %dir_name)
            # Change the basis set
            GT.change_basis_set(basis_set=basis_set, ref_control_file = "../control",  title = self.lot, ri=False)
            # Remove dft, rij, and disp3 from the control file
            GT.kdg("dft")
            GT.kdg("rij")
            GT.kdg("disp3")
            GT.kdg("energy")
            GT.kdg("grad")
            GT.kdg("arh")

        # write additional information about timings and memory usage into statistics.ccsdf12
        OT.add_dg_to_control("profile")
 
        mol = GT.coord_to_mol()

        # 2. Perform the HF calculation
        bck_path = os.path.join(ccsd_path, "BACK")
        os.chdir(ccsd_path)
        os.system("cpc %s" %bck_path)
        os.chdir(self.path)
        GT_bck = GeneralTools(bck_path)
        HF_energy = GT_bck.dscf()
        converged = GT_bck.check_scf_converged("dscf")
        if not converged:
            bck_path_2 = os.path.join(ccsd_path, "BACK_dsta_5")
            os.chdir(ccsd_path)
            os.system("cpc %s" %bck_path_2)
            os.chdir(self.path)
            GT_bck_2 = GeneralTools(bck_path_2)
            GT_bck_2.change_scfdamp(start=5.0,end=0.5)
            HF_energy = GT_bck_2.dscf()
            converged = GT_bck_2.check_scf_converged("dscf")
            if not converged:
                print('The SCF calculation with start damping 5.0 did not converge.')
                sys.exit()
            else:
                GT_bck_2.change_scfdamp(start=1.0)
                HF_energy = GT_bck_2.dscf()
                converged = GT_bck_2.check_scf_converged("dscf")
                if not converged:
                    print('The SCF calculation with start damping 1.0 did not converge.')
                    sys.exit()
                for f in os.listdir(bck_path_2):
                    shutil.move(os.path.join(bck_path_2, f), os.path.join(ccsd_path, f))
                shutil.rmtree(bck_path_2)
        else:
            for f in os.listdir(bck_path):
                shutil.move(os.path.join(bck_path, f), os.path.join(ccsd_path, f))
        shutil.rmtree(bck_path)

        # / remove the tmp directory
        shutil.rmtree(tmp_path)
 
       
        # 3. Add the pnoccsd options
        if pno:
            GT.define_pnoccsd(pnoccsd=pnoccsd, max_mem=self.max_mem, F12=F12)
            ccsd_out_name="pnoccsd.out"
            OT.pnoccsd()
        else:
            GT.define_canonical_ccsd(model=model, max_mem=self.max_mem, F12=F12)
            ccsd_out_name="ccsdf12.out"
            OT.ccsdf12()
        self._clean_ccsd(ccsd_path)

        # 4. Write the json file to later add to the database
        energy = GT.get_ccsd_energy(F12=F12, Tstar=Tstar, model=model, ccsd_out_name=ccsd_out_name, n_el = mol.get_n_el())
        info_dict = {}
        info_dict["ospecID"] = ospecID
        info_dict["info"] = "E(%s)=%5.10f;" %(self.lot, energy)
        self.write_json(info_dict, name = 'ccsd')
        return         



    def write_submit_py_level1(self, path_ref_educts = [], path_ref_products = [], path_ref_ts = '', start_lot = '', reactionID = 0, rbonds = [], atom_ids_dict = {}, reaxff = "cho", mode = 'mg'):
        """ 
            This method writes a script which can later be used to submit jobs to a queuing system by writing a job submission script. 
            See in the Slurm class the write_submission_script.

        Args: 
            path_ref_X : the path to the mfpx or xyz structure files of the xTB or ReaxFF optimized structures
            start_lot  : the start level of theory; e.g. ReaxFF/xTB
            reactionID : the reference reactionID from the database
            rbonds       : the list of reactive bonds
            atom_ids_dict: dictionary which holds the atom ids matching the ReaxFF optimized structures to that of the trajectory.
        """
        f_path = os.path.join(self.path,"submit.py")
        f = open(f_path,"w")
        f.write("import os\n")
        f.write("import molsys\n")
        f.write("from molsys.util import refine_qm_turbomole\n\n")
        f.write("lot               = '%s'\n" %self.lot)
        f.write("max_mem           = %d\n"   %self.max_mem)
        f.write("gcart             = %d\n"   %self.gcart)
        f.write("rbonds            = %s\n"   %str(rbonds))
        f.write("path_ref_educts   = %s\n"   %str(path_ref_educts))
        f.write("path_ref_products = %s\n"   %str(path_ref_products))
        f.write("path_ref_ts       = '%s'\n" %path_ref_ts)
        f.write("atom_ids_dict     = %s\n"   %str(atom_ids_dict))
        f.write("start_lot         = '%s'\n" %start_lot)
        f.write("reactionID        = %d\n"   %reactionID)
        f.write("reaxff            = '%s'\n" %reaxff)
        f.write("mode              = '%s'\n" %mode )
        f.write("OT = refine_qm_turbomole.OptimizationTools(lot = lot, max_mem = max_mem, gcart = gcart, reaxff = reaxff)\n")
        f.write("OT.reaction_workflow(rbonds, path_ref_educts, path_ref_products, path_ref_ts, atom_ids_dict, start_lot, reactionID, mode)\n")
        f.close()
        return

    def write_submit_py_level2(self, specID, ts, reaxff = "cho"):
        """ writes a script to submit the jobs for the DFT calculations starting from another DFT calculation        
        """
        f_path = os.path.join(self.path,"submit.py")
        f = open(f_path,"w")
        f.write("import os\n")
        f.write("import molsys\n")
        f.write("from molsys.util import refine_qm_turbomole\n\n")
        f.write("lot               = '%s'\n" %self.lot)
        f.write("max_mem           = %d\n"   %self.max_mem)
        f.write("gcart             = %d\n"   %self.gcart)
        f.write("specID            = %d\n"   %specID)
        f.write("reaxff            = '%s'\n"   %reaxff)

        f.write("OT = refine_qm_turbomole.OptimizationTools(lot = lot, max_mem = max_mem, gcart = gcart, reaxff = reaxff)\n")
        if ts:
            f.write("OT.dft_re_optimization(specID=specID, ts=True)\n")
        else:
            f.write("OT.dft_re_optimization(specID=specID, ts=False)\n")
        f.close()
        return


    def write_submit_py_level3(self, ospecID, pnoccsd, submit_py='submit_pnoccsd.out', reaxff = "cho"):
        f_path = os.path.join(self.path,submit_py)
        f = open(f_path,"w")
        f.write("import os\n")
        f.write("import molsys\n")
        f.write("from molsys.util import refine_qm_turbomole\n\n")
        f.write("lot               = '%s'\n" %self.lot)
        f.write("max_mem           = %d\n"   %self.max_mem)
        f.write("ospecID            = %d\n"   %ospecID)
        f.write("pnocssd = %s\n" %repr(pnoccsd))
        f.write("reaxff            = '%s'\n"   %reaxff)
        f.write("OT = refine_qm_turbomole.OptimizationTools(lot = lot, max_mem = max_mem, reaxff = reaxff)\n")
        f.write("OT.refine_ccsd(ospecID=ospecID, pnoccsd=pnocssd)")
        f.close()
        return


class Slurm:

    def get_avail_nodes():
        """ Returns a list of available nodes.
        """
        sinfo = os.popen('sinfo --format="%n %t %c %m"').read()
        n_t_c_m = sinfo.split("\n")
        avail_nodes = []
        for lines in n_t_c_m:
            line = lines.split()
            if len(line)==4 and line[1] == "idle":
                avail_nodes.append((line[0],int(line[2]),int(line[3])))
        return avail_nodes


    def get_avail_nodes_of_(partition="normal"):
        """ Returns a list of available nodes under that partition.
        """
        sinfo = os.popen('sinfo --format="%n %t %P"').read()
        n_t_P = sinfo.split("\n")
        avail_nodes = []
        for lines in n_t_P:
            line = lines.split()
            if len(line)==3 and line[1] == "idle" and line[2].startswith(partition):
                avail_nodes.append(line[0])
        return avail_nodes

    def get_partition_info(partition="normal"):
        """ Returns the number of CPUs and memory of a node which belongs to that partition.
        """
        sinfo = os.popen('sinfo --format="%P %c %m"').read()
        P_c_m = sinfo.split("\n")
        for lines in P_c_m:
            line = lines.split()
            if len(line)==3 and line[0].startswith(partition):
                CPUS = int(line[1])
                MEM  = int(line[2])
        return CPUS, MEM

    def write_submission_script(path=os.getcwd(), TURBODIR="", ntasks=8, partition="normal", exclusive = False, submit_sh='submit.sh', submit_py='submit.py', submit_out='submit.out'):
        """ Writes a SLURM job submission script which will run whatever is written in submit.py. See write_submit_py under OptimizationTools.
        """
        s_script_path = os.path.join(path, submit_sh)
        s_script = open(s_script_path,"w")
        s_script.write("#!/bin/bash\n")
        s_script.write("#SBATCH --ntasks=%d\n" %ntasks)
        s_script.write("#SBATCH --nodes=1\n")
        s_script.write("#SBATCH --error=job.%J.err\n")
        s_script.write("#SBATCH --output=job.%J.out\n")
        s_script.write("#SBATCH --time=999:00:00\n")
        s_script.write("#SBATCH --partition=%s\n" %partition)
        if exclusive:
            s_script.write("#SBATCH --exclusive\n")
        s_script.write("#=====================================\n")
        s_script.write("# Setup the environment for Turbomole \n")
        s_script.write("#=====================================\n")
        s_script.write("export TURBODIR=%s\n" %TURBODIR)
        s_script.write("source $TURBODIR/Config_turbo_env\n")
        s_script.write("export PATH=$TURBODIR/scripts:$PATH\n")
        s_script.write("export PARA_ARCH=SMP\n")
        s_script.write("export PATH=$TURBODIR/bin/`sysname`:$PATH\n")
        s_script.write("export OMP_NUM_THREADS=$SLURM_NTASKS\n")
        s_script.write("export PARNODES=$SLURM_NTASKS\n")
        s_script.write("#=====================================\n")
        s_script.write("#  Copy every file and run the job    \n")
        s_script.write("#=====================================\n")
        s_script.write("sbcast %s $TMPDIR/%s\n" %(submit_py,submit_py))
        s_script.write("cpc $TMPDIR\n")
        s_script.write("cp *mfpx $TMPDIR\n")
        s_script.write("cd $TMPDIR\n")
        s_script.write("python3 %s > %s\n" %(submit_py,submit_out))
        s_script.write("#=====================================\n")
        s_script.write("# Copy everything back                \n")
        s_script.write("#=====================================\n")
        s_script.write("cp -r $TMPDIR/* $SLURM_SUBMIT_DIR/\n")
        s_script.write("exit\n")
        s_script.close()
        return



class Harvest:
   
    # TODO change the freeh directory to a general name 
    def __init__(self, path=os.getcwd(), freehdir = 'freeh', verbose = False):
        if not os.path.isdir(path):
            raise FileNotFoundError("The directory %s does not exist." % path)
        else:
            self.path = path
            self.maindir = os.getcwd()
        self.verbose = verbose
        self.freehdir = freehdir
        self.kB = scipy.constants.physical_constants['Boltzmann constant'][0] # J/K
        self.h  = scipy.constants.physical_constants['Planck constant'][0] # J.s
        self.R  = scipy.constants.physical_constants['molar gas constant'][0] # J/mol/K or Pa.m3/mol/K
        self.NA = scipy.constants.physical_constants['Avogadro constant'][0] # molecule/mol
        self.c  = scipy.constants.physical_constants['speed of light in vacuum'][0]*100.0 # cm/s
        return

    def thermodynamics(self, onlyqvib = False, f_cor = 1.0, Tstart = 300.0, Tend = 3000.0, numT = 271, P = 0.1, getGSH = False):
        ''' Calls the freeh program for the temperature range needed at a given pressure.

        Args:
        onlyqvib  : Do you only want to get the logarithm of vibrational partition function?
        f_cor     : Zero point vibrational energy correction factor.
        T_start   : Start temperature in K
        T_end     : End temperature in K
        numT      : Number of temperatures
        P         : Pressure in MPa
            
        Returns:
        lnQ       : Numpy array of partition functions
        T         : Numpy array of temperatures
        '''
        os.chdir(self.path)
        # 1. initialize the temperature list, and the thermodynanmic variable dictionaries.
        T=[] # temperature in K
        lnQ=[] # dictionary of partition functions. Later needed to calculate the concentration of the activated complex, 
             # and hence, reaction rate and the rate constant from Transition State Theory.
        if getGSH:
            G=[] # dictionary of Gibbs Free Energy | 
            S=[] # dictionary of Entropy           | Later needed to fit NASA polynomials
            H=[] # dictionary of Enthalpy          |
            #Cp=[]

        # 2. Write an input file for the freeh program
        f = open('freeh.inp','w+')
        f.write('\n%f\ntstart=%f tend=%f numt=%d pstart=%f\nq' %(f_cor,Tstart,Tend,numT,P))
        f.close()
        # run the freeh program
        os.system('%s -freerotor < freeh.inp > freeh.out  2> /dev/null' %self.freehdir)

        # 3. Read the output and get the thermodynamic variables
        linenum = 0
        with open('freeh.out') as freehout:
            for i, line in enumerate(freehout, start=1):
                if 'The quasi-RRHO Approach Short Output' in line:
                    linenum=i+11 # The line where the thermodynamic properties are started to be list.
                if 'Total degeneracy of the electronic wavefunction' in line: qel = float(line.strip().split()[-1])
                if linenum <= i <= linenum+numT-1 and linenum != 0:
                    Ttmp=float(line.split()[0])
                    T.append(Ttmp) # in K
                    lnqtrans=float(line.split()[2])
                    #V = kB*Ttmp/PressurePa
                    #qtransperunitvolume = qtrans/V
                    lnqrot=float(line.split()[3])
                    # vibrational partition function with the energy levels relative to the bottom of the internuclear potential well.
                    lnqvib=float(line.split()[4])
                    if onlyqvib:
                       lnQ.append(lnqvib)
                    else:
                       lnQ.append(lnqtrans+lnqrot+lnqvib+numpy.log(qel))
                    if getGSH:
                       G.append(float(line.split()[5])*1.0E3) # in J/mol
                       S.append(float(line.split()[6])*1.0E3) # in J/mol/K
                       H.append(float(line.split()[8])*1.0E3) # in J/mol
                       # Cp.append(float(line.split()[7])*1.0E3) # in J/mol/K
        os.chdir(self.maindir)
        lnQ = numpy.array(lnQ)
        T = numpy.array(T)
        if getGSH:
            G = numpy.array(G)
            S = numpy.array(S)
            H = numpy.array(H)
            return lnQ, T, G, S, H
        return lnQ, T
    
    def func(self, T, lnA, n, E):
        lnk = lnA+n*numpy.log(T)-E/T
        return lnk
    
    def fitArr(self, lnk, T, lnA0, n0, E0, k_unit='', title='', subtitle=''):
        """
        Returns:
        popt is basically the fitted values
        """
        popt, pcov = scipy.optimize.curve_fit(f = self.func, xdata = T, ydata = lnk, p0 = [lnA0, n0, E0])
        if self.verbose:
            lnkfit = self.func(T, *popt)
            print('lnk')
            print(lnk)
            print('lnkfit')
            print(lnkfit)
            plt.rcParams.update({'font.size': 14})
            plt.xlabel('1000/T (1/K)')
            plt.ylabel('lnk (k in %s)' %k_unit)
            plt.plot(1000.0/T,  lnkfit,       label = 'fitted'     )
            plt.plot(1000.0/T,  lnk,   '.',   label = 'calculated' )
            plt.legend()
            plt.suptitle(title)
            plt.title(subtitle)
            plt.savefig('lnk%s.png' %(title.replace(" ","")+subtitle.replace(" ","")))
            print('p0 = [lnA0, n0, E0] = [', lnA0, n0, E0,']')
            print('p  = [lnA,  n,  E]  = [', popt[0], popt[1], popt[2],']')
            plt.close()
        return popt, pcov
 

    def get_kappa(self, T, E_el_act, imfreq):
       # The tunneling factor
       # since the frequency is imaginary, the square of it will give a minus. Therefore, the equation is normally given as 1 - 1/24*(...), 
       # but here since frequency is read as a negative value, its square will be positive. Therefore, 1 + 1/24*(...)
       kappa = 1.0 + 1.0/24.0*numpy.square(((self.h*imfreq*self.c)/(self.kB*T)))*(1.0 + (self.R*T)/(1000.0*E_el_act*self.NA))
       return kappa


    def get_lnk(self, T, P, E_el_act, imfreq, lnQ_educts, lnQ_ts):
        """
        Args:
        T          : K        : Numpy  array of temperatures
        P          : MPa      : The pressure
        E_el_act   : kJ/mol   : Electronic activation energy
        imfreq     : cm^-1    : The imaginary frequency
        lnQ_educts : unitless : List of list of the natural logarithm of the educts for the temperature range
        lnQ_ts     : unitless : List of the natural logarithm of the transition state for the temperature range
        """
        kappa = self.get_kappa(T, E_el_act, imfreq)
        #print('kappa=', kappa)
        # Units of V: (J*mol/K)*K/MPa = Pa*m³/mol/MPa = 10⁻⁶m³/mol = cm³/mol
        lnV = numpy.log(self.NA*self.kB*T/P) # Natural logarithm of the molar volume 
        lnQ = 0.0
        lnQ += lnQ_ts
        n = -1
        for lnQ_ed in lnQ_educts:
            n += 1
            lnQ -= lnQ_ed
        if self.verbose:
            print('lnQ', lnQ)
            print('E_el_act*1.0E3/(self.R*T)', E_el_act*1.0E3/(self.R*T))
        lnk = numpy.log(kappa) + numpy.log(self.kB*T/self.h) + lnQ + n*lnV - E_el_act*1.0E3/(self.R*T)
        if n == 0:
            k_unit = '1/s'
        elif n == 1:
            k_unit = 'cm^3/mol/s'
        else:
            k_unit = '((cm^3/mol)^%d/s)' %n
        return lnk, k_unit


    def get_p0(self, E_el_act, lnk, T, test=False):
        """
        Returns initial guess for the fit parameters of the modified Arrhenius law
        k = A T^n e^(-E/T)

        Args:
        E_el_act : kJ/mol : Electronic activation energy
        lnk      : Natural logarithm of the rate constants
        T        : Temperature
        """
        E0 = E_el_act*1.0E3/self.R
        n0 = (lnk[10]-lnk[-10] + E0*(1.0/T[10]-1.0/T[-10]))/(numpy.log(T[10]/T[-10]))
        lnA0 = lnk[-10] - n0*numpy.log(T[-10]) - E0/T[-10]
        if test:
            print('p0 = [lnA0, n0, E0] = [', lnA0, n0, E0,']')
        return lnA0, n0, E0
