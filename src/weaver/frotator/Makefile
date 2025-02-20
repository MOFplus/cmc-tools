#=======================================================================
# Define default settings
#=======================================================================

SHELL=/bin/sh
MAKE=make -f Makefile

# never forget the PIC -> we build an so!
SRCFILES= frotator.f90
MODS= frotator


LDLIBS=-L/usr/lib -lblas
#F2PYOPT=   -DF2PY_REPORT_ATEXIT -DF2PY_REPORT_ON_ARRAY_COPY=1 -liomp5
#F2PYOPT=   -DF2PY_REPORT_ATEXIT
F2PYOPT=   -DF2PY_REPORT_ATEXIT_DISABLE
#-DF2PY_REPORT_ON_ARRAY_COPY=0 -liomp5
 F2PY_COMP=gnu95
#F2PY_COMP=intelem


#=====================================================================
all:
	#f2py --fcompiler=$(F2PY_COMP) --f90flags='$(F90FLAGS)'  $(F2PYOPT) -c $(LDLIBS) -m $(MODS) $(SRCFILES)
	#### mv $(MODS).*.so $(MODS).so
	#cp $(MODS).so ..
	f2py3 --fcompiler=$(F2PY_COMP) --f90flags='$(F90FLAGS)'  $(F2PYOPT) -c $(LDLIBS) -m $(MODS)3 $(SRCFILES)
	mv $(MODS)3.*.so $(MODS)3.so
	cp $(MODS)3.so ..

#=====================================================================
# Clean up the source directory
clean:
	rm -f $(OBJ_MOD) $(OBJ_PAR) $(OBJ_SRC) *.mod *.so

# #=====================================================================
# # Declare dependencies
# .f.o: 
# 	$(FC) $(FFLAGS) $*.f
# .c.o: 
# 	$(CC) -c $*.c
# 
# #=====================================================================
# # Declare dependency on module files
# 
# $(OBJ_SRC): $(OBJ_MOD)

