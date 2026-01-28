SHELL=/bin/sh
MAKE=make-f Makefile

# Library name
MOD=_ric

# F2PY debug flags
F2PY_DEBUG=
F2PY_OPT=
F2PY_COMP=gnu95
LAPACK=-llapack

# List of all Fortran source files
SRC= ric_common.f90 \
     ric_math.f90 \
     ric_stretches.f90 \
     ric_in_bends.f90 \
     ric_out_bends.f90 \
     ric_lin_bends.f90 \
     ric_torsions.f90 \
     ric_eckart.f90 \
     ric.f90 \

all:
	f2py --fcompiler=$(F2PY_COMP) --f90flags='$(F90FLAGS)'  $(F2PYOPT) -c $(LDLIBS) -m $(MOD) $(SRC) $(LAPACK)
