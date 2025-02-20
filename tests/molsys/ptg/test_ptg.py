import pytest

# example molecules from symmol 2002-11-04
# results may be different for symmol custom option

import molsys
import sys
from os.path import realpath, splitext, sep

srcdir = 'xyz'
fnames = ["%02d.xyz" % i for i in range(1,53)] # to keep original symmol 2002-11-04 examples
sch_symbols = [
    "Cs",    # 01
    "D6h",   # 02
    "D3h",   # 03
    "C2v",   # 04
    "D3h",   # 05
    "Td",    # 06
    "Oh",    # 07
    "D5h",   # 08
    "D5d",   # 09
    "C5v",   # 10
    "S4",    # 11
    "C3v",   # 12
    "C3v",   # 13
    "C3v",   # 14
    "C1",    # 15
    "D1d",   # 16
    "Ci",    # 17
    "D1d",   # 18
    "D2",    # 19
    "C2v",   # 20
    "slow",  # 21
    "Ih",    # 22
    "Ih",    # 23
    "Ih",    # 24
    "Ih",    # 25
    "Ih",    # 26
    "Cs",    # 27
    "Cs",    # 28
    "C2v",   # 29
    "Td",    # 30
    "C*v",   # 31
    "Td",    # 32
    "D3h",   # 33
    "D2d",   # 34
    "D5h",   # 35
    "Ih",    # 36
    "I",     # 37
    "Td",    # 38
    "Td",    # 39
    "I",     # 40
    "D100",  # 41
    "Ih",    # 42
    "C2v",   # 43
    "Oh",    # 44
    "D3",    # 45
    "D3",    # 46
    "C3v",   # 47
    "C1",    # 48
    "C2v",   # 49
    "Cs",    # 50
    "wrong", # 51
]

@pytest.mark.parametrize("fname,sch_symbol", zip(fnames, sch_symbols))
def test_check_pointgroup(fname, sch_symbol):
    if sch_symbol == "slow":
        pytest.xfail("skipped slow example: a cube (to speed up)")
    if sch_symbol == "wrong":
        pytest.xfail("failed known bug: a linear 5C molecule (to be submitted to pymatgen)")
    fpath = srcdir + sep + fname
    m = molsys.mol.from_file(fpath)
    m.addon("ptg")
    m.ptg.analyze()
    assert m.ptg.sch_symbol == sch_symbol, "found and expected symbols are different: %s != %s" \
        % (mp.ptg.sch_symbol, sch_symbol)
