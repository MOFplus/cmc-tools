import pytest

import molsys
import molsys.util.toper as toper

m = molsys.mol()
m.read("HKUST-1.mfpx")
tt = toper.topotyper(m)

@pytest.mark.slow
def test_get_net():
    assert tt.get_net() == ["tbo"]

def test_write_bbs():
    tt.write_bbs("run/bbs")
    assert len(tt.unique_bbs) == 2
