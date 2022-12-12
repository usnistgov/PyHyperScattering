import sys,os
sys.path.append("src/")

from PyHyperScattering import __version__

def test_has_version():
    assert type(__version__)==str
