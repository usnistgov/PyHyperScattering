from distutils.core import setup
import sys,os
sys.path.append(os.path.dirname(__file__))
import versioneer

setup( version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass())

