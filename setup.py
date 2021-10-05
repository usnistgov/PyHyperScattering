from distutils.core import setup
import versioneer

setup( version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass())

