from PyHyperScattering import load
from PyHyperScattering import integrate
from PyHyperScattering import util
from PyHyperScattering import optional_dependencies

from . import _version
__version__ = _version.get_versions()['version']

# Check for commonly used optional dependencies on import
# This will issue at most one warning per missing package per session
optional_dependencies.warn_if_missing('cupy')
optional_dependencies.warn_if_missing('dask')
optional_dependencies.warn_if_missing('holoviews')
