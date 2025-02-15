"""Module for managing optional dependencies in PyHyperScattering."""

import importlib
import warnings
from functools import wraps

# Dictionary of optional dependencies and their associated features
OPTIONAL_DEPS = {
    'cupy': {
        'group': 'performance',
        'feature': 'GPU acceleration'
    },
    'dask': {
        'group': 'performance',
        'feature': 'parallel processing and chunked loading'
    },
    'holoviews': {
        'group': 'ui',
        'feature': 'interactive visualization'
    },
    'hvplot': {
        'group': 'ui',
        'feature': 'interactive plotting'
    },
    'tiled': {
        'group': 'bluesky',
        'feature': 'bluesky data access'
    },
    'databroker': {
        'group': 'bluesky',
        'feature': 'bluesky data access'
    },
    'PIL': {
        'group': 'io',
        'feature': 'image file loading'
    },
    'fabio': {
        'group': 'io',
        'feature': 'image file loading'
    },
    'h5py': {
        'group': 'io',
        'feature': 'HDF5/NEXUS file loading'
    },
    'pyFAI': {
        'group': 'processing',
        'feature': 'azimuthal integration'
    },
    'astropy': {
        'group': 'io',
        'feature': 'FITS file loading'
    },
    'scikit-image': {
        'group': 'processing',
        'feature': 'image processing'
    }
}

_warned_packages = set()

def check_optional_dependency(package_name):
    """
    Check if an optional dependency is available.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if package is available, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def requires_optional(package_name):
    """
    Decorator to mark functions that require optional dependencies.
    
    Args:
        package_name (str): Name of the required package
        
    Returns:
        callable: Decorated function that checks for the dependency
        
    Raises:
        ImportError: If the required package is not installed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not check_optional_dependency(package_name):
                dep_info = OPTIONAL_DEPS.get(package_name, {'group': 'unknown', 'feature': 'unknown'})
                if package_name not in _warned_packages:
                    warnings.warn(
                        f"The {dep_info['feature']} feature requires {package_name}, which is not installed. "
                        f"Install it with 'pip install pyhyperscattering[{dep_info['group']}]'",
                        ImportWarning,
                        stacklevel=2
                    )
                    _warned_packages.add(package_name)
                raise ImportError(
                    f"Cannot use {func.__name__}: {package_name} is required but not installed. "
                    f"Install it with 'pip install pyhyperscattering[{dep_info['group']}]'"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

def warn_if_missing(package_name):
    """
    Issue a warning if an optional package is missing, but only once per session.
    
    Args:
        package_name (str): Name of the package to check
    """
    if not check_optional_dependency(package_name) and package_name not in _warned_packages:
        dep_info = OPTIONAL_DEPS.get(package_name, {'group': 'unknown', 'feature': 'unknown'})
        warnings.warn(
            f"The {dep_info['feature']} feature requires {package_name}, which is not installed. "
            f"Install it with 'pip install pyhyperscattering[{dep_info['group']}]'",
            ImportWarning,
            stacklevel=2
        )
        _warned_packages.add(package_name)
