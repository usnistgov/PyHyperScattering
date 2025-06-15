.. _installation:

Installation
============

You can install PyHyperScattering from PyPI using ``pip``:

.. code-block:: bash

   pip install PyHyperScattering

The project defines a number of optional dependency groups in
``pyproject.toml``.  These extras can be installed using the standard
``pip`` extras syntax.  For example, to install the optional
``grazing`` dependencies run:

.. code-block:: bash

   pip install PyHyperScattering[grazing]

The available extras include:

* ``grazing`` – packages for grazing-incidence scattering analysis
* ``bluesky`` – integrate with ``bluesky`` data sources like ``tiled``
* ``performance`` – acceleration packages such as ``pyopencl`` and ``dask``
* ``ui`` – interactive plotting tools including ``holoviews`` and ``hvplot``
* ``doc`` – dependencies needed to build the documentation
* ``dev`` – development utilities (linters, build helpers)
* ``test`` – full test suite requirements
* ``all-nogpu`` – everything except GPU-related packages
* ``all`` – install every optional dependency

Refer to ``pyproject.toml`` for the exact list of packages in each group.
