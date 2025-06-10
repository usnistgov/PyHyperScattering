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

Consult ``pyproject.toml`` for additional groups such as ``bluesky`` or
``ui`` if you need those capabilities.
