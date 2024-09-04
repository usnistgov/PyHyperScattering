
Scope and Package Outline
-------------------------
PyHyperScattering is a Python package for analyzing hyperspectral scattering data - that is, x-ray and neutron scattering data that is assessed as a series rather than as a set of discrete, disconnected points. 

The package is designed to provide useful tooling for end-to-end analysis using the XArray package, emphasizing the translation of bespoke, instrument-specific parameters to a common, interoperable XArray format. The package is designed to be modular and extensible, with a focus on the following key areas of data analysis:

1. Data Import and Translation "loading" (PyHyperScattering.load): Importing and translating raw data from a variety of sources into a common format.  Individual loaders are provided for a variety of instruments and a variety of input formats; tooling in PyHyperScattering.load.FileLoader provides a prototype for quickly building new file-based loaders by providing only an image loading function and a metadata loading function.  Intensity corrrections, where possible, are applied in the loading stage.

2. Data Reduction "integration" (PyHyperScattering.integrate): Reduction of raw data to a common format of I(q,chi) or in some cases I(qx,qy).  We endeavor to use other, high-quality, community supported reduction engines such as PyFAI, PyGIX, and scikit-image warp_polar.

3. Utility Modules "util": A collection of utility modules for such tasks as file input/output, beam centering and masking, and other common tasks.

4. Bespoke analyses "rsoxs" and others: A collection of bespoke analyses for specific measurement techniques.  These analyses are intended to be a 'cookbook' of common recipes for users to build their own scientific data pipelines.

PyHyperScattering is unapologetically API-first, a library rather than a program.  We enforce a strict separation of scientific logic code from user interfaces.  This is a choice to provide the code with the best possible chance to be reused in new ways not anticipated by the original development.  For instance, a data loading and reduction routine might be used in a Jupyter notebook, dispatched from a shell script, used to back a webapp using a framework like NiceGUI, used at-scale to reduce all data from a beamline via Prefect, etc.  By separating UI from science, we make it possible to have the same well-tested code back all these applications.
