.. _Development:

Development and Contributing Info
==============================================================

PyHyperScattering is an open-source collaboration maintained by the `National Institute of 
Standards and Technology (NIST) <https://www.nist.gov/>`_. This package is under active 
development, and the team welcome DMs with questions on the NIST RSoXS slack, Nikea, and NSLS2 slack 
channels, or by email to `Dr. Peter Beaucage <mailto:peter.beaucage@nist.gov>`_. For more information 
about contributing, development philosophy, and licensing, see :ref:`the Development page <Development>`.


Scope and Package Outline
-------------------------
PyHyperScattering is a Python package for analyzing hyperspectral scattering data - that is, x-ray and neutron scattering data that is assessed as a series rather than as a set of discrete, disconnected points. 

The package is designed to provide useful tooling for end-to-end analysis using the XArray package, emphasizing the translation of bespoke, instrument-specific parameters to a common, interoperable XArray format. The package is designed to be modular and extensible, with a focus on the following key areas of data analysis:

1. Data Import and Translation "loading" (PyHyperScattering.load): Importing and translating raw data from a variety of sources into a common format.  Individual loaders are provided for a variety of instruments and a variety of input formats; tooling in PyHyperScattering.load.FileLoader provides a prototype for quickly building new file-based loaders by providing only an image loading function and a metadata loading function.  Intensity corrrections, where possible, are applied in the loading stage.

2. Data Reduction "integration" (PyHyperScattering.integrate): Reduction of raw data to a common format of I(q,chi) or in some cases I(qx,qy).  We endeavor to use other, high-quality, community supported reduction engines such as PyFAI, PyGIX, and scikit-image warp_polar.

3. Utility Modules "util": A collection of utility modules for such tasks as file input/output, beam centering and masking, and other common tasks.

4. Bespoke analyses "rsoxs" and others: A collection of bespoke analyses for specific measurement techniques.  These analyses are intended to be a 'cookbook' of common recipes for users to build their own scientific data pipelines.

PyHyperScattering is unapologetically API-first, a library rather than a program.  We enforce a strict separation of scientific logic code from user interfaces.  This is a choice to provide the code with the best possible chance to be reused in new ways not anticipated by the original development.  For instance, a data loading and reduction routine might be used in a Jupyter notebook, dispatched from a shell script, used to back a webapp using a framework like NiceGUI, used at-scale to reduce all data from a beamline via Prefect, etc.  By separating UI from science, we make it possible to have the same well-tested code back all these applications.




Contributing
------------
Contributions are welcome! Please view our `Contributor Guidelines <https://github.com/usnistgov/PyHyperScattering/blob/main/CONTRIBUTING.md>`_ on github.

License
-------
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States. This software may be subject to foreign copyright. Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE. IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
With respect to the example data package, the following terms apply:
The data/work is provided by NIST as a public service and is expressly provided “AS IS.” NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any representations regarding the use of the data or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world.
You may improve, modify, and create derivative works of the data or any portion of the data, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the data and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the data:  Data citation recommendations are provided at https://www.nist.gov/open/license.
Permission to use this data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST’s creation of the data/work.

