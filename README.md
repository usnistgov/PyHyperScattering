![# PyHyperScattering](https://user-images.githubusercontent.com/875623/234083915-e62ee2d4-ad7f-4d91-a847-18fa54652ffc.png)

Python utilities for loading, reducing, slicing, and plotting hyperspectral scattering datasets.

This is a package approaching scope-completeness, but still under extremely active development and notably without any guarantee of API stability (we do try to not change things without reason, though). An increasing number of parts of it are unit-tested for functionality, stability, and/or scientific correctness, but large swaths are not. Its documentation is certainly lacking and we are actively seeking help in generating good documentation (see the Issues page on github for areas).  Use at your own risk.  If you're interested in contributing, please contact Peter Beaucage (peter.beaucage@nist.gov).

The (quite incomplete) documentation is located at https://pages.nist.gov/PyHyperScattering, and the tutorials in the repository are occasionally helpful.  Several core developers are active on the NIST RSoXS slack, Nikea, and NSLS2 slacks and welcome DMs with questions, or email Peter Beaucage.

Example data used in the tutorials and tests can be downloaded from our [GitHub release](https://github.com/usnistgov/PyHyperScattering/releases/tag/0.0.0-example-data).  See [Obtaining Example Data](docs/source/getting_started/example_data.rst) for more information.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/usnistgov/PyHyperScattering/HEAD)
![Unit Tests](https://github.com/usnistgov/PyHyperScattering/actions/workflows/main.yml/badge.svg)
![CodeQL](https://github.com/usnistgov/PyHyperScattering/actions/workflows/codeql-analysis.yml/badge.svg)

Legal
=====

NIST Disclaimer
---------------
Any identification of commercial or open-source software in this document is
done so purely in order to specify the methodology adequately. Such
identification is not intended to imply recommendation or endorsement by the
National Institute of Standards and Technology, nor is it intended to imply
that the softwares identified are necessarily the best available for the
purpose.

NIST License
------------
This software was developed by employees of the National Institute of Standards
and Technology (NIST), an agency of the Federal Government and is being made
available as a public service. Pursuant to title 17 United States Code Section
105, works of NIST employees are not subject to copyright protection in the
United States.  This software may be subject to foreign copyright.  Permission
in the United States and in foreign countries, to the extent that NIST may hold
copyright, to use, copy, modify, create derivative works, and distribute this
software and its documentation without fee is hereby granted on a non-exclusive
basis, provided that this notice and disclaimer of warranty appears in all
copies. 

THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER
EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY
THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM
INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE
SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT
SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT,
INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR
IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY,
CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR
PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT
OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
