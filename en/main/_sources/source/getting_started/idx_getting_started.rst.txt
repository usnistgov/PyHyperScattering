.. _Getting_Started:

Getting Started
=================

.. toctree::
   :maxdepth: 1
   :hidden:
   
   Loading Data <loading>
   Integration <integration>
   Custom Analysis <learning-to-fly>
   Utilities <utilities>

PyHyperScattering aims to make working with hyperspectral x-ray and neutron scattering data easy, to make programs that work with such data a combination of simple, logical commands with minimal 'cruft'.  In the era of modern computing, there is no reason you should have to think about for loops and how you're storing different intermediate data products - you should be able to go immediately from raw data to an analysis with clear commands, punch down to the data you need for your science quickly.  The goal is for these tools to make the mechanics of hyperspectral scattering easier and in so doing, more reproducible, explainable, and robust.

It grew out of the NIST RSoXS program, but aims to be broadly applicable.  If it's scattering data, you should be able to load it, reduce it, and slice it.

Run tutorials using `Binder: <https://mybinder.org/v2/gh/usnistgov/PyHyperScattering/HEAD>`_


The tools PyHyper provides are basically divided into three categories:


:ref:`Loading Data <loading>`
-------------------------------
Get your data from a source - files on disk or a network connection to a DataBroker or whatever -  into a standardized structure we call a raw xarray.
The metadata should be loaded and converted to a standardized set of terms, intensity corrections applied, and the frames stacked along whatever dimensions you want.



:ref:`Integration <integration>`
-----------------------------------
Convert your data from pixel space or qx/qy space to chi-q space - perfect for slicing.  generally tools here are built on pyFAI, though some variants also use warp_polar from numpy.


:ref:`Custom Analysis <analysis>`
------------------------------------
Slice your data along specified dimensions and visualize the results.


:ref:`Utilities <utilities>`
--------------------------------
These are pre-canned routines for common analyses, such as RSoXS anisotropic ratio or peak/background fitting.  the intent here is that the barrier to building your own code is low, and contributions in module space are especially welcomed and encouraged.

Have fun!