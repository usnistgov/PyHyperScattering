.. _User_Guide:

User Guide
===========
.. toctree::
   :maxdepth: 1
   :hidden:
   
   Catalog Search <search_catalog>

Work in progress cookbook for specific tasks.

:ref:`Catalog Search <search_catalog>`
-------------------------------
When using PyHyperScattering to load Bluesky catalog files through tiled (i.e., in the SST1RSoXSDB module), one of the first steps is to select which scans to load. The searchCatalog function provides this functionality and returns a pandas DataFrame with all the selected scans.