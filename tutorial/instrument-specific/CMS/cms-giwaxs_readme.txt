The general workflow to use these notebooks is as follows:

Create a folder for your analysis and copy these notebooks into them, then:
1) cms-giwaxs_poni_generation notebook to load your calibrant file(s) and 
   generate poni(s) needed for data processing. This notebook is incomplete and 
   has not been tested recently.  

2) cms-giwaxs_...procesing... to load the raw .tiff data, convert it all to 
   reciprocal space (cartesian and polar coordinates), and save as datasets 
   -> zarr stores. 

   For processing file sets of single images (no extra dimensions), it is 
   streamlined to use the single_images_to_dataset() method in the 
   PyHyperScattering.util.IntegrationUtils.CMSGIWAXS class.

3) cms-giwaxs...plotting... to load the zarr stores and plot the loaded xarrays
