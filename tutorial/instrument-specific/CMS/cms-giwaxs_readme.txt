The general workflow to use these notebooks is as follows:

Create a folder for your analysis and copy these notebooks into them, then:
1) cms-giwaxs_poni_generation notebook to load your calibrant file(s) and generate poni(s) needed for data processing
2) cms-giwaxs_...procesing... to load the raw .tiff data, convert it all to reciprocal space (cartesian and polar coordinates), and save as datasets -> zarr stores
3) cms-giwaxs...plotting... to load the zarr stores and plot the data however is desired

