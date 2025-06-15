.. _loading:

Loading: getting your data into PyHyperScattering
=================================================

PyHyperScattering organizes raw scattering images into `xarray` objects.
Image reading is handled by subclasses of :class:`~PyHyperScattering.FileLoader.FileLoader`.
The key method defined in ``FileLoader.py`` is ``loadFileSeries`` which
converts a folder of detector files into a single DataArray.

The simplified signature is::

    def loadFileSeries(self, basepath, dims, coords={}, file_filter=None,
                       md_filter={}, quiet=True, output_qxy=False,
                       dest_qx=None, dest_qy=None, output_raw=False,
                       image_slice=None):

It walks a directory, loads each file with ``loadSingleImage`` and stacks
frames along the requested dimensions.  Metadata keys listed in
``md_filter`` must be present for a frame to be included.  The returned
DataArray always contains ``pix_x`` and ``pix_y`` axes together with any
additional dimensions discovered from metadata or ``coords``.

Once an array is created you can write it to disk using the
``fileio`` accessor defined in ``FileIO.py``.  This accessor provides
convenience methods such as ``savePickle`` and ``saveNetCDF``::

    loaded = loader.loadFileSeries(my_path, dims=["energy"]) 
    loaded.fileio.saveNetCDF("run.nc")

The same accessor can also export to NeXus or Zarr formats while taking
care to sanitize attributes for serialization.

For curve fitting the package offers a ``fit`` accessor
implemented in ``Fitting.py``.  The ``apply`` method stacks all
dimensions except the chosen fit axis, applies a user supplied function
with progress bars, and returns the results as another xarray object::

    result = data.fit.apply(PyHyperScattering.Fitting.fit_lorentz_bg)

This design keeps reduction and analysis steps in the familiar xarray
workflow while hiding repetitive boilerplate.
