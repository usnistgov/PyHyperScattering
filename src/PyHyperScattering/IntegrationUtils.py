import warnings
import xarray as xr
import numpy as np
import math
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    from matplotlib.path import Path as MplPath
    import holoviews as hv
    import hvplot.xarray
    import skimage.draw
    import ipywidgets as widgets
    from IPython.display import display

except (ModuleNotFoundError,ImportError):
    warnings.warn('Could not import a dependency for interactive integration utils.  Install pyhyperscattering[ui] or pyhyperscattering[all].',stacklevel=2)
import pandas as pd

import json

class Check:
    '''
    Quick Utility to display a mask next to an image, to sanity check the orientation of e.g. an imported mask
    
    '''
    def checkMask(integrator,img,img_min=1,img_max=10000,img_scaling='log',alpha=1):
        '''
            draw an overlay of the mask and an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots(1,1)
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        ax.imshow(integrator.mask,origin='lower',alpha=alpha)
    def checkCenter(integrator,img,img_min=1,img_max=10000,img_scaling='log'):
        '''
            draw the beamcenter on an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots()
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        beamcenter = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 5, color='lawngreen')
        guide1 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 50, color='lawngreen',fill=False)
        guide2 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 150, color='lawngreen',fill=False)
        ax.add_patch(beamcenter)
        ax.add_patch(guide1)
        ax.add_patch(guide2)
    def checkAll(integrator,img,img_min=1,img_max=10000,img_scaling='log',alpha=1,d_inner=50,d_outer=150):
        '''
            draw the beamcenter and overlay mask on an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots()
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        beamcenter = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 5, color='lawngreen')
        guide1 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), d_inner, color='lawngreen',fill=False)
        guide2 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), d_outer, color='lawngreen',fill=False)
        ax.add_patch(beamcenter)
        ax.add_patch(guide1)
        ax.add_patch(guide2)
        ax.imshow(integrator.mask,origin='lower',alpha=alpha)


class DrawMask:
    '''
    Utility class for interactively drawing a mask in a Jupyter notebook.


    Usage: 

        Instantiate a DrawMask object using a PyHyper single image frame.

        Call DrawMask.ui() to generate the user interface

        Call DrawMask.mask to access the underlying mask, or save/load the raw mask data with .save or .load


    '''
    
    def __init__(self,frame, cmap='viridis', clim=(5e0, 5e3), width=800, height=700):
        '''
        Construct a DrawMask object

        Args:
            frame (xarray): a single data frame with pix_x and pix_y axes

        '''

        if len(frame.shape) > 2:
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)
            
        self.frame = frame
        
        self.fig = frame.hvplot(cmap=cmap, clim=clim, logz=True, data_aspect=1, 
                                width=width, height=height)

        self.poly = hv.Polygons([])
        self.path_annotator = hv.annotate.instance()

    def ui(self):
        '''
        Draw the DrawMask UI in a Jupyter notebook.


        Returns: the holoviews object

        '''
        print('Usage: click the "PolyAnnotator" tool at top right.  DOUBLE CLICK to start drawing a masked object, SINGLE CLICK to add a vertex, then DOUBLE CLICK to finish.  Click/drag individual vertex to adjust.')
        annotator_plot = self.path_annotator(
                                    self.fig * self.poly.opts(responsive=False), 
                                    annotations=['Label'], 
                                    vertex_annotations=['Value'])
        return annotator_plot.opts(toolbar='left')


    def save(self,fname):
        '''
        Save a parametric mask description as a json dump file.

        Args:
            fname (str): name of the file to save

        '''
        dflist = []
        for i in range(len(self.path_annotator.annotated)):
            dflist.append(self.path_annotator.annotated.iloc[i].dframe(['x','y']).to_json())
        
        with open(fname, 'w') as outfile:
            json.dump(dflist, outfile)
            
    def load(self,fname):
        '''
        Load a parametric mask description from a json dump file.

        Args:
            fname (str): name of the file to read from

        '''
        with open(fname,'r') as f:
            strlist = json.load(f)
        # print(strlist)
        dflist = []
        for item in strlist:
            dflist.append(pd.read_json(item))
        # print(dflist)
        self.poly = hv.Polygons(dflist)
        
        self.path_annotator(
                self.fig * self.poly.opts(
                            width=self.frame.shape[1], 
                            height=self.frame.shape[0], 
                            responsive=False), 
                annotations=['Label'], 
            vertex_annotations=['Value'])
        
        
    @property
    def mask(self):
        '''
        Render the mask as a numpy boolean array.
        '''
        mask = np.zeros(self.frame.shape).astype(bool)
        for i in range(len(self.path_annotator.annotated)):
            mask |= skimage.draw.polygon2mask(self.frame.shape,self.path_annotator.annotated.iloc[i].dframe(['x','y']))

        return mask


class DrawMaskMatplotlib:
    '''
    Interactive polygon mask tool built on matplotlib/ipympl, as an alternative to DrawMask
    for notebook environments where DrawMask's holoviews/bokeh-based .ui() does not render
    as an interactive widget.

    Usage:

        %matplotlib widget
        mask = DrawMaskMatplotlib(frame)
        mask.ui()  # or mask.ui(vmin=..., vmax=...) to override auto-scaling

        Left-click on the image to add vertices to the current region.
        Click "Close region" to commit it (needs >= 3 vertices) and start a new one.
        Click "Undo last vertex" / "Remove last region" to fix mistakes.

        mask.finish()                          # commit whichever region is still open
        mask.save(file_path)                   # write a mask description to a json file
        scan_to_integrate.mask = mask.mask      # boolean array, True = masked out

    The saved/loaded file format is the same one used by DrawMask.save()/.load() and by
    PFGeneralIntegrator.loadPyHyperMask, so files are interchangeable between the two tools.
    '''

    def __init__(self, frame):
        '''
        Construct a DrawMaskMatplotlib object

        Args:
            frame (xarray or ndarray): a single data frame with pix_x and pix_y axes

        '''
        self.image_data = np.squeeze(frame.to_numpy() if hasattr(frame, "to_numpy") else np.asarray(frame))
        if len(self.image_data.shape) > 2:
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)
        self.shape = self.image_data.shape  # (rows=pix_y, cols=pix_x)
        self.polygons = []       # list of Nx2 (x, y) vertex arrays, already committed
        self._current_verts = []
        self._current_artist = None
        self._fig = None
        self._ax = None
        self._cid = None
        self._status = None

    def ui(self, vmin=None, vmax=None):
        '''
        Draw the DrawMaskMatplotlib UI in a Jupyter notebook (requires %matplotlib widget).

        Args:
            vmin (float): optional lower bound for the log color scale (auto-scaled from the 1st percentile of positive finite pixels if omitted)
            vmax (float): optional upper bound for the log color scale (auto-scaled from the 99th percentile of positive finite pixels if omitted)

        '''
        fig, ax = plt.subplots(figsize=(6, 6))
        finite_positive = self.image_data[np.isfinite(self.image_data) & (self.image_data > 0)]
        if finite_positive.size and (vmin is None or vmax is None):
            auto_vmin, auto_vmax = np.percentile(finite_positive, [1, 99])
            if vmin is None:
                vmin = auto_vmin
            if vmax is None:
                vmax = auto_vmax
        if finite_positive.size and vmin is not None and vmax is not None and vmax > vmin:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None  # no usable positive data range; fall back to linear auto-scaling
        ax.imshow(self.image_data, norm=norm, origin="upper")
        ax.set_title("Left-click to add vertices; 'Close region' to finish")

        self._fig, self._ax = fig, ax
        self._current_verts = []
        (self._current_artist,) = ax.plot([], [], "o-", color="red", markersize=4)
        self._cid = fig.canvas.mpl_connect("button_press_event", self._on_click)

        close_button = widgets.Button(description="Close region")
        undo_button = widgets.Button(description="Undo last vertex")
        remove_button = widgets.Button(description="Remove last region")
        self._status = widgets.Label(value=f"Regions saved: {len(self.polygons)}")
        close_button.on_click(self._close_region)
        undo_button.on_click(self._undo_vertex)
        remove_button.on_click(self._remove_last)

        display(widgets.HBox([close_button, undo_button, remove_button]))
        display(self._status)
        plt.show()

    def _on_click(self, event):
        if event.inaxes != self._ax or event.button != 1 or event.xdata is None:
            return
        self._current_verts.append((event.xdata, event.ydata))
        self._redraw_current()

    def _redraw_current(self):
        xs, ys = zip(*self._current_verts) if self._current_verts else ([], [])
        self._current_artist.set_data(xs, ys)
        self._fig.canvas.draw_idle()

    def _undo_vertex(self, _btn=None):
        if self._current_verts:
            self._current_verts.pop()
            self._redraw_current()

    def _close_region(self, _btn=None):
        if len(self._current_verts) >= 3:
            self.polygons.append(np.array(self._current_verts))
            self._current_verts = []
            self._redraw_current()
            self._status.value = f"Regions saved: {len(self.polygons)}"

    def _remove_last(self, _btn=None):
        if self.polygons:
            self.polygons.pop()
            self._status.value = f"Regions saved: {len(self.polygons)}"

    def finish(self):
        '''
        Commit whichever region is currently being drawn (call before save()).
        '''
        self._close_region()

    @property
    def mask(self):
        '''
        Render the mask as a numpy boolean array.
        '''
        ny, nx = self.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask_arr = np.zeros(self.shape, dtype=bool)
        for verts in self.polygons:
            inside = MplPath(verts).contains_points(points).reshape(ny, nx)
            mask_arr |= inside
        return mask_arr

    def save(self, file_path):
        '''
        Save a parametric mask description as a json dump file, in the same format used by
        DrawMask.save() and consumed by PFGeneralIntegrator.loadPyHyperMask.

        Args:
            file_path (str): name of the file to save

        '''
        records = []
        for verts in self.polygons:
            xs, ys = verts[:, 0], verts[:, 1]
            records.append(json.dumps({
                "x": {str(i): float(x) for i, x in enumerate(xs)},
                "y": {str(i): float(y) for i, y in enumerate(ys)},
            }))
        with open(file_path, "w") as f:
            json.dump(records, f)

    def load(self, file_path):
        '''
        Load polygon vertices from an existing mask description file (reads DrawMask-format
        files too).

        Args:
            file_path (str): name of the file to read from

        '''
        with open(file_path, "r") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [json.dumps(raw)]
        self.polygons = []
        for item in raw:
            record = json.loads(item) if isinstance(item, str) else item
            xs = [record["x"][k] for k in sorted(record["x"], key=int)]
            ys = [record["y"][k] for k in sorted(record["y"], key=int)]
            self.polygons.append(np.column_stack((xs, ys)))


class CMSGIWAXS:
    """For streamlined loading for CMS data"""
    def __init__(self, files, loader, integrator):
        """
        Inputs: files: indexable object str or pathlib.Path filepaths to 
                       raw GIWAXS data
                loader: custom PyHyperScattering CMSGIWAXSLoader object, must 
                        return DataArray with attributes metadata
                integrator: instance of PGGeneralIntegrator object
        """
        self.files = files
        self.loader = loader
        self.integrator = integrator

    def single_images_to_dataset(self):
        """
        Method that takes a subscriptable object of filepaths corresponding to raw GIWAXS
        beamline data, loads the raw data into an xarray DataArray, generates pygix-transformed 
        cartesian and polar DataArrays, and creates 3 corresponding xarray Datasets 
        containing a DataArray per sample. 
        The raw dataarrays must contain the attributes 'scan_id' and 'incident_angle'

        Outputs: 2 Datasets: raw & reciprocal space (cartesian or polar based on integrator object)
        """
        # Select the first element of the sorted set outside of the for loop to initialize the xr.DataSet
        DA = self.loader.loadSingleImage(self.files[0])
        assert 'scan_id' in DA.attrs.keys(), "'scan_id' is a required attribute to use this function"

        # Update incident angle per sample:
        assert 'incident_angle' in DA.attrs.keys(), "'incident_angle' is a required attribute to use this function"
        self.integrator.incident_angle = float(DA.incident_angle[2:])

        # Integrate single image
        integ_DA = self.integrator.integrateSingleImage(DA)

        # Save coordinates for interpolating other dataarrays 
        integ_coords = integ_DA.coords

        # Create a DataSet, each DataArray will be named according to it's scan id
        raw_DS = DA.to_dataset(name=DA.scan_id)
        integ_DS = integ_DA.to_dataset(name=DA.scan_id)

        # Populate the DataSet with 
        for filepath in tqdm(self.files[1:], desc=f'Transforming Raw Data'):
            DA = self.loader.loadSingleImage(filepath)
            integ_DA = self.integrator.integrateSingleImage(DA)
            
            integ_DA = integ_DA.interp(integ_coords)

            raw_DS[f'{DA.scan_id}'] = DA
            integ_DS[f'{DA.scan_id}'] = integ_DA

        return raw_DS, integ_DS
