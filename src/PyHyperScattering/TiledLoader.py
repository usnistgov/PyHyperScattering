"""
Base class for instrument-specific Tiled data loaders.

This module provides a common base class for different NSLS-II beamline data loaders.
"""

import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import numpy as np
from tqdm.auto import tqdm
import asyncio
import time
import copy
import multiprocessing
import dask

try:
    os.environ["TILED_SITE_PROFILES"] = "/nsls2/software/etc/tiled/profiles"
    from tiled.client import from_profile, from_uri
    from httpx import HTTPStatusError
    import tiled
    try:
        # 2025-present databroker queries
        from bluesky_tiled_plugins.queries import RawMongo, Key, FullText, Contains, Regex
    except ImportError:
        from databroker.queries import RawMongo, Key, FullText, Contains, Regex

except Exception:
    print("Imports of some libraries needed for Tiled access failed. Install pyhyperscattering[bluesky].")


class TiledLoader:
    """
    Base loader for bluesky run xarrays from NSLS-II instruments.
    
    This serves as a base class for instrument-specific loaders like SST1RSoXSDB and SMITiled.
    """

    file_ext = ""
    md_loading_is_quick = True
    pix_size_1 = 0.06
    pix_size_2 = 0.06
    beamline_profile = "default" # this must be overridden by subclasses
    tiled_base_url = "https://tiled.nsls2.bnl.gov" # for BNL, but could be other facility
    # Default empty lookup table - to be overridden by subclasses
    md_lookup = {
        "sam_x": [],
        "sam_y": [],
        "sam_z": [],
        "sam_th": [],
        "polarization": [],
        "energy": [],
        "exposure": [],
        "time": ["time"],
    }

    search_output_value_library =  [
                ["scan_id", "scan_id", r"catalog.start", "default"],
                ["uid", "uid", r"catalog.start", "ext_bio"],
                ["start_time", "time", r"catalog.start", "default"],
                ["exit_status", "exit_status", r"catalog.stop", "default"],
                ["num_Images", "primary", r'catalog.stop["num_events"]', "default"],
            ] # basic set, to be appended to by subclasses

    def __init__(
        self,
        corr_mode=None,
        user_corr_fun=None,
        catalog=None,
        catalog_kwargs={},
        use_chunked_loading=False,
        suppress_time_dimension=True,
    ):
        """
        Initialize the base instrument loader.

        Args:
            corr_mode (str): origin to use for the intensity correction. Can be 'expt','i0','expt+i0','user_func','old',or 'none'
            user_corr_fun (callable): takes the header dictionary and returns the value of the correction.
            catalog (DataBroker Catalog): overrides the internally-set-up catalog with a version you provide
            catalog_kwargs (dict): kwargs to be passed to a from_profile catalog generation script.
            use_chunked_loading (bool): if True, returns Dask backed arrays for further Dask processing.
            suppress_time_dimension (bool): if True, time is never a dimension that you want in your data and will be dropped.
        """
        if corr_mode is None:
            warnings.warn(
                "Correction mode was not set, not performing *any* intensity corrections. Are you"
                " sure this is "
                + "right? Set corr_mode to 'none' to suppress this warning.",
                stacklevel=2,
            )
            self.corr_mode = "none"
        else:
            self.corr_mode = corr_mode
            
        if use_chunked_loading:
            catalog_kwargs["structure_clients"] = "dask"
        self.use_chunked_loading = use_chunked_loading
        
        if catalog is None:
            try:
                self.c = from_profile(self.beamline_profile, **catalog_kwargs)
            except tiled.profiles.ProfileNotFound:
                print(f'Could not directly connect to Tiled using the {self.beamline_profile} profile.\n  Making network connection.\n  Enter your BNL credentials now or pass an api key like catalog_kwargs={"api_key":"..."}.')
                self.c = from_uri(self.tiled_base_url, **catalog_kwargs)[self.beamline_profile]['raw']
        else:
            self.c = catalog
            if use_chunked_loading:
                raise SyntaxError(
                    "use_chunked_loading is incompatible with externally supplied catalog. When"
                    ' creating the catalog, pass structure_clients = "dask" as a kwarg.'
                )
            if len(catalog_kwargs) != 0:
                raise SyntaxError(
                    "catalog_kwargs is incompatible with externally supplied catalog. Pass those"
                    " kwargs to whoever gave you the catalog you passed in."
                )
        self.suppress_time_dimension = suppress_time_dimension
        dask.config.set(num_workers=min(12, multiprocessing.cpu_count()))  # Limit worker threads to avoid timeouts

    def runSearch(self, **kwargs):
        """
        Search the catalog using given commands.

        Args:
            **kwargs: passed through to the RawMongo search method of the catalog.

        Returns:
            result (obj): a catalog result object
        """
        q = RawMongo(**kwargs)
        return self.c.search(q)

    def searchCatalog(
        self,
        outputType: str = "default",
        cycle: str = None,
        proposal: str = None,
        saf: str = None,
        user: str = None,
        institution: str = None,
        project: str = None,
        sample: str = None,
        sampleID: str = None,
        plan: str = None,
        userOutputs: list = [],
        debugWarnings: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Search the Bluesky catalog for scans matching all provided keywords and return metadata as a dataframe.

        Matches are made based on the values in the top level of the 'start' dict within the metadata of each
        entry in the Bluesky Catalog (databroker.client.CatalogOfBlueskyRuns). Based on the search arguments provided,
        a pandas dataframe is returned where rows correspond to catalog entries (scans) and columns contain metadata.
        Several presets are provided for choosing which columns are generated, along with an interface for
        user-provided search arguments and additional metadata. Fails gracefully on bad user input/ changes to
        underlying metadata scheme.

        Args:
            outputType (str, optional): modulates the content of output columns in the returned dataframe
                'default' returns scan_id, start time, cycle, institution, project, sample_name, sample_id, plan name, detector,
                polarization, exit_status, and num_images
                'scans' returns only the scan_ids (1-column dataframe)
                'ext_msmt' returns default columns AND bar_spot, sample_rotation, polarization (_very_ slow)
                'ext_bio' returns default columns AND uid, saf, user_name
                'all' is equivalent to 'default' and all other additive choices
            cycle (str, optional): NSLS2 beamtime cycle, regex search e.g., "2022" matches "2022-2", "2022-1"
            proposal (int, optional): NSLS2 PASS proposal ID, numeric, exact match, e.g., 310176
            saf (str, optional): Safety Approval Form (SAF) number, exact match, e.g., "309441"
            user (str, optional): User name, case-insensitive, regex search e.g., "eliot" matches "Eliot", "Eliot Gann"
            institution (str, optional): Research Institution, case-insensitive, exact match, e.g., "NIST"
            project (str, optional): Project code, case-insensitive, regex search,
                e.g., "liquid" matches "Liquids", "Liquid-RSoXS"
            sample (str, optional): Sample name, case-insensitive, regex search, e.g., "BBP_" matches "BBP_PF902A"
            sampleID (str, optional): Sample ID, case-insensitive, regex search, e.g., "BBP_" matches "BBP_PF902A"
            plan (str, optional): Measurement Plan, case-insensitive, regex search,
                e.g., "Full" matches "full_carbon_scan_nd", "full_fluorine_scan_nd"
                e.g., "carbon|oxygen|fluorine" matches carbon OR oxygen OR fluorine scans
            userOutputs (list of lists, optional): Additional metadata to be added to output can be specified as a list of lists.
            debugWarnings (bool, optional): if True, raises a warning with debugging information whenever a key can't be found.
            
        Returns:
            Pandas dataframe containing the results of the search, or an empty dataframe if the search fails
        """
        # Pull in the reference to the databroker.client.CatalogOfBlueskyRuns attribute
        bsCatalog = self.c

        ### Part 1: Search the database sequentially, reducing based on matches to search terms
        # Plan the 'default' search through the keyword parameters, build list of [metadata ID, user input value, match type]
        defaultSearchDetails = [
            ["cycle", cycle, "case-insensitive"],
            ["proposal_id", proposal, "numeric"],
            ["saf_id", saf, "case-insensitive exact"],
            ["user_name", user, "case-insensitive"],
            ["institution", institution, "case-insensitive exact"],
            ["project_name", project, "case-insensitive"],
            ["sample_name", sample, "case-insensitive"],
            ["sample_id", sampleID, "case-insensitive"],
            ["plan_name", plan, "case-insensitive"],
        ]

        outputValueLibrary = self.search_output_value_library

        # Pull any user-provided search terms
        userSearchList = []
        for userLabel, value in kwargs.items():
            # Minimial check for bad user input
            if isinstance(value, str):
                userSearchList.append([userLabel, value, ""])
            elif isinstance(value, int) or isinstance(value, float):
                userSearchList.append([userLabel, value, "numeric"])
            elif isinstance(value, list) and len(value) == 2:
                userSearchList.append([userLabel, value[0], value[1]])
            else:  # bad user input
                raise ValueError(
                    "Error parsing a keyword search term, check the format. Skipped argument:"
                    f" {value} "
                )

        # combine the lists of lists
        fullSearchList = defaultSearchDetails + userSearchList

        df_SearchDet = pd.DataFrame(
            fullSearchList, columns=["Metadata field:", "User input:", "Search scheme:"]
        )

        # Iterate through search terms sequentially, reducing the size of the catalog based on successful matches
        reducedCatalog = bsCatalog
        for _, searchSeries in df_SearchDet.iterrows():
            # Skip arguments with value None, and quits if the catalog was reduced to 0 elements
            if (searchSeries.iloc[1] is not None) and (len(reducedCatalog) > 0):
                # For numeric entries, do Key equality
                if "numeric" in str(searchSeries.iloc[2]):
                    reducedCatalog = reducedCatalog.search(
                        Key(searchSeries.iloc[0]) == float(searchSeries.iloc[1])
                    )

                else:  # Build regex search string
                    reg_prefix = ""
                    reg_postfix = ""

                    # Regex cheatsheet:
                    # (?i) is case insensitive
                    # ^_$ forces exact match to _, ^ anchors the start, $ anchors the end
                    if "case-insensitive" in str(searchSeries.iloc[2]):
                        reg_prefix += "(?i)"
                    if "exact" in searchSeries.iloc[2]:
                        reg_prefix += "^"
                        reg_postfix += "$"

                    regexString = reg_prefix + str(searchSeries.iloc[1]) + reg_postfix

                    # Search/reduce the catalog
                    reducedCatalog = reducedCatalog.search(Regex(searchSeries.iloc[0], regexString))

                # If a match fails, notify the user which search parameter yielded 0 results
                if len(reducedCatalog) == 0:
                    warnString = (
                        f"No results found when searching {str(searchSeries.iloc[0])}. "
                        + f"If this is a user-provided search parameter, check spelling/syntax."
                    )
                    warnings.warn(warnString, stacklevel=2)
                    return pd.DataFrame()

        ### Part 2: Build and return output dataframe

        if outputType == "scans":
            # Branch 2.1, if only scan IDs needed, build and return a 1-column dataframe
            scan_ids = []
            for scanEntry in tqdm(reducedCatalog.values(), desc="Building scan list"):
                scan_ids.append(scanEntry.start["scan_id"])
            return pd.DataFrame(scan_ids, columns=["Scan ID"])

        else:  # Branch 2.2, Output metadata from a variety of sources within each the catalog entry
            missesDuringLoad = False
            # Store details of output values as a list of lists
            # List elements are [Output Column Title, Bluesky Metadata Code, Metadata Source location, Applicable Output flag]


            # Subset the library based on the output flag selected
            activeOutputValues = []
            activeOutputLabels = []
            for outputEntry in outputValueLibrary:
                if (
                    (outputType == "all")
                    or (outputEntry[3] == outputType)
                    or (outputEntry[3] == "default")
                ):
                    activeOutputValues.append(outputEntry)
                    activeOutputLabels.append(outputEntry[0])

            # Add any user-provided Output labels
            userOutputList = []
            for userOutEntry in userOutputs:
                # Minimial check for bad user input
                if isinstance(userOutEntry, list) and len(userOutEntry) == 3:
                    activeOutputValues.append(userOutEntry)
                    activeOutputLabels.append(userOutEntry[0])
                else:  # bad user input
                    raise ValueError(
                        (
                            f"Error parsing user-provided output request {userOutEntry}, check the"
                            " format."
                        ),
                        stacklevel=2,
                    )

            # Add any user-provided search terms
            for userSearchEntry in userSearchList:
                activeOutputValues.append(
                    [
                        userSearchEntry[0],
                        userSearchEntry[0],
                        r"catalog.start",
                        "default",
                    ]
                )
                activeOutputLabels.append(userSearchEntry[0])

            # Build output dataframe as a list of lists
            outputList = []

            # Outer loop: Catalog entries
            for scanEntry in tqdm(reducedCatalog.values(), desc="Retrieving results..."):
                singleScanOutput = []

                # Pull the start and stop docs once
                currentCatalogStart = scanEntry.start
                currentCatalogStop = scanEntry.stop

                currentScanID = currentCatalogStart["scan_id"]

                # Inner loop: append output values
                for outputEntry in activeOutputValues:
                    outputVariableName = outputEntry[0]
                    metaDataLabel = outputEntry[1]
                    metaDataSource = outputEntry[2]

                    try:  # Add the metadata value depending on where it is located
                        if metaDataLabel == "time":
                            singleScanOutput.append(
                                datetime.datetime.fromtimestamp(currentCatalogStart["time"])
                            )
                            # see Zen of Python # 9,8 for justification
                        elif metaDataSource == r"catalog.start":
                            singleScanOutput.append(currentCatalogStart[metaDataLabel])
                        elif metaDataSource == r'catalog.start["plan_args"]':
                            singleScanOutput.append(
                                currentCatalogStart["plan_args"][metaDataLabel]
                            )
                        elif metaDataSource == r"catalog.stop":
                            singleScanOutput.append(currentCatalogStop[metaDataLabel])
                        elif metaDataSource == r'catalog.stop["num_events"]':
                            singleScanOutput.append(
                                currentCatalogStop["num_events"][metaDataLabel]
                            )
                        elif metaDataSource == r'catalog.baseline["data"]':
                            singleScanOutput.append(
                                scanEntry.baseline["data"][metaDataLabel].__array__().mean()
                            )
                        else:
                            if debugWarnings:
                                warnings.warn(
                                    (
                                        f"Failed to locate metadata for {outputVariableName} in"
                                        f" scan {currentScanID}."
                                    ),
                                    stacklevel=2,
                                )
                            missesDuringLoad = True

                    except (KeyError, TypeError):
                        if debugWarnings:
                            warnings.warn(
                                (
                                    f"Failed to locate metadata for {outputVariableName} in scan"
                                    f" {currentScanID}."
                                ),
                                stacklevel=2,
                            )
                        missesDuringLoad = True
                        singleScanOutput.append("N/A")

                # Append to the filled output list for this entry to the list of lists
                outputList.append(singleScanOutput)

            # Convert to dataframe for export
            if missesDuringLoad:
                warnings.warn(
                    (
                        f'One or more missing field(s) during this load were replaced with "N/A". '
                        f" Re-run with debugWarnings=True to see details."
                    ),
                    stacklevel=2,
                )
            return pd.DataFrame(outputList, columns=activeOutputLabels)

    def background(f):
        """Decorator to run a function in the background."""
        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
        return wrapped

    @background
    def do_list_append(run, scan_ids, sample_ids, plan_names, uids, npts, start_times):
        """Helper function to append run data to lists."""
        doc = run.start
        scan_ids.append(doc["scan_id"])
        sample_ids.append(doc["sample_id"])
        plan_names.append(doc["plan_name"])
        uids.append(doc["uid"])
        try:
            npts.append(run.stop["num_events"]["primary"])
        except (KeyError, TypeError):
            npts.append(0)
        start_times.append(doc["time"])

    def loadSeries(
        self,
        run_list,
        meta_dim,
        loadrun_kwargs={},
    ):
        """
        Loads a series of runs into a single xarray object, stacking along meta_dim.

        Useful for a set of samples, or a set of polarizations, etc., taken in different scans.

        Args:
            run_list (list): list of scan ids to load
            meta_dim (str): dimension to stack along. must be a valid attribute/metadata value, such as polarization or sample_name
            loadrun_kwargs (dict): kwargs to be passed to loadRun

        Returns:
            raw: xarray.Dataset with all scans stacked
        """
        scans = []
        axes = []
        label_vals = []
        for run in run_list:
            loaded = self.loadRun(self.c[run], **loadrun_kwargs).unstack("system")
            axis = list(loaded.indexes.keys())
            try:
                axis.remove("pix_x")
                axis.remove("pix_y")
            except ValueError:
                pass
            try:
                axis.remove("qx")
                axis.remove("qy")
            except ValueError:
                pass
            axes.append(axis)
            scans.append(loaded)
            label_val = loaded.__getattr__(meta_dim)
            try:
                if len(label_val) > 1 and type(label_val) != str:
                    label_val = label_val.mean()
            except TypeError:
                pass  # assume if there is no len, then this is a single value and everything is fine
            label_vals.append(label_val)
        assert len(axes) == axes.count(
            axes[0]
        ), f"Error: not all loaded data have the same axes. This is not supported yet.\n {axes}"
        axes[0].insert(0, meta_dim)
        new_system = axes[0]
        return (
            xr.concat(scans, dim=meta_dim)
            .assign_coords({meta_dim: label_vals})
            .stack(system=new_system)
        )

    def peekAtMd(self, run):
        """
        Peek at the metadata in a run.
        
        Args:
            run: BlueskyRun object or scan id
            
        Returns:
            md: Dictionary of metadata
        """
        if isinstance(run, int):
            run = self.c[run]
        md = {}
        md["start"] = dict(run.start)
        try:
            md["stop"] = dict(run.stop)
        except AttributeError:
            md["stop"] = {}
        return md

    def loadMd(self, run):
        """
        Return a dict of metadata entries from the databroker run xarray.
        
        Args:
            run: BlueskyRun object or scan id
            
        Returns:
            md: Dictionary of metadata
        """
        if isinstance(run, int):
            run = self.c[run]
            
        md = self.peekAtMd(run)

        # Set pixel size from detector type
        # Set pixel size first from md  
        # (if not avail, set from detector type)
        try:
            pix_size_x = md["start"]["pix_size_x"]
            pix_size_y = md["start"]["pix_size_y"]
        except (KeyError, ValueError):
            pix_size_x = self.pix_size_1
            pix_size_y = self.pix_size_2

        # Add convenience items to the md dict
        md["detector"] = md["start"].get("RSoXS_Main_DET", None)
        
        # Get items from md_lookup if detector not in md
        if not md["detector"]:
            # Try to find detector from primary data streams
            try:
                for key in run.primary.keys():
                    if "_image" in key:
                        md["detector"] = key.replace("_image", "")
                        break
            except:
                pass
        
        # Add standard convenience entries to md
        md["pix_size_x"] = pix_size_x
        md["pix_size_y"] = pix_size_y
        
        # Get values for standard metadata keys using the lookup table
        for name, possible_labels in self.md_lookup.items():
            for label in possible_labels:
                try:
                    md[name] = md["start"][label]
                    break
                except (KeyError, ValueError):
                    continue

        return md

    def loadMonitors(
        self,
        entry,
        integrate_onto_images: bool = True,
        useShutterThinning: bool = True,
        n_thinning_iters: int = 1,
        directLoadPulsedMonitors: bool = True
    ):
        """
        Load the monitor streams for entry.

        Creates a dataset containing all monitor streams (e.g., Mesh Current, etc.) as data variables mapped
        against time. Optionally, all streams can be indexed against the primary measurement time for the images using
        integrate_onto_images.

        Parameters
        ----------
        entry : databroker.client.BlueskyRun
            Bluesky Document containing run information.
        integrate_onto_images : bool, optional
            whether or not to map timepoints to the image measurement times, by default True
        useShutterThinning : bool, optional
            Whether or not to attempt to thin the raw time streams to remove data collected during shutter opening/closing
        n_thinning_iters : int, optional
            how many iterations of thinning to perform, by default 1
        directLoadPulsedMonitors : bool, optional
            Whether or not to load the pulsed monitors using direct reading, by default True

        Returns
        -------
        xr.Dataset
            xarray dataset containing all monitor streams as data variables mapped against the dimension "time"
        """
        # This is a complex method that needs to be implemented by each subclass
        # to handle the specific monitor data structure of each beamline
        raise NotImplementedError("loadMonitors must be implemented by subclasses")

    def loadRun(
        self,
        run,
        dims=None,
        coords={},
        return_dataset=False,
        useMonitorShutterThinning=True,
    ):
        """
        Loads a run entry from a catalog result into a raw xarray.

        Args:
            run (DataBroker result, int of a scan id, list of scan ids): a single run from BlueSky
            dims (list): list of dimensions you'd like in the resulting xarray.
            coords (dict): user-supplied dimensions, see syntax examples in documentation.
            return_dataset (bool, default False): return both the data and the monitors as a xr.dataset.
            useMonitorShutterThinning (bool): Whether to thin monitor data based on shutter timing.
            
        Returns:
            raw (xarray): raw xarray containing your scan in PyHyper-compliant format
        """
        # This method needs to be implemented by each subclass
        # to handle the specific data loading for each beamline
        raise NotImplementedError("loadRun must be implemented by subclasses")
