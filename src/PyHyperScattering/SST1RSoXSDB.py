from PIL import Image
import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.ndimage
import asyncio
import time
import copy
import multiprocessing

try:
    os.environ["TILED_SITE_PROFILES"] = "/nsls2/software/etc/tiled/profiles"
    from tiled.client import from_profile,from_uri
    from httpx import HTTPStatusError
    import tiled
    import dask
    try: 
        from bluesky_tiled_plugins.queries import RawMongo, Key, FullText, Contains, Regex # 2025-present databroker queries
    except ImportError: 
        from databroker.queries import RawMongo, Key, FullText, Contains, Regex

except Exception:
    print("Imports of some libraries needed for SST-1 RSoXS failed.  If you are trying to use SST-1 RSoXS, install pyhyperscattering[bluesky].")

import copy


class SST1RSoXSDB:
    """
    Loader for bluesky run xarrays form NSLS-II SST1 RSoXS instrument


    """

    file_ext = ""
    md_loading_is_quick = True
    pix_size_1 = 0.06
    pix_size_2 = 0.06

    # List of metadata key names used historically at SST1 RSoXS
    md_lookup = {

        "sam_x": ["solid_sample_x", # Started using ~March 2025
                  "manipulator_x", # Started using ~January 2025
                 "RSoXS Sample Outboard-Inboard",
                 ],
        "sam_y": ["solid_sample_y", # Started using ~March 2025
                  "manipulator_y", # Started using ~January 2025
                  "RSoXS Sample Up-Down",
                 ],
        "sam_z": ["solid_sample_z", # Started using ~March 2025
                  "manipulator_z", # Started using ~January 2025
                  "RSoXS Sample Downstream-Upstream",
                 ],
        "sam_th": ["solid_sample_r", # Started using ~March 2025
                   "manipulator_r", # Started using ~January 2025
                   "RSoXS Sample Rotation",
                  ],
        "polarization": ["en_polarization_setpoint",
                        ],
        "energy": ["en_energy_setpoint",
                   "en_monoen_setpoint",
                  ],
        "exposure": ["RSoXS Shutter Opening Time (ms)", 
                    ],
        "time": ["time",
                     ],

    }
    

    def __init__(
        self,
        corr_mode=None,
        user_corr_fun=None,
        dark_subtract=True,
        dark_pedestal=0,
        exposure_offset=0,
        catalog=None,
        catalog_kwargs={},
        use_precise_positions=False,
        use_chunked_loading=False,
        suppress_time_dimension=True,
    ):
        """
        Args:
            corr_mode (str): origin to use for the intensity correction.  Can be 'expt','i0','expt+i0','user_func','old',or 'none'
            user_corr_func (callable): takes the header dictionary and returns the value of the correction.
            dark_pedestal (numeric): value to add to the whole image before doing dark subtraction, to avoid non-negative values.
            exposure_offset (numeric): value to add to the exposure time.
            catalog (DataBroker Catalog): overrides the internally-set-up catalog with a version you provide
            catalog_kwargs (dict): kwargs to be passed to a from_profile catalog generation script.  For example, you can ask for Dask arrays here.
            use_precise_positions (bool): if False, rounds sam_x and sam_y to 1 digit.  If True, keeps default rounding (4 digits).  Needed for spiral scans to work with readback positions.
            use_chunked_loading (bool): if True, returns Dask backed arrays for further Dask processing.  if false, behaves in conventional Numpy-backed way
            suppress_time_dimension (bool): if True, time is never a dimension that you want in your data and will be dropped (default).  if False, time will be a dimension in almost every scan.
        """

        if corr_mode == None:
            warnings.warn(
                "Correction mode was not set, not performing *any* intensity corrections.  Are you"
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
                self.c = from_profile("rsoxs", **catalog_kwargs)
            except tiled.profiles.ProfileNotFound:
                print('could not directly connect to Tiled using a system profile.\n  Making network connection.\n  Enter your BNL credentials now or pass an api key like catalog_kwargs={"api_key":"..."}.')
                self.c = from_uri('https://tiled.nsls2.bnl.gov',**catalog_kwargs)['rsoxs']['raw']
        else:
            self.c = catalog
            if use_chunked_loading:
                raise SyntaxError(
                    "use_chunked_loading is incompatible with externally supplied catalog.  when"
                    ' creating the catalog, pass structure_clients = "dask" as a kwarg.'
                )
            if len(catalog_kwargs) != 0:
                raise SyntaxError(
                    "catalog_kwargs is incompatible with externally supplied catalog.  pass those"
                    " kwargs to whoever gave you the catalog you passed in."
                )
        self.dark_subtract = dark_subtract
        self.dark_pedestal = dark_pedestal
        self.exposure_offset = exposure_offset
        self.use_precise_positions = use_precise_positions
        self.suppress_time_dimension = suppress_time_dimension

        self.catalog_df = None
        self.catalog_df_kwargs = None

        dask.config.set(num_workers=min(12,multiprocessing.cpu_count())) # this limits the number of worker threads, which should help avoid timeouts on some large data

        
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

    def summarize_run(self, *args, **kwargs):
        """Deprecated function for searching the bluesky catalog for a run. Replaced by searchCatalog()

        To be removed in PyHyperScattering 1.0.0+.
        """
        warnings.warn(
            (
                "summarize_run has been renamed to searchCatalog.  This will stop working in"
                " PyHyperScattering 1.0.0 and later."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.searchCatalog(*args, **kwargs)

    def browseCatalog(self,force_refresh = False,**kwargs):
        """
        Browse the catalog.

        Args:
            **kwargs: passed through to searchCatalog, and bounds the set of runs fetched/displayed.

        Returns:
            result (obj): an ipyaggrid instance to browse the catalog.

        """
        from ipyaggrid import Grid
        if self.catalog_df is None or self.catalog_df_kwargs != kwargs or force_refresh:
            self.catalog_df = self.searchCatalog(**kwargs)
            self.catalog_df_kwargs = kwargs
        else:
            print(f'Not updating stored dataframe with kwargs {kwargs}')
        column_names = []
        
        pretty_names = {
            "scan_id": "Scan ID",
            "start_time": "Start Time",
            "cycle": "Cycle",
            "institution": "Institution",
            "project": "Project",
            "sample_name": "Sample Name",
            "sample_id": "Sample ID",
            "plan": "Plan Name",
            "detector": "Detector",
            "polarization": "Polarization",
            "exit_status": "Exit Status",
            "num_Images": "# Imgs/Pts",
        }
        additional_options = {
            "scan_id": {"width": 150, "sortable": True, "sort": "desc", "lockPosition": "left"},
            "institution": {"width": 125},
            "project": {"width": 125},
            "cycle": {"width": 125},
            "exit_status": {"width": 125},
            "num_Images": {"width": 125},
            "plan": {"filter":"agMultiColumnFilter","filters":['textFilter','setFilter']},
            "project": {"filter":"agMultiColumnFilter","filters":['textFilter','setFilter']},
            "sample_name":{"filter":"agMultiColumnFilter","filters":['textFilter','setFilter']},
                        }

        for field in self.catalog_df.columns:
            col_config =  {'field': field}
            if field in pretty_names:
                col_config['headerName'] = pretty_names[field]
            if field in additional_options:
                col_config.update(additional_options[field])
            column_names.append(col_config)


        grid = Grid(
            grid_data=self.catalog_df,
            grid_options={
                "columnDefs": column_names,
                "enableSorting": True,
                "enableFilter": True,
                "enableColResize": True,
                "enableRangeSelection": True,
                "rowSelection": "multiple",
#                "pagination": True,
#                "paginationPageSize": 100,
                'defaultColDef':
                {'sortable':True,
                'resizable':True,
                'floatingFilter':True,},
                'autoSizeStrategy': {'type':'fitCellContents'},

            },
            quick_filter=True,
            theme="ag-theme-bootstrap",
        )


        return grid

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
        scan_id: int = None,
        userOutputs: list = [],
        debugWarnings: bool = False,
        existingCatalog: pd.DataFrame = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Search the Bluesky catalog for scans matching all provided keywords and return metadata as a dataframe.

        Matches are made based on the values in the top level of the 'start' dict within the metadata of each
        entry in the Bluesky Catalog (databroker.client.CatalogOfBlueskyRuns). Based on the search arguments provided,
        a pandas dataframe is returned where rows correspond to catalog entries (scans) and columns contain  metadata.
        Several presets are provided for choosing which columns are generated, along with an interface for
        user-provided search arguments and additional metadata. Fails gracefully on bad user input/ changes to
        underlying metadata scheme.

        Ex1: All of the carbon,fluorine,or oxygen scans for a single sample series in the most recent cycle:
            bsCatalogReduced4 = db_loader.searchCatalog(sample="bBP_", institution="NIST", cycle = "2022-2", plan="carbon|fluorine|oxygen")

        Ex2: Just all of the scan Ids for a particular sample:
            bsCatalogReduced4 = db_loader.searchCatalog(sample="BBP_PFP09A", outputType='scans')

        Ex3: Complex Search with custom parameters
            bsCatalogReduced3 = db_loader.searchCatalog(['angle', '-1.6', 'numeric'], outputType='all',sample="BBP_", cycle = "2022-2",
            institution="NIST",plan="carbon", userOutputs = [["Exposure Multiplier", "exptime", r'catalog.start'], ["Stop
            Time","time",r'catalog.stop']])

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
            scan_id (int, optional): Scan ID, exact numeric match, e.g., 12345
            **kwargs: Additional search terms can be provided as keyword args and will further filter
                the catalog Valid input follows metadataLabel='searchTerm' or metadataLavel = ['searchTerm','matchType'].
                Metadata labels must match an entry in the 'start' dictionary of the catalog. Supported match types are
                combinations of 'case-insensitive', 'case-sensitive', and 'exact' OR 'numeric'. Default behavior is to
                do a case-sensitive regex match. For metadata labels that are not valid python names, create the kwarg
                dict before passing into the function (see example 3). Additional search terms will appear in the
                output data columns.
                Ex1: passing in cycle='2022' would match 'cycle'='2022-2' AND 'cycle='2022-1'
                Ex2: passing in grazing=[0,'numeric'] would match grazing==0
                Ex3: create kwargs first, then pass it into the function.
                    kwargs = {'2weird metadata label': "Bob", 'grazing': 0, 'angle':-1.6}
                    db_loader.searchCatalog(sample="BBP_PFP09A", outputType='scans', **kwargs)
            userOutputs (list of lists, optional): Additional metadata to be added to output can be specified as a list of lists. Each
                sub-list specifies a metadata field as a 3 element list of format:
                [Output column title (str), Metadata label (str), Metadata Source (raw str)],
                Valid options for the Metadata Source are any of [r'catalog.start', r'catalog.start["plan_args"], r'catalog.stop',
                r'catalog.stop["num_events"]']
                e.g., userOutputs = [["Exposure Multiplier","exptime", r'catalog.start'], ["Stop Time","time",r'catalog.stop']]
            debugWarnings (bool, optional): if True, raises a warning with debugging information whenever a key can't be found.
            existingCatalog (pd.Dataframe, optional): if provided, results with scan_id that appear in this dataframe and equal number of points will not be re-downloaded.
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
            ["scan_id", scan_id, "numeric"],
        ]

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
                    "Error parsing a keyword search term, check the format.  Skipped argument:"
                    f" {value} "
                )

        # combine the lists of lists
        fullSearchList = defaultSearchDetails + userSearchList

        # df_SearchDet = pd.DataFrame(
        #     fullSearchList, columns=["Metadata field:", "User input:", "Search scheme:"]
        # )

        # Iterate through search terms sequentially, reducing the size of the catalog based on successful matches

        reducedCatalog = bsCatalog
        for searchSeries in fullSearchList:
            # Skip arguments with value None, and quits if the catalog was reduced to 0 elements
            if (searchSeries[1] is not None) and (len(reducedCatalog) > 0):
                # For numeric entries, do Key equality
                if "numeric" in str(searchSeries[2]):
                    reducedCatalog = reducedCatalog.search(
                        Key(searchSeries[0]) == float(searchSeries[1])
                    )

                else:  # Build regex search string
                    reg_prefix = ""
                    reg_postfix = ""

                    # Regex cheatsheet:
                    # (?i) is case insensitive
                    # ^_$ forces exact match to _, ^ anchors the start, $ anchors the end
                    if "case-insensitive" in str(searchSeries[2]):
                        reg_prefix += "(?i)"
                    if "exact" in searchSeries[2]:
                        reg_prefix += "^"
                        reg_postfix += "$"

                    regexString = reg_prefix + str(searchSeries[1]) + reg_postfix

                    # Search/reduce the catalog
                    reducedCatalog = reducedCatalog.search(Regex(searchSeries[0], regexString))

                # If a match fails, notify the user which search parameter yielded 0 results
                if len(reducedCatalog) == 0:
                    warnString = (
                        f"No results found when searching {str(searchSeries[0])}. "
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
            outputValueLibrary = [
                ["scan_id", "scan_id", r"catalog.start", "default"],
                ["start_time", "time", r"catalog.start", "default"],
                ["cycle", "cycle", r"catalog.start", "default"],
                ["saf", "SAF", r"catalog.start", "ext_bio"],
                ["user_name", "user_name", r"catalog.start", "ext_bio"],
                ["institution", "institution", r"catalog.start", "default"],
                ["project", "project_name", r"catalog.start", "default"],
                ["sample_name", "sample_name", r"catalog.start", "default"],
                ["sample_id", "sample_id", r"catalog.start", "default"],
                ["bar_spot", "bar_spot", r"catalog.start", "ext_msmt"],
                ["plan", "plan_name", r"catalog.start", "default"],
                ["detector", "RSoXS_Main_DET", r"catalog.start", "default"],
                ["polarization", "en_polarization", r'catalog.baseline["data"]', "ext_msmt"],
                ["sample_rotation", "angle", r"catalog.start", "ext_msmt"],
                ["exit_status", "exit_status", r"catalog.stop", "default"],
                ["num_Images", "primary", r'catalog.stop["num_events"]', "default"],
                ["uid", "uid", r"catalog.start", "default"],
            ]

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
            for scanEntry in tqdm(reducedCatalog.items(), desc="Retrieving results"):
                singleScanOutput = []

                if existingCatalog is not None:
                    if scanEntry[0] in existingCatalog.uid.values:
                        # if the scan is already in the catalog, skip it
                        continue
                
                scanEntry = scanEntry[1]

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


    def findAppropriateDiodes(self,run_to_find,cat_diodes=None, diode_name='diode', same_cycle=True,time_cutoff_days = 3.0):
        '''
        Finds appropriate diode scans for a given run.
        "Appropriate" scans are somewhat controlled by the kwargs to this function, but in general:
            - same detector
            - same cycle
            - same edge
            - within 3 days of the run

        Args:
            run_to_find (pd.DataFrame or numeric): a dataframe with a single row of the run you are trying to find a diode for, or the scan_id
                        (if scan_id it will be loaded using searchCatalog)
            cat_diodes (pd.DataFrame): a dataframe of diode scans to search through.  If None, will search the catalog for diode scans.
            diode_name (str): the sample_name of the diode to use in search.  Default is 'diode'.
            same_cycle (bool): if True, only searches for diodes in the same cycle as the run_to_find.
            time_cutoff_days (float): the maximum time difference between the run and the diode scan to be considered "relevant".
        Returns:
            pd.DataFrame: a dataframe of relevant diode scans, ordered by distance in time from your run.  
            Will warn if the closest diode is more than 1 day away.
            To get *a* singular "best diode", just take the first row of the returned dataframe, i.e,:
                best_diode = findAppropriateDiodes(run_to_find,cat_diodes,diode_name,same_cycle,time_cutoff_days).iloc[0]



        '''
        import pandas as pd
        import warnings
        pd.options.mode.copy_on_write = True 
        time_cutoff = pd.Timedelta(3,'day')

        if not isinstance(run_to_find,pd.DataFrame):
            run_to_find = self.searchCatalog(scan_id=run_to_find)

        if cat_diodes is None:
            kwargs = {'sample':diode_name}
            if same_cycle:
                kwargs['cycle'] = run_to_find['cycle'].iloc[0]
            cat_diodes = self.searchCatalog(**kwargs)
        
        def _plan_to_edge_name(cat_diodes):
            cat_diodes['edge_name'] = cat_diodes['plan'].str.replace('nexafs','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('rsoxs','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('full','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('short','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('very','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('scan','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('nd','')
            cat_diodes['edge_name'] = cat_diodes['edge_name'].str.replace('_','')
            
            # expand short edge abbreviations
            try:
                from rsoxs_scans.defaults import edge_names
            except ImportError:
                edge_names = {
                            "c": "carbon",
                            "carbon": "carbon",
                            "carbonk": "carbon",
                            "ck": "carbon",
                            "n": "nitrogen",
                            "nitrogen": "nitrogen",
                            "nitrogenk": "nitrogen",
                            "nk": "nitrogen",
                            "f": "fluorine",
                            "fluorine": "fluorine",
                            "fluorinek": "fluorine",
                            "fk": "fluorine",
                            "o": "oxygen",
                            "oxygen": "oxygen",
                            "oxygenk": "oxygen",
                            "ok": "oxygen",
                            "ca": "calcium",
                            "calcium": "calcium",
                            "calciumk": "calcium",
                            "cak": "calcium",
                            'al': 'aluminium',
                            'aluminum': 'aluminium',
                        }   
            for k,v in edge_names.items():
                cat_diodes['edge_name'] = cat_diodes['edge_name'].replace(k,v) 
            return cat_diodes
        
        run_to_find = _plan_to_edge_name(run_to_find)
        cat_diodes['time_proximity'] = cat_diodes['start_time'] - run_to_find['start_time'].iloc[0]
        cat_diodes['abs_time_proximity'] = np.abs(cat_diodes['start_time'] - run_to_find['start_time'].iloc[0])
        cat_diodes['same_scan'] = cat_diodes['plan'] == run_to_find['plan'].iloc[0]
        cat_diodes['same_detector'] = cat_diodes['detector'] == run_to_find['detector'].iloc[0]
        cat_diodes['is_older'] = cat_diodes['time_proximity'] < pd.Timedelta(0)
        cat_diodes = _plan_to_edge_name(cat_diodes)
        cat_diodes['same_edge'] = cat_diodes['edge_name'] == run_to_find['edge_name'].iloc[0]
        
        relevant_diodes = cat_diodes[cat_diodes.same_edge]
        relevant_diodes = relevant_diodes[relevant_diodes.same_detector]
        relevant_diodes = relevant_diodes[relevant_diodes.abs_time_proximity < time_cutoff]
        relevant_diodes = relevant_diodes.sort_values(by='abs_time_proximity')
        
        if (relevant_diodes['abs_time_proximity'].min()) > pd.Timedelta(1,unit='day'):
            warnings.warn(f"Stale diode!  The closest relevant diode scan to the requested scan is {relevant_diodes['abs_time_proximity'].min()} from the measurement.")
        return relevant_diodes
    

    def background(f):
        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

        return wrapped

    @background
    def do_list_append(run, scan_ids, sample_ids, plan_names, uids, npts, start_times):
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

            meta_dim (str): dimension to stack along.  must be a valid attribute/metadata value, such as polarization or sample_name

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
        ), f"Error: not all loaded data have the same axes.  This is not supported yet.\n {axes}"
        axes[0].insert(0, meta_dim)
        new_system = axes[0]
        # print(f'New system to be stacked as: {new_system}')
        # print(f'meta_dimension = {meta_dim}')
        # print(f'labels in this dim are {label_vals}')
        return (
            xr.concat(scans, dim=meta_dim)
            .assign_coords({meta_dim: label_vals})
            .stack(system=new_system)
        )

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
            run (DataBroker result, int of a scan id, list of scan ids, list of DataBroker runs): a single run from BlueSky
            dims (list): list of dimensions you'd like in the resulting xarray.  See list of allowed dimensions in documentation.  If not set or None, tries to auto-hint the dims from the RSoXS plan_name.
            CHANGE: List of dimensions you'd like. If not set, will set all possibilities as dimensions (x, y, theta, energy, polarization)
            coords (dict): user-supplied dimensions, see syntax examples in documentation.
            return_dataset (bool,default False): return both the data and the monitors as a xr.dataset.  If false (default), just returns the data.
        Returns:
            raw (xarray): raw xarray containing your scan in PyHyper-compliant format

        """
        if isinstance(run,int):
            run = self.c[run]
        elif isinstance(run,pd.DataFrame):
            run = list(run.scan_id)
        if isinstance(run,list):
            return self.loadSeries(
                run,
                "sample_name",
                loadrun_kwargs={
                    "dims": dims,
                    "coords": coords,
                    "return_dataset": return_dataset,
                },
            )

        md = self.loadMd(run)
       

        monitors = self.loadMonitors(run)

        if dims is None:
            if ("NEXAFS" or "nexafs") in md["start"]["plan_name"]:
                raise NotImplementedError(
                    f"Scan {md['start']['scan_id']} is a {md['start']['plan_name']} NEXAFS scan. "
                    " NEXAFS loading is not yet supported."
                )  # handled case change in "NEXAFS"
            elif (
                "full" in md["start"]["plan_name"]
                or "short" in md["start"]["plan_name"]
                or "custom_rsoxs_scan" in md["start"]["plan_name"]
            ) and dims is None:
                dims = ["energy"]
            elif "spiralsearch" in md["start"]["plan_name"] and dims is None:
                dims = ["sam_x", "sam_y"]
            elif "count" in md["start"]["plan_name"] and dims is None:
                dims = ["time"]
            else:
                axes_to_include = []
                rsd_cutoff = 0.005

                # begin with a list of the things that are primary streams
                axis_list = list(run["primary"]["data"].keys())

                # next, knock out anything that has 'image', 'fullframe' in it - these aren't axes
                axis_list = [x for x in axis_list if "image" not in x]
                axis_list = [x for x in axis_list if "fullframe" not in x]
                axis_list = [x for x in axis_list if "stats" not in x]
                axis_list = [x for x in axis_list if "saturated" not in x]
                axis_list = [x for x in axis_list if "under_exposed" not in x]

                # remove new detector terms #193
                axis_list = [x for x in axis_list if "acquire_time" not in x]
                axis_list = [x for x in axis_list if "tiff_time_stamp" not in x]
                

                # remove hinted Energy and EPU60 items #161
                axis_list = [x for x in axis_list if "EPU60" not in x]
                axis_list = [x for x in axis_list if "Energy" not in x]
                
                # knock out any known names of scalar counters
                axis_list = [x for x in axis_list if "Beamstop" not in x]
                axis_list = [x for x in axis_list if "Current" not in x]

                if self.suppress_time_dimension:
                    axis_list = [x for x in axis_list if x != "time"]

                # now, clean up duplicates.
                axis_list = [x for x in axis_list if "setpoint" not in x]
                # now, figure out what's actually moving.  we use a relative standard deviation to do this.
                # arbitrary cutoff of 0.5% motion = it moved intentionally.
                for axis in axis_list:
                    std = np.std(run["primary"]["data"][axis])
                    motion = np.abs(np.max(run["primary"]["data"][axis])-np.min(run["primary"]["data"][axis]))
                    if motion == 0:
                        rsd = 0
                    else:
                        rsd = std / motion
                    # print(f'Evaluating {axis} for inclusion as a dimension with rsd {rsd}...')
                    if rsd > rsd_cutoff:
                        axes_to_include.append(axis)
                        # print(f'    --> it was included')

                # next, construct the reverse lookup table - best mapping we can make of key to pyhyper word
                # we start with the lookup table used by loadMd()
                reverse_lut = {md_key_beamline: md_key_PHS 
                               for md_key_PHS, md_key_beamline_list in self.md_lookup.items() 
                               for md_key_beamline in md_key_beamline_list}
# this creates a reverse mapping where the keys are the possible names in 'beamline language' of a parameter, and the values are the name in 'PyHyper language'.  This exploits the fact that dicts are ordered to provide rank-sorted mapping of parameters.
                # here, we broaden the table to make a value that default sources from '_setpoint' actually match on either
                # the bare value or the readback value.
                reverse_lut_adds = {}
                for k in reverse_lut.keys():
                    if "setpoint" in k:
                        reverse_lut_adds[k.replace("_setpoint", "")] = reverse_lut[k]
                        reverse_lut_adds[k.replace("_setpoint", "_readback")] = reverse_lut[k]
                reverse_lut.update(reverse_lut_adds)

                pyhyper_axes_to_use = []
                for x in axes_to_include:
                    try:
                        pyhyper_axes_to_use.append(reverse_lut[x])
                    except KeyError:
                        pyhyper_axes_to_use.append(x)
                dims = pyhyper_axes_to_use

        data = run["primary"]["data"][md["detector"] + "_image"]
        if isinstance(data,tiled.client.array.ArrayClient):
            data = run["primary"]["data"].read()[md["detector"] + "_image"]
        elif isinstance(data,tiled.client.array.DaskArrayClient):
            data = run["primary"]["data"].read()[md["detector"] + "_image"]
        # Handle extra dimensions (non-pixel and non-intended dimensions from repeat exposures) by averaging them along the dim_0 axis
        if len(data.shape) > 3:
            data = data.mean("dim_0", keepdims=True)
            
        data = data.astype(int)  # convert from uint to handle dark subtraction

        if self.dark_subtract:
            dark = run["dark"]["data"][md["detector"] + "_image"]
            if (
                type(dark) == tiled.client.array.ArrayClient
                or type(dark) == tiled.client.array.DaskArrayClient
            ):
                dark = run["dark"]["data"].read()[md["detector"] + "_image"]
            darkframe = np.copy(data.time)
            for n, time in enumerate(dark.time):
                darkframe[(data.time - time) > 0] = int(n)
            data = data.assign_coords(dark_id=("time", darkframe))

            def subtract_dark(img, pedestal=100, darks=None):
                return img + pedestal - darks[int(img.dark_id.values)]

            data = data.groupby("time",squeeze=False).map(subtract_dark, darks=dark, pedestal=self.dark_pedestal)

        dims_to_join = []
        dim_names_to_join = []

        for dim in dims:
            try:
                test = len(md[dim])  # this will throw a typeerror if single value
                if type(md[dim]) == dask.array.core.Array:
                    dims_to_join.append(md[dim].compute())
                else:
                    dims_to_join.append(md[dim])
                dim_names_to_join.append(dim)
            except TypeError:
                dims_to_join.append([md[dim]] * run.start["num_points"])
                dim_names_to_join.append(dim)

        for key, val in coords.items():
            dims_to_join.append(val)
            dim_names_to_join.append(key)

        index = pd.MultiIndex.from_arrays(dims_to_join, names=dim_names_to_join)
        # handle the edge case of a partly-finished scan
        if len(index) != len(data["time"]):
            index = index[: len(data["time"])]
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, 'system')
        retxr = (
            data.sum("dim_0")
            .rename({"dim_1": "pix_y", "dim_2": "pix_x"})
            .rename({"time": "system"})
            .assign_coords(mindex_coords)
        )  # ,md['detector']+'_image':'intensity'})

        # this is needed for holoviews compatibility, hopefully does not break other features.
        retxr = retxr.assign_coords(
            {
                "pix_x": np.arange(0, len(retxr.pix_x)),
                "pix_y": np.arange(0, len(retxr.pix_y)),
            }
        )
        try:
            monitors = (
                monitors.rename({"time": "system"})
                .reset_index("system")
                .assign_coords(mindex_coords)
            )

            if "system_" in monitors.indexes.keys():
                monitors = monitors.drop("system_")

        except Exception as e:
            warnings.warn(
                (
                    "Monitor streams loaded successfully, but could not be correlated to images. "
                    " Check monitor stream for issues, probable metadata change."
                ),
                stacklevel=2,
            )
        retxr.attrs.update(md)

        exposure = retxr.attrs.get("exposure")
        
        if exposure is not None:
            retxr.attrs["exposure"] = (len(data.dim_0) * exposure)
        else:
            retxr.attrs["exposure"] = None  # or 0, or skip setting it  # patch for multi exposures
        
        # now do corrections:
        frozen_attrs = retxr.attrs
        if self.corr_mode == "i0":
            retxr = retxr / monitors["RSoXS Au Mesh Current"]
        elif self.corr_mode != "none":
            warnings.warn(
                "corrections other than none or i0 are not supported at the moment",
                stacklevel=2,
            )

        retxr.attrs.update(frozen_attrs)

        # deal with the edge case where the LAST energy of a run is repeated... this may need modification to make it correct (did the energies shift when this happened??)
        try:
            if retxr.system[-1] == retxr.system[-2]:
                retxr = retxr[:-1]
        except IndexError:
            pass

        if return_dataset:
            # retxr = (index,monitors,retxr)
            monitors.attrs.update(retxr.attrs)
            retxr = monitors.merge(retxr)

        if self.use_chunked_loading:
            # dask and multiindexes are like PEO and PPO.  They're kinda the same thing and they don't like each other.
            retxr = retxr.unstack("system")

        return retxr

    def peekAtMd(self, run):
        return self.loadMd(run)

    def loadMonitors(
        self,
        entry,
        integrate_onto_images: bool = True,
        useShutterThinning: bool = True,
        n_thinning_iters: int = 1,
        directLoadPulsedMonitors: bool = True
    ):
        """Load the monitor streams for entry.

        Creates a dataset containing all monitor streams (e.g., Mesh Current, Shutter Timing, etc.) as data variables mapped
        against time. Optionally, all streams can be indexed against the primary measurement time for the images using
        integrate_onto_images. Whether or not time integration attempts to account for shutter opening/closing is controlled
        by useShutterThinning. Warning: for exposure times < 0.5 seconds at SST (as of 9 Feb 2023), useShutterThinning=True
        may excessively cull data points.

        Parameters
        ----------
        entry : databroker.client.BlueskyRun
            Bluesky Document containing run information.
            ex: phs.load.SST1RSoXSDB.c[scanID] or databroker.client.CatalogOfBlueskyRuns[scanID]
        integrate_onto_images : bool, optional
            whether or not to map timepoints to the image measurement times (as held by the 'primary' stream), by default True
            Presently bins are averaged between measurements intervals.
        useShutterThinning : bool, optional
            Whether or not to attempt to thin (filter) the raw time streams to remove data collected during shutter opening/closing, by default False
            As of 9 Feb 2023 at NSLS2 SST1, using useShutterThinning= True for exposure times of < 0.5s is
            not recommended because the shutter data is unreliable and too many points will be removed
        n_thinning_iters : int, optional
            how many iterations of thinning to perform, by default 1
            (former default was 5 before gated monitor loading was added)
            If you receive errors in assigning image timepoints to counters, try fewer iterations
        directLoadPulsedMonitors : bool, optional
            Whether or not to load the pulsed monitors using direct reading, by default True
            This only applies if integrate_onto_images is True; otherwise you'll get very raw data.
            If False, the pulsed monitors will be loaded using a shutter-thinning and masking approach as with continuous counters

        Returns
        -------
        xr.Dataset
            xarray dataset containing all monitor streams as data variables mapped against the dimension "time"
        """

        raw_monitors = None

        # Iterate through the list of streams held by the Bluesky document 'entry', and build 
        for stream_name in list(entry.keys()):
            # Add monitor streams to the output xr.Dataset
            if "monitor" in stream_name:
                if raw_monitors is None:  # First one
                    # incantation to extract the dataset from the bluesky stream
                    raw_monitors = entry[stream_name].data.read()
                else:  # merge into the to existing output xarray
                    raw_monitors = xr.merge((raw_monitors, entry[stream_name].data.read()))

        # At this stage monitors has dimension time and all streams as data variables
        # the time dimension inherited all time values from all streams
        # the data variables (Mesh current, sample current etc.) are all sparse, with lots of nans

        # if there are no monitors, return an empty xarray Dataset
        if raw_monitors is None:
            return xr.Dataset()

        # For each nan value, replace with the closest value ahead of it in time
        # For remaining nans, replace with closest value behind it in time
        monitors = raw_monitors.ffill("time").bfill("time")

        # If we need to remap timepoints to match timepoints for data acquisition
        if integrate_onto_images:
            try:
                # Pull out ndarray of 'primary' timepoints (measurement timepoints)
                primary_time = entry.primary.data["time"].__array__()
                primary_time_bins = np.insert(primary_time, 0,0)

                # If we want to exclude values for when the shutter was opening or closing
                # This doesn't work for exposure times ~ < 0.5 s, because shutter stream isn't reliable
                if useShutterThinning:
                    # Create new data variable to hold shutter toggle values that are thinned
                    # Shutter Toggle stream is 1 when open (or opening) and 0 when closed (or closing)
                    monitors["RSoXS Shutter Toggle_thinned"] = monitors["RSoXS Shutter Toggle"]

                    # Perform thinning to remove edge cases where shutter may be partially open or closed
                    monitors["RSoXS Shutter Toggle_thinned"].values = scipy.ndimage.binary_erosion(
                        monitors["RSoXS Shutter Toggle"].values,
                        iterations=n_thinning_iters,
                        border_value=0,
                    )

                    # Filter monitors to only include timepoints where shutter was open (as determined by thinning)
                    # Drop any remaining missing values along the time axis
                    monitors = monitors.where(monitors["RSoXS Shutter Toggle_thinned"] > 0).dropna(
                        "time"
                    )

                #return monitors
                # Bin the indexes in 'time' based on the intervales between timepoints in 'primary_time' and evaluate their mean
                # Then rename the 'time_bin' dimension that results to 'time'
                monitors = (
                    monitors.groupby_bins("time",primary_time_bins,include_lowest=True)
                    .mean()
                    .rename({"time_bins": "time"})
                )
                '''
                # Add primary measurement time as a coordinate in monitors that is named 'time'
                # Remove the coordinate 'time_bins' from the array
                monitors = (
                    monitors.assign_coords({"time": primary_time})
                    .drop_indexes("time_bins")
                    .reset_coords("time_bins", drop=True)
                )'''

                # load direct/pulsed monitors

                for stream_name in list(entry.keys()):
                    if "monitor" in stream_name and ("Beamstop" in stream_name or "Sample" in stream_name):
                        # the pulsed monitors we know about are "SAXS Beamstop", "WAXS Beamstop", "Sample Current"
                        # if others show up here, they could be added
                        out_name = stream_name.replace("_monitor", "")
                        mon = entry[stream_name].data.read()[out_name].compute()
                        SIGNAL_THRESHOLD = 0.1
                        threshold = SIGNAL_THRESHOLD*mon.mean('time')
                        mon_filter = xr.zeros_like(mon)
                        mon_filter[mon<threshold] = 0
                        mon_filter[mon>threshold] = 1
                        mon_filter.values = scipy.ndimage.binary_erosion(mon_filter)
                        mon_filtered = mon.where(mon_filter==1)
                        mon_binned = (mon_filtered.groupby_bins("time",primary_time_bins,include_lowest=True)
                                        .mean()
                                        .rename({"time_bins":"time"})
                                        )

                        if not directLoadPulsedMonitors:
                            out_name = 'pl_' + out_name

                        monitors[out_name] = mon_binned
                monitors = monitors.assign_coords({"time": primary_time})

              
            except Exception as e:
                # raise e # for testing
                warnings.warn(
                    (
                        "Error while time-integrating monitors onto images.  Usually, this"
                        " indicates a problem with the monitors, this is a critical error if doing"
                        " normalization otherwise fine to ignore."
                    ),
                    stacklevel=2,
                )
        return monitors

    def loadMd(self, run):
        """
        return a dict of metadata entries from the databroker run xarray


        """
        md = {}

        # items coming from the start document
        start = run.start

        meas_time = datetime.datetime.fromtimestamp(run.start["time"])
        md["meas_time"] = meas_time
        md["sample_name"] = start["sample_name"]
        if start["RSoXS_Config"] == "SAXS":
            md["rsoxs_config"] = "saxs"
            # discrepency between what is in .json and actual
            if (meas_time > datetime.datetime(2020, 12, 1)) and (
                meas_time < datetime.datetime(2021, 1, 15)
            ):
                md["beamcenter_x"] = 489.86
                md["beamcenter_y"] = 490.75
                md["sdd"] = 521.8
            elif (meas_time > datetime.datetime(2020, 11, 16)) and (
                meas_time < datetime.datetime(2020, 12, 1)
            ):
                md["beamcenter_x"] = 371.52
                md["beamcenter_y"] = 491.17
                md["sdd"] = 512.12
            elif (meas_time > datetime.datetime(2022, 5, 1)) and (
                meas_time < datetime.datetime(2022, 7, 7)
            ):
                # these params determined by Camille from Igor
                md["beamcenter_x"] = 498  # not the best estimate; I didn't have great data
                md["beamcenter_y"] = 498
                md["sdd"] = 512.12  # GUESS; SOMEONE SHOULD CONFIRM WITH A BCP MAYBE??
            else:
                md["beamcenter_x"] = run.start["RSoXS_SAXS_BCX"]
                md["beamcenter_y"] = run.start["RSoXS_SAXS_BCY"]
                md["sdd"] = run.start["RSoXS_SAXS_SDD"]

        elif "WAXS" in start["RSoXS_Config"]:
            md["rsoxs_config"] = "waxs"
            if (meas_time > datetime.datetime(2020, 11, 16)) and (
                meas_time < datetime.datetime(2021, 1, 15)
            ):
                md["beamcenter_x"] = 400.46
                md["beamcenter_y"] = 530.99
                md["sdd"] = 38.745
            elif (meas_time > datetime.datetime(2022, 5, 1)) and (
                meas_time < datetime.datetime(2022, 7, 7)
            ):
                # these params determined by Camille from Igor
                md["beamcenter_x"] = 397.91
                md["beamcenter_y"] = 549.76
                md["sdd"] = 34.5  # GUESS; SOMEONE SHOULD CONFIRM WITH A BCP MAYBE??
            else:
                md["beamcenter_x"] = run.start["RSoXS_WAXS_BCX"]  # 399 #
                md["beamcenter_y"] = run.start["RSoXS_WAXS_BCY"]  # 526
                md["sdd"] = run.start["RSoXS_WAXS_SDD"]

        else:
            md["rsoxs_config"] = "unknown"
            warnings.warn(
                (
                    f'RSoXS_Config is neither SAXS or WAXS. Looks to be {start["RSoXS_Config"]}. '
                    " Might want to check that out."
                ),
                stacklevel=2,
            )


        if md["rsoxs_config"] == "saxs":
            md["detector"] = "Small Angle CCD Detector"
        elif md["rsoxs_config"] == "waxs":
            md["detector"] = "Wide Angle CCD Detector"
        else:
            warnings.warn(f"Cannot auto-hint detector type without RSoXS config.", stacklevel=2)

        # items coming from baseline
        baseline = run["baseline"]["data"]

        # items coming from primary
        try:
            primary = run["primary"]["data"]
        except (KeyError, HTTPStatusError):
            raise Exception(
                "No primary stream --> probably you caught run before image was written.  Try again."
            )

        md_lookup = copy.deepcopy(self.md_lookup)
        # Add additional metadata not included in lookup dictionary
        md_key_names_beamline = []
        
        for key in md_lookup.keys(): # Make a single list with all historical keys, in the order to check for them.
            md_key_names_beamline = md_key_names_beamline + md_lookup[key] 
        
        for key in primary.keys():
            if key not in md_key_names_beamline:
                if "_image" not in key:
                    md_lookup[key] = [key]
                    
        # Find metadata from Tiled primary and store in PyHyperScattering metadata dictionary
        for key_name_PHS, key_names_beamline in md_lookup.items(): # first iterate over PyHyper 'words' and ordered lists to check
            for key_name_beamline in key_names_beamline: # next, go through that list in order
                if (key_name_PHS in md
                    and md[key_name_PHS] is not None): 
                    break # If the pyhyper key is already filled in, no need to try other beamline keys, so stop processing this PyHyper key.
                try: 
                    md[key_name_PHS] = primary[key_name_beamline].read() # first try finding metadata in primary stream
                except (KeyError, HTTPStatusError):
                    try:
                        baseline_value = baseline[key_name_beamline] # Next, try finding metadata in baseline
                        
                        if isinstance(baseline_value, (tiled.client.array.ArrayClient, tiled.client.array.DaskArrayClient)): 
                            baseline_value = baseline_value.read() # For tiled_client.array data types, need to use .read() to get the values
                        
                        md[key_name_PHS] = baseline_value.mean().round(4) # Rounded for stacking purposes, to avoid slightly different values when not meaningful
                        
                        if baseline_value.var() > 1e-4*abs(baseline_value.mean()):                             
                            warnings.warn(f"{key_name_PHS} changed during scan: {baseline_value}.",stacklevel=2)
                    except (KeyError, HTTPStatusError): 
                        md[key_name_PHS] = None
            if md[key_name_PHS] is None:
                warnings.warn(f"Could not find any of {key_names_beamline} in either baseline or primary. Setting {key_name_PHS} to None.",stacklevel=2)
                        
        md["epoch"] = md["meas_time"].timestamp() # Epoch = the time the entire run started, used for multi-scan stacking

        # looking at exposure tests in the stream and issuing warnings
        if "Wide Angle CCD Detector_under_exposed" in md:
            if np.any(md["Wide Angle CCD Detector_under_exposed"]):
                message = "\nWide Angle CCD Detector is reported as underexposed\n"
                message += "at one or more energies per definitions here:\n"
                message += "https://github.com/NSLS-II-SST/rsoxs/blob/10c2c41b695c1db552f62decdde571472b71d981/rsoxs/Base/detectors.py#L110-L119\n"
                if np.all(md["Wide Angle CCD Detector_under_exposed"]):
                    message += "Wide Angle CCD Detector is reported as underexposed at all energies."
                else:
                    idx = np.where(md["Wide Angle CCD Detector_under_exposed"])
                    try:
                        warning_e = md["energy"][idx]
                        message += f"Affected energies include: \n{warning_e}"
                    except Exception:
                        message += f"Affected frames were {idx}."
                warnings.warn(message, stacklevel=2)
        else:
            warnings.warn(
                "'Wide Angle CCD Detector_under_exposed' not found in stream."
            )
        if "Wide Angle CCD Detector_saturated" in md:
            if np.any(md["Wide Angle CCD Detector_saturated"]):
                message = "\nWide Angle CCD Detector is reported as saturated\n"
                message += "at one or more energies per definitions here:\n"
                message += "https://github.com/NSLS-II-SST/rsoxs/blob/10c2c41b695c1db552f62decdde571472b71d981/rsoxs/Base/detectors.py#L110-L119\n"
                if np.all(md["Wide Angle CCD Detector_saturated"]):
                    message += "\tWide Angle CCD Detector is reported as saturated at all energies."
                else:
                    idx = np.where(md["Wide Angle CCD Detector_saturated"])
                    try:
                        warning_e = md["energy"][idx]
                        message += f"Affected energies include: \n{warning_e}"
                    except Exception:
                        message += f"Affected frames were {idx}."
                warnings.warn(message, stacklevel=2)
        else:
            warnings.warn(
                "'Wide Angle CCD Detector_saturated' not found in stream."
            )

        try:
            md["wavelength"] = 1.239842e-6 / md["energy"]
        except TypeError:
            md["wavelength"] = None
            
        md["sampleid"] = start["scan_id"]

        md["dist"] = md["sdd"] / 1000

        md["pixel1"] = self.pix_size_1 / 1000
        md["pixel2"] = self.pix_size_2 / 1000

        if not self.use_precise_positions:
            md["sam_x"] = md["sam_x"].round(1)
            md["sam_y"] = md["sam_y"].round(1)

        md["poni1"] = md["beamcenter_y"] * md["pixel1"]
        md["poni2"] = md["beamcenter_x"] * md["pixel2"]

        md["rot1"] = 0
        md["rot2"] = 0
        md["rot3"] = 0

        md.update(run.metadata)
        return md

    def loadSingleImage(self, filepath, coords=None, return_q=False, **kwargs):
        """
        DO NOT USE

        This function is preserved as reference for the qx/qy loading conventions.

        NOT FOR ACTIVE USE.  DOES NOT WORK.
        """
        if len(kwargs.keys()) > 0:
            warnings.warn(
                f"Loader does not support features for args: {kwargs.keys()}",
                stacklevel=2,
            )
        img = Image.open(filepath)

        headerdict = self.loadMd(filepath)
        # two steps in this pre-processing stage:
        #     (1) get and apply the right scalar correction term to the image
        #     (2) find and subtract the right dark
        if coords != None:
            headerdict.update(coords)

        # step 1: correction term

        if self.corr_mode == "expt":
            corr = headerdict["exposure"]  # (headerdict['AI 3 Izero']*expt)
        elif self.corr_mode == "i0":
            corr = headerdict["AI 3 Izero"]
        elif self.corr_mode == "expt+i0":
            corr = headerdict["exposure"] * headerdict["AI 3 Izero"]
        elif self.corr_mode == "user_func":
            corr = self.user_corr_func(headerdict)
        elif self.corr_mode == "old":
            corr = (
                headerdict["AI 6 BeamStop"]
                * 2.4e10
                / headerdict["Beamline Energy"]
                / headerdict["AI 3 Izero"]
            )
            # this term is a mess...  @TODO check where it comes from
        else:
            corr = 1

        if corr < 0:
            warnings.warn(
                f"Correction value is negative: {corr} with headers {headerdict}.",
                stacklevel=2,
            )
            corr = abs(corr)

        # # step 2: dark subtraction
        # try:
        #     darkimg = self.darks[headerdict['EXPOSURE']]
        # except KeyError:
        #     warnings.warn(f"Could not find a dark image with exposure time {headerdict['EXPOSURE']}.  Using zeros.",stacklevel=2)
        #     darkimg = np.zeros_like(img)

        # img = (img-darkimg+self.dark_pedestal)/corr
        if return_q:
            qpx = (
                2 * np.pi * 60e-6 / (headerdict["sdd"] / 1000) / (headerdict["wavelength"] * 1e10)
            )
            qx = (np.arange(1, img.size[0] + 1) - headerdict["beamcenter_x"]) * qpx
            qy = (np.arange(1, img.size[1] + 1) - headerdict["beamcenter_y"]) * qpx
            # now, match up the dims and coords
            return xr.DataArray(
                img, dims=["qy", "qx"], coords={"qy": qy, "qx": qx}, attrs=headerdict
            )
        else:
            return xr.DataArray(img, dims=["pix_x", "pix_y"], attrs=headerdict)
