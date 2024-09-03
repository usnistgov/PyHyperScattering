.. _search_catalog:

Searching the catalog: finding your scans
==============================================================
As mentioned, finding your scans is one of the first steps in processing your scans stored in the Bluesky catalog. The general steps for using the searchCatalog function are:
    1. Define the tiled Bluesky catalog (potentially optional step as it can be set by default usingthe SST1RSoXSDB)
    2. Define the databroker loader object (i.e. SST1RSoXSDB.)
    3. Call the searchCatalog method of SST1RSoXSDB, with chosen selection criteria

The most common selection criteria (pulled directly from the searchCatalog docstring):
    cycle (str, optional): NSLS2 beamtime cycle, regex search e.g., "2022" matches "2022-2", "2022-1"
    proposal (str, optional): NSLS2 PASS proposal ID, case-insensitive, exact match, e.g., "GU-310176"
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

In the NSLS-II JupyterHub, a catalog generated using the tiled "from_profile" function. "from_profile" only works on BNL network machines, so "from_uri" or other methods must be used to build a catalog on machines outside of the BNL network. 

Below are some options for further slicing the returned pandas DataFrame (calling this DataFrame "df"), as this is often useful for omitting bad scans or selecting specific ranges/filters. Refer to pandas documentation for more, rich information on all the different ways you can slice and manipulate DataFrames.
    - Select specific values of a column heading; e.g. select the 'rsoxs_carbon' plan:
        runs_of_interest = df[(df['plan']=='rsoxs_carbon')] 
    - Select 'rsoxs_carbon' plan and only scans with 228 images:
        runs_of_interest = df[(df['plan']=='rsoxs_carbon') & (df['num_Images']==228)]
    - Select 'rsoxs_carbon' plan and omit some bad scans, select scans below or above given scan ID numbers:
        runs_of_interest = df[(df['plan']=='rsoxs_carbon') & ((df['scan_id']<80033) | (df['scan_id']>80046))]
    - Omit a specific index from the pandas DataFrame:
        runs_of_interest = runs_of_interest.drop(index=31)




