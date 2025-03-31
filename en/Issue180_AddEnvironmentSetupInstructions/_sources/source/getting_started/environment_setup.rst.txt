.. _Set_up_Python:

Set up Python
=============

Please follow the instructions below to set up an appropriate environment to use PyHyperScattering. All of these setup steps should be run in a terminal, not a Jupyter notebook.

These instructions have been tested for the use of PyHyperScattering in a local JupyterLab notebook using the Anaconda distribution (https://www.anaconda.com/download) on a Windows computer. The instructions below *might* work for other platforms (e.g., NSLS II Jupyterhub, Google Colab), but there are no guarantees; recently, the NSLS II Jupyterhub has been especially incompatible PyHyperScattering.

Download Git
------------

To aid this workflow, download Git (https://git-scm.com/download/win).  Then in the command prompt (not Anaconda Prompt), run ``winget install --id Git.Git -e --source winget``.  After this, if you are able to run ``git --version`` and have a version number outputted, the installation was successful.  If Anaconda Prompt was open, it may need to be restarted.

Create and activate an environment
----------------------------------

- Open the Anaconda Prompt.  Do not use the terminal feature after opening JupyterLab.
  
- Identify the current conda environment, which appears in parentheses at the beginning of the command prompt. If a conda environment that is not base and is not the desired environment is active, deactivate the conda environment. Do not deactivate the base environment.

  .. code-block:: bash

     conda deactivate

- Create a new environment if a suitable one does not exist.  Replace ``YOUR_ENVIRONMENT_NAME`` with a an environment name of choice that does not contain any spaces.  If needed, the ``...`` can be replaced with other conda packages to be installed in this environment.  The current notebook was run with the ``...`` omitted.  After loading some packages, you will be asked if you want to proceed.  Enter y (yes).
  
  .. code-block:: bash
  
     conda create -n YOUR_ENVIRONMENT_NAME ipykernel ...

  If there already is an appropriate environment, skip this step.  The purpose of the conda environment is to contain the necessary package versions that will enable data reduction and not conflict with other packages.

- Activate the desired environment.  After running this command, the selected environment name should appear in parentheses in the command prompt.
  
  .. code-block:: bash
    
       conda activate YOUR_ENVIRONMENT_NAME
  
  If you do not remember your environment name, you can run ``conda env list`` to display a list of environments that currently exist.  If there is an environment you want to delete, first ensure it is not active, and then run ``conda remove -n YOUR_ENVIRONMENT_NAME --all``.  The flat ``--all`` removes the entire environment.

- Run the following to add the environment to your Jupyter notebook selection.  The display name and environment name do not have to be the same.
  
  .. code-block:: bash
  
     python -m ipykernel install --user --name YOUR_ENVIRONMENT_NAME --display-name YOUR_ENVIRONMENT_NAME

Install packages and dependencies
---------------------------------

Install Python
~~~~~~~~~~~~~~

Check the Python version.  Use version 3.11 or lower for PyHyperScattering to work.

.. code-block:: bash
  
   python --version
  
If needed, install the correct Python version.

.. code-block:: bash
  
   conda install python==3.11
  
If a CondaSSL error is encountered during this step, the following solution can be run, and then Python installation can be retried: https://github.com/conda/conda/issues/8273

Install PyHyperScattering
~~~~~~~~~~~~~~~~~~~~~~~~~

``pip install pyhyperscattering[bluesky, ui]`` installs PyHyperScattering.  The ``bluesky`` portion installs Bluesky-related dependencies needed to access the NSLS II Tiled database. The ``ui`` portion installs the necessary dependencies to draw a mask.  In some cases, it may be necessary to clone and check out a later PyHyperScattering commit or branch instead of the default version. Below are some examples.

- ``pip install "git+https://github.com/usnistgov/PyHyperScattering.git#egg=PyHyperScattering[bluesky]"`` installs the latest commit on the main branch.

- ``pip install "git+https://github.com/usnistgov/PyHyperScattering.git@Issue170_UpdateDatabrokerImports#egg=PyHyperScattering[bluesky]"`` installs the latest commit on the branch named ``Issue170_UpdateDatabrokerImports``.

- ``pip install "git+https://github.com/usnistgov/PyHyperScattering.git@6657973#egg=PyHyperScattering[bluesky]"`` installs commit ``6657973``.


Install JupyterLab
~~~~~~~~~~~~~~~~~~

``pip install jupyterlab`` is required if using the Anaconda distribution but might not be necessary in other cases (e.g., NSLS II JupyterHub, Google Colab)

Other installs
~~~~~~~~~~~~~~

If there are errors during installation or later on, it might be necessary to install additional packages and then retry the pip installs.  Below is a list of what might be needed.

- Microsoft C++ Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/).  This is installed outside the Anaconda prompt.  Computer should be restarted after this installation.

- ``pip install --upgrade holoviews``  This may be necessary if mask drawing is not working.  The ``--upgrade`` is necessary to ensure that the package will get upgraded even if some version of it is currently installed.

- ``pip install natsort`` allows use of the natsort package, but is not necessary for the main functioning of PyHyperScattering.


Open JupyterLab
---------------

- Start up JupyterLab from the Anaconda command prompt.  Do not open JupyterLab using Anaconda's GUI menu.
  
  .. code-block:: bash
  
     jupyter-lab

- When prompted to select a kernel, choose the desired environment.  If not prompted, ensure that the kernel on the top right-hand corner of the page is set to the correct environment name.

- Proceed to using a Jupyter notebook of choice to reduce and analyze data.


Additional resources
--------------------
- Full list of PyHyperScattering dependencies: https://github.com/usnistgov/PyHyperScattering/blob/main/pyproject.toml
- Further guidance on creating and managing environments: https://jupyter.nsls2.bnl.gov/hub/guide
- Conda documentation: https://docs.conda.io/projects/conda/en/stable/
- Xarray documentation: https://docs.xarray.dev/en/stable/
- Numpy documentation: https://numpy.org/doc/2.1/
- MatPlotLib documentation: https://matplotlib.org/stable/index.html

