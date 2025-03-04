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
  
- Identify the current conda environment, which appears in parenthases at the beginning of the command prompt.  If a conda environment that is not base and is not the desired environment is active, deactivate the conda environment.  Do not deactivate the base environment.
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
  
