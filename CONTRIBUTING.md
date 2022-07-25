Contribution to PyHyper is encouraged!

It is recommended that you familiarize yourself with the operation of the code before contributing, to get a 'lay of the land' in terms of what goes where, etc.  A good rule of thumb: the features in the tutorials should work as written without 'hacking' the code.  If they don't, the problem is likely with the documentation, not the code.  Reach out to a project maintainer for help or open an issue so we know where to target doc improvements (or, after opening an issue, add what you learn TO the docs and contribute them back!)

Once you do that, you can follow these instructions to set up a development environment.

**Note: on a local machine, you can probably bypass 1, 3, 4, 5 by using Github Desktop.  Note the policies about commits being in a branch, good commit messages.

1) Clone the raw git code.

	In a suitable place on your machine, run

	    git clone https://github.com/usnistgov/PyHyperScattering.git
	    cd PyHyperScattering

	**If you are not NIST staff, you should fork the repository first and clone your fork, because you will not be able to push back to usnistgov.  Contributions from outside NIST are encouraged, but we can only accept changes back via pull request.
	**If you are at NIST, create a new branch
		git checkout -b [few word descriptor of the feature you plan to add]

		for example

		git checkout -b qxy_support


2) Install the package in pip
	From a terminal, change directory to where you just cloned PyHyper and run:
		
		pip install -e .

	This should install the development copy in place of any distribution PyHyper you have, where applicable.

	To revert, simply
		
		pip install PyHyperScattering==(production release version)

2a) Set up your development environment to import from your local, development copy (and not any pip installed production release).

	The maintainers mostly develop in Jupyter (Sorry!!).

	In general, you want to add the path to your development installation earlier in the path tree than the pip-installed version.

	In my notebooks, I add this cell to the top:

		%load_ext autoreload
		%autoreload 2
		from PyHyperScattering import __version__
		print(f'USING PYHYPER VERSION: {__version__}')

	__version__ will prints as either a release number (if sourced from production packages), or a longer string if sourced from a working tree, something like: 0.6+2.g6d02fb3.dirty

	 If it prints a git hash, you're working from your development copy. 
	 (the format is [last tag]+[number of untagged commits].[commit hash].[dirty if there are unstaged changes])

3) Do your development

	If you imported autoreload above, any changes made to code should live-update into your notebook (cool!)

4) Commit changes

	In your terminal, just run
		git add [changed file names]
		git commit

	You can view status with 
		git status

	You'll be prompted to give a commit message.  Please try to make them useful - the default 'updated x.py' are virtually useless.
	Please also try to remove any instrumentation print statements from the code. 

5) Push changes back to repo

	You're encouraged to push back whenever features are partly complete or ready for testing.

	Just run

		git push -u origin [branch_name]

6) Circle thru 3-5 till the project is done.

7) Open a pull request using the GitHub web interface to merge your changes into main.

8) Discussion ensues.

9) A maintainer will merge changes into main

10) Your changes will be included in the pip package on the next release!
