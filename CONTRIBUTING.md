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


To run testing and linting locally
----------------------------------

You need to have access to the example data, you also need pytest and flake8 installed (pip install pytest flake8)

Run the following commands:
first cd to the root of your PyHyper git repo, then:
```
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/cyrsoxs-example.zip
unzip cyrsoxs-example.zip
rm cyrsoxs-example.zip
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/Example.zip
unzip Example.zip
rm Example.zip
wget https://github.com/usnistgov/PyHyperScattering/releases/download/0.0.0-example-data/mask-test-pack.zip
unzip mask-test-pack.zip
rm mask-test-pack.zip
```
then, while you are in this directory, to lint run
```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
errors in the first one *will* be a problem and will stop CI from running.
The second is just correcting your grammar.  No need to do anything about its output.

and to test run
```
pytest
```

