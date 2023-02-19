Contribution to PyHyperScattering is encouraged!
======

It's recommended that you familiarize yourself with the operation of the code before contributing, to get a 'lay of the land' in terms of what goes where, etc.  A good rule of thumb: the features in the tutorials should work as written without 'hacking' the code.  If they don't, the problem is likely with the documentation, not the code.  Reach out to a project maintainer for help or open an issue so we know where to target doc improvements (or, after opening an issue, add what you learn TO the docs and contribute them back!)

It's also generally a good idea to start small; most issues tagged 'documentation' and 'good first issue' are nice ideas of where to start. Reaching out to the existing development experts for help getting started or advice in advance of putting in major development effort is strongly recommended.  Peter Beaucage (peter.beaucage@nist.gov) is the original developer and lead maintainer.  He tends to hang around the NIST RSoXS, NSLS II, and nikea slack workspaces if you're on any of them and is physically based at the NIST Center for Neutron Research.  Peter Dudenas (NIST at LBNL) and Eliot Gann (NIST at BNL) are also good local resources.

Specifically, if you intend to work on any core modules except (a) single instrument support that inherits from FileLoader, (b) utility modules like FileIO, RSoXS, or other additions, or (c) documentation and tutorial changes, it is strongly recommended that you have a good foundational knowledge of xarray, consult with project maintainers for a developer-level intro to the codebase, and ask questions frequently.  Many 'guts' of PyHyper are very complicated and rely on legacy xarray concepts.

That caveat emptor now delivered, you can follow these instructions to set up a development environment.

Short Form Project Policies (for everyone, but the essentials upfront for people who know git flow already)
=========

1.  Contributions should be motivated by issues; create an issue first discussing your problem and intended fix.  They don't need to be 'War and Peace' but enough detail to understand is useful.  This is a place where project contributors can try to provide helpful advice.
2.  We have branch protections on, so you need to bring new material in via a pull request (PR) from either a branch in the main repo (NIST staff) or a personal fork (all others).  If you make a branch in the main repo, please name it using the GitHub auto-name-from-issue feature (in the sidebar of an open issue).
3.  Pushing early and often to a branch is helpful for new contributors.  Draft PRs are also encouraged to discuss implementation details.
4.  PRs require one approving review by a core project maintainer before merging.  It's not rude to merge yourself once this review is given, if you have addressed all feedback.
5.  New features *require* tests and documentation.  We use Sphinx/readthedocs to auto generate API documentation from docstrings.  The pragmatic nature of the project means that we can and often do merge in features without tests/docs, but we never close an issue without full testing and documentation.
	(talk to a maintainer about building loader tests - we have a mechanism for hosting small example datasets to test with in GitHub Actions)
	
Detailed Instructions 
======

**Note: If you are developing on a local machine rather than a shared resource, you can probably bypass 1, 3, 4, 5 by using Github Desktop.  Note the policies above.

1) Clone the raw git code.

	In a suitable place on your machine, run

	    git clone https://github.com/usnistgov/PyHyperScattering.git
	    cd PyHyperScattering

	**If you are not NIST staff, you should fork the repository first and clone your fork, because you will not be able to push back to usnistgov.  Contributions from outside NIST are encouraged, but we can only accept changes back via pull request.
	
	**If you are at NIST, create a new branch
		
		`git checkout -b [few word descriptor of the feature you plan to add]`

		for example

		`git checkout -b qxy_support`
		
		If you have created a branch from an issue as requested above, this name will be provided by GitHub, simply copy the GitHub command block whhich will look something like:
		
		`git fetch origin`
		`git checkout NN_description_of_your_feature`
		
		As a reminder, to switch branches, just do
		
		`git checkout branch_name`
		

2) Install the package in pip
	From a terminal, change directory to where you just cloned PyHyper and run:
		
		`pip install -e .`

	This should install the development copy in place of any distribution PyHyper you have, where applicable.

	To revert, simply
		
		`pip install PyHyperScattering==(production release version)`

2a) Set up your development environment to import from your local, development copy (and not any pip installed production release).

	The maintainers mostly develop in Jupyter.  This is not necessarily advised, just a statement of bad practices.

	In general, you want to add the path to your development installation earlier in the path tree than the pip-installed version.

	In my notebooks, I add this cell to the top:

		```
		%load_ext autoreload
		%autoreload 2
		from PyHyperScattering import __version__
		print(f'USING PYHYPER VERSION: {__version__}')
		```
		
	__version__ will prints as either a release number (if sourced from production packages), or a longer string if sourced from a working tree, something like: 0.6+2.g6d02fb3.dirty

	 If it prints a git hash, you're working from your development copy. 
	 (the format is `[last tag]+[number of untagged commits].[commit hash].[dirty if there are unstaged changes]`)

3) Do your development

	If you imported autoreload above, any changes made to code should live-update into your notebook.

4) Commit changes

	In your terminal, just run
		```
		git add [changed file names]
		git commit
		```

	You can view status with 
		`git status`

	You'll be prompted to give a commit message.  Please try to make them useful - the default 'updated x.py' are virtually useless.
	Please also try to remove any instrumentation print statements from the code. We will take them out in code review but might make fun of you ;)

5) Push changes back to repo

	You're encouraged to push back whenever features are partly complete or ready for testing.

	Just run

		`git push`

6) Circle thru 3-5 till the project is done.

7) Open a pull request using the GitHub web interface to merge your changes into main.

8) Discussion ensues.

9) A maintainer will merge changes into main

10) Your changes will be included in the pip package on the next release!


To run testing and linting locally
----------------------------------

You need to have access to the example data, you also need pytest and flake8 installed (`pip install pytest flake8`)

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

