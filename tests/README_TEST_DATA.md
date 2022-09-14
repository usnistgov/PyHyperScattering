Most PyHyper tests are going to want to work with real data.  We have a hacky way to do this.

Prerequisites: keep in mind that your test data is going to be downloaded and computed on _a lot_, at least 12 times for _every commit that is made to PyHyper_.  
Put a little effort into making it a minimal test example so that we don't overburden the continuous integration infrastructure.


1.  Upload a zip file containing a directory containing your data as a binary asset to the special release 0.0.0_example_data.
2.  Modify .github/actions/main.yml to add a wget and unzip of your data pack in the appropriate section - note windows wget and mac/linux wget are different, look closely at the file.
      You may not actually have permissions to write to this file.  Contact a repo admin if not.
3. Any test you write *should* now have your directory in the root directory, see example tests in this directory for loaders.

