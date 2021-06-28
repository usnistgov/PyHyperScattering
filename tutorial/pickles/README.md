This directory contains pickled PyHyper products, for use in testing and tutorials.


~~~~IMPORTANT~~~~

Pickle files are a known security vulnerability point and can cause arbitrary code execution.
It is a high priority issue to provide a secure serialization method for PyHyper datasets.
Nevertheless, we have a need to distribute these before that serialization method is ready.

The system that prepared these files is not believed to be compromised and is professionally managed by cybersecurity experts.
We can make no guarantees of the security of these files, however.

YOU MUST VERIFY THE CHECKSUMS OF THE FILES BEFORE USING THEM, ESPECIALLY ON SENSITIVE SYSTEMS.

MD5 (SST1_Dec20_int_stack.p) = 9ccfbb02fd99be5d889d709b36a10bff
MD5 (demo_int_saxs.p) = 48584b9d8736eb608f4d53cc07e9db57

These checksums will be posted in the release notes label.  
If they differ between the label and this file, or between this file and the output of 
    > md5 (file).p
it is possible someone is doing something nasty.  Proceed with caution.
