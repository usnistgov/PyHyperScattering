.. _analysis:

Learning to Fly 
=================

From this point, you're in a place where the next steps largely depend on the analysis you want to do with your data.  


The tools and language are that of the xarray package, so the main goal of this section is to teach you just enough xarray to have a vocabulary for converting scientific lines of thought into commands and visualize/dissect the results.



The most important command, by far, is select or ``.sel``.  It lets you grab
specific coordinates from your dataset.  Once selected, you can perform
reductions such as averaging with ``.mean`` and quickly visualize results with
``.plot``.

Example workflow::

    ds = load_some_data()

    # choose the energy slice nearest 284 eV
    e_slice = ds.sel(energy=284, method="nearest")

    # average over repeated frames
    q_avg = e_slice.mean(dim="repeat")

    # plot intensity versus q
    q_avg.plot(x="q")

You can do the same with ``q`` slices or any other coordinate using the same
pattern of ``.sel`` followed by a reduction and plot.  See the
`xarray documentation <https://docs.xarray.dev/en/stable/>`_ for more
information and examples.


