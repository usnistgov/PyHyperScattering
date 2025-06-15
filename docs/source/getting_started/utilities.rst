.. _utilities:

Utility Modules
=================

Despite the power of writing your own analyses, it seems likely that many of the analyses you'll want to do won't actually be that novel.

To that end, we furnish two modules of 'canned' approaches: RSoXS and fitting. User contributions are encouraged in this area.

RSoXS
----------

Tools in :mod:`PyHyperScattering.util.RSoXS` help with common resonant soft x-ray scattering analyses.  The ``rsoxs`` accessor on ``DataArray`` objects provides helpers for chi slicing and for computing anisotropy.  The :py:meth:`~PyHyperScattering.util.RSoXS.RSoXS.AR` method returns the anisotropic ratio from one or two polarized measurements.

Example::

    ar = data.rsoxs.AR()

or, with both polarizations available::

    ar = pair.rsoxs.AR(calc2d=True)

See the :mod:`PyHyperScattering.util.RSoXS` API reference for details.

Fitting
-------------

Fitting helpers live in :mod:`PyHyperScattering.util.Fitting`.  The ``fit`` accessor stacks all dimensions except the chosen fit axis and applies a supplied function.  Several ready-made fit functions are provided, including :py:func:`~PyHyperScattering.util.Fitting.fit_lorentz`, :py:func:`~PyHyperScattering.util.Fitting.fit_lorentz_bg` and :py:func:`~PyHyperScattering.util.Fitting.fit_cos_anisotropy`.

Example::

    results = data.fit.apply(
        PyHyperScattering.util.Fitting.fit_lorentz_bg,
        guess=[0, 0, 0.0002, 0],
        pos_int_override=True,
    )

See the :mod:`PyHyperScattering.util.Fitting` API reference for a full list of utilities.
