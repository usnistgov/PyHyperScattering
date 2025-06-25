.. _integration:

Integration: raw intensity to I(q)
====================================

Raw detector images generally contain two pixel axes.  To visualize
and analyze scattering patterns these need to be transformed into
chi/q or qz/qxy coordinates.  PyHyperScattering wraps several
integrators to accomplish this, with most implementations relying on
`pyFAI <https://pyfai.readthedocs.io/>`_.

``PFGeneralIntegrator`` provides a thin layer over pyFAI's azimuthal
integrator.  Pass in calibration parameters or a ``.poni`` file and a
mask, then call ``integrateImageStack`` to convert an xarray with
``pix_x``/``pix_y`` axes to intensity in ``q`` and optionally ``chi``::

   from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator
   integrator = PFGeneralIntegrator(ponifile="calib.poni",
                                   maskmethod="image", maskpath="mask.tif")
   iq = integrator.integrateImageStack(raw_data)

For grazing-incidence experiments ``PGGeneralIntegrator`` uses
``pygix`` to output either reciprocal-space ``q_xy``/``q_z`` data or
caked ``q`` vs ``chi``.  Datasets already in ``qx``/``qy`` coordinates
can be integrated with ``WPIntegrator`` which relies on
``skimage.transform.warp_polar`` (or its GPU version).

All integrators return properly labeled xarray objects so the rest of
the workflow can use normal xarray indexing and visualization.
