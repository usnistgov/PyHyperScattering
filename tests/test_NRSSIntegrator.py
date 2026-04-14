import numpy as np
import pytest
import xarray as xr

from PyHyperScattering.integrate import NRSSIntegrator, WPIntegrator


def _synthetic_qxy_image(
    *,
    nx=101,
    ny=91,
    qmax=1.0,
    phys_size_nm=None,
    attrs=None,
    dims=("qy", "qx"),
):
    if phys_size_nm is None:
        qx = np.linspace(-qmax, qmax, nx, dtype=np.float64)
        qy = np.linspace(-qmax, qmax, ny, dtype=np.float64)
    else:
        qx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=float(phys_size_nm)))
        qy = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=float(phys_size_nm)))
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    image = (
        np.exp(-((qx_grid - 0.18) ** 2 + (qy_grid + 0.07) ** 2) / 0.010)
        + 0.65 * np.exp(-((qx_grid + 0.24) ** 2 + (qy_grid - 0.22) ** 2) / 0.018)
        + 0.08
    )
    da = xr.DataArray(
        image,
        dims=("qy", "qx"),
        coords={"qy": qy, "qx": qx},
        attrs=dict(attrs or {}),
    )
    if tuple(dims) == ("qy", "qx"):
        return da
    if tuple(dims) == ("qx", "qy"):
        return da.transpose("qx", "qy")
    raise ValueError(f"Unsupported dims ordering {dims!r}.")


def test_nrss_integrator_2d_matches_wp_semantics():
    img = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 1, "phys_size_nm": 5.0},
        dims=("qx", "qy"),
    )
    nrss = NRSSIntegrator(force_np_backend=True)
    wp = WPIntegrator(force_np_backend=True)

    reduced_nrss = nrss.integrateImageStack(img)
    reduced_wp = wp.integrateImageStack(img)

    np.testing.assert_allclose(reduced_nrss.coords["q"].values, reduced_wp.coords["q"].values)
    np.testing.assert_allclose(reduced_nrss.values, reduced_wp.values)
    assert reduced_nrss.attrs["radial_semantics"] == "q_perp"
    assert reduced_nrss.attrs["nrss_semantic_mode"] == "2d_reciprocal_plane"
    assert reduced_nrss.attrs["source_integrator"] == "NRSSIntegrator"


def test_nrss_integrator_3d_corrects_radial_axis():
    energy_ev = 285.0
    img = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 128, "phys_size_nm": 5.0, "energy_ev": energy_ev},
    )
    reduced = NRSSIntegrator(force_np_backend=True).integrateImageStack(img)

    q = np.asarray(reduced.coords["q"].values, dtype=np.float64)
    q_perp = np.asarray(reduced.coords["q_perp"].values, dtype=np.float64)
    wavelength_nm = 1239.84197 / energy_ev
    k = 2.0 * np.pi / wavelength_nm
    expected = np.full_like(q_perp, np.nan)
    valid = (k * k - q_perp * q_perp) >= 0.0
    qz = -k + np.sqrt(k * k - q_perp[valid] * q_perp[valid])
    expected[valid] = np.sqrt(q_perp[valid] * q_perp[valid] + qz * qz)

    np.testing.assert_allclose(q[valid], expected[valid], rtol=0.0, atol=1e-12)
    assert np.nanmax(np.abs(q[valid] - q_perp[valid])) > 1e-3
    assert reduced.attrs["radial_semantics"] == "q_abs_detector_corrected"
    assert reduced.attrs["nrss_semantic_mode"] == "3d_detector_aware"


def test_nrss_integrator_falls_back_to_explicit_kwargs():
    img = _synthetic_qxy_image(phys_size_nm=5.0, attrs={})
    reduced = NRSSIntegrator(force_np_backend=True).integrateImageStack(
        img,
        phys_size_nm=5.0,
        z_dim=64,
        energy_ev=285.0,
    )

    assert isinstance(reduced, xr.DataArray)
    assert reduced.attrs["phys_size_nm"] == 5.0
    assert reduced.attrs["z_dim"] == 64
    assert reduced.attrs["energy_ev"] == 285.0
    assert reduced.attrs["nrss_semantic_mode"] == "3d_detector_aware"


def test_nrss_integrator_preserves_stack_axis():
    img0 = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 1, "phys_size_nm": 5.0},
        dims=("qx", "qy"),
    )
    img1 = (1.15 * img0).assign_coords(qx=img0.qx, qy=img0.qy)
    stacked = xr.concat([img0, img1], dim=xr.IndexVariable("energy", [285.0, 286.0]))

    reduced = NRSSIntegrator(force_np_backend=True).integrateImageStack(stacked)

    assert reduced.dims == ("energy", "chi", "q")
    assert list(reduced.coords["energy"].values) == [285.0, 286.0]
    assert reduced.sizes["chi"] == 360
    assert reduced.attrs["radial_semantics"] == "q_perp"


def test_nrss_integrator_batched_matches_legacy_for_2d_stack():
    img0 = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 1, "phys_size_nm": 5.0},
        dims=("qx", "qy"),
    )
    img1 = (1.15 * img0).assign_coords(qx=img0.qx, qy=img0.qy)
    stacked = xr.concat([img0, img1], dim=xr.IndexVariable("energy", [285.0, 286.0]))

    integrator = NRSSIntegrator(force_np_backend=True)
    legacy = integrator.integrateImageStack(stacked, method="legacy")
    batched = integrator.integrateImageStack(stacked, method="batched")

    np.testing.assert_allclose(batched.values, legacy.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batched.q.values, legacy.q.values, rtol=0.0, atol=1e-12)


def test_nrss_integrator_3d_stack_uses_shared_physical_q_axis():
    img0 = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 64, "phys_size_nm": 5.0},
        dims=("qx", "qy"),
    )
    img1 = (1.15 * img0).assign_coords(qx=img0.qx, qy=img0.qy)
    stacked = xr.concat([img0, img1], dim=xr.IndexVariable("energy", [285.0, 286.0]))

    reduced = NRSSIntegrator(force_np_backend=True).integrateImageStack(stacked)

    q = np.asarray(reduced.coords["q"].values, dtype=np.float64)
    assert reduced.dims == ("energy", "chi", "q")
    assert reduced.attrs["radial_semantics"] == "q_abs_detector_corrected"
    assert reduced.attrs["radial_coordinate_mode"] == "shared_q_grid_interpolated"
    assert "q_abs" in reduced.coords
    assert np.issubdtype(q.dtype, np.floating)
    assert np.all(np.diff(q) > 0.0)
    assert not np.array_equal(q, np.arange(q.size, dtype=np.float64))


def test_nrss_integrator_batched_matches_legacy_for_3d_stack():
    img0 = _synthetic_qxy_image(
        phys_size_nm=5.0,
        attrs={"z_dim": 64, "phys_size_nm": 5.0},
        dims=("qx", "qy"),
    )
    img1 = (1.15 * img0).assign_coords(qx=img0.qx, qy=img0.qy)
    stacked = xr.concat([img0, img1], dim=xr.IndexVariable("energy", [285.0, 286.0]))

    integrator = NRSSIntegrator(force_np_backend=True)
    legacy = integrator.integrateImageStack(stacked, method="legacy")
    batched = integrator.integrateImageStack(stacked, method="batched")

    np.testing.assert_allclose(batched.values, legacy.values, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(batched.q.values, legacy.q.values, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(batched.q_abs.values, legacy.q_abs.values, rtol=0.0, atol=1e-12)


def test_wp_integrator_semantics_note():
    img = _synthetic_qxy_image()
    reduced = WPIntegrator(force_np_backend=True).integrateImageStack(img)

    assert reduced.attrs["radial_semantics"] == "q_perp"
    assert reduced.attrs["source_integrator"] == "WPIntegrator"


def test_wp_integrator_preserves_minimal_nrss_metadata_attrs():
    img = _synthetic_qxy_image(phys_size_nm=5.0, attrs={"phys_size_nm": 5.0, "z_dim": 64})
    reduced = WPIntegrator(force_np_backend=True).integrateImageStack(img)

    assert reduced.attrs["phys_size_nm"] == 5.0
    assert reduced.attrs["z_dim"] == 64
    assert reduced.attrs["radial_semantics"] == "q_perp"


def test_nrss_integrator_raises_on_phys_size_q_axis_mismatch_by_default():
    img = _synthetic_qxy_image(attrs={"phys_size_nm": 5.0, "z_dim": 64, "energy_ev": 285.0})
    with np.testing.assert_raises_regex(
        ValueError,
        "inconsistent q coordinates",
    ):
        NRSSIntegrator(force_np_backend=True).integrateImageStack(img)


def test_nrss_integrator_can_warn_or_skip_phys_size_q_axis_validation():
    img = _synthetic_qxy_image(attrs={"phys_size_nm": 5.0, "z_dim": 64, "energy_ev": 285.0})

    with pytest.warns(UserWarning, match="inconsistent q coordinates"):
        reduced_warn = NRSSIntegrator(
            force_np_backend=True,
            validate_q_coords_against_phys_size="warn",
        ).integrateImageStack(img)
    assert isinstance(reduced_warn, xr.DataArray)

    reduced_skip = NRSSIntegrator(
        force_np_backend=True,
        validate_q_coords_against_phys_size=False,
    ).integrateImageStack(img)
    assert isinstance(reduced_skip, xr.DataArray)
