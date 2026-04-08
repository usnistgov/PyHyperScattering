import ast
import warnings

import numpy as np
import xarray as xr

from PyHyperScattering.WPIntegrator import WPIntegrator


class NRSSIntegrator(WPIntegrator):
    """
    Integrator for NRSS/CyRSoXS-style detector outputs stored on qx/qy coordinates.

    In reciprocal-plane / 2D mode the reduction is equivalent to WPIntegrator and the
    radial coordinate is detector-plane q_perp. In detector-aware / 3D mode the same
    polar remesh is used, but the radial coordinate is relabeled with detector-corrected
    |q| based on the NRSS backend geometry.
    """

    _TWO_D_MODES = {
        "2d",
        "2d_reciprocal_plane",
        "reciprocal_plane",
        "reciprocal-plane",
        "q_perp",
    }
    _THREE_D_MODES = {
        "3d",
        "3d_detector_aware",
        "detector_aware",
        "detector-aware",
        "q_abs_detector_corrected",
    }

    def __init__(
        self,
        return_cupy=False,
        force_np_backend=False,
        use_chunked_processing=False,
        phys_size_nm=None,
        shape_zyx=None,
        z_dim=None,
        energy_ev=None,
        projection_mode=None,
        validate_q_coords_against_phys_size="raise",
    ):
        super().__init__(
            return_cupy=return_cupy,
            force_np_backend=force_np_backend,
            use_chunked_processing=use_chunked_processing,
        )
        self._default_metadata = {
            "phys_size_nm": phys_size_nm,
            "shape_zyx": shape_zyx,
            "z_dim": z_dim,
            "energy_ev": energy_ev,
            "projection_mode": projection_mode,
            "validate_q_coords_against_phys_size": validate_q_coords_against_phys_size,
        }

    def integrateSingleImage(self, img, **metadata_kwargs):
        metadata = self._resolve_metadata(img, metadata_kwargs)
        return self._integrate_single_image(img, metadata)

    def integrateImageStack(self, img_stack, method=None, chunksize=None, **metadata_kwargs):
        if (self.use_chunked_processing and method is None) or method == "dask":
            raise NotImplementedError(
                "NRSSIntegrator does not support dask-backed reduction yet because "
                "detector-corrected q coordinates can vary between slices."
            )
        if method is None or method == "legacy":
            return self.integrateImageStack_legacy(img_stack, **metadata_kwargs)
        raise NotImplementedError(f"unsupported integration method {method}")

    def integrateImageStack_legacy(self, data, **metadata_kwargs):
        spatial_dims = self._spatial_dims(data)
        index_dims = [dim for dim in data.dims if dim not in spatial_dims]
        if len(index_dims) == 0:
            return self.integrateSingleImage(data, **metadata_kwargs)

        stacked_name = None
        if len(index_dims) == 1:
            stacked = data
            stacked_name = index_dims[0]
        else:
            stacked_name = "pyhyper_internal_multiindex"
            stacked = data.stack({stacked_name: index_dims})

        arrays = []
        q_axes = []
        q_perp_axis = None
        common_attrs = None
        common_mode = None
        q_semantics_vary = False

        for i in range(stacked.sizes[stacked_name]):
            reduced = self.integrateSingleImage(stacked.isel({stacked_name: i}, drop=False), **metadata_kwargs)
            arrays.append(np.asarray(reduced.values))
            q_axes.append(np.asarray(reduced.coords["q"].values, dtype=np.float64))
            if "q_perp" in reduced.coords:
                q_perp_axis = np.asarray(reduced.coords["q_perp"].values, dtype=np.float64)
            else:
                q_perp_axis = np.asarray(reduced.coords["q"].values, dtype=np.float64)
            if common_attrs is None:
                common_attrs = dict(reduced.attrs)
                common_mode = reduced.attrs.get("nrss_semantic_mode")
            elif reduced.attrs.get("nrss_semantic_mode") != common_mode:
                q_semantics_vary = True

        if common_attrs is None:
            raise AssertionError("NRSSIntegrator did not produce any reduced slices.")

        result = xr.DataArray(
            np.stack(arrays, axis=0),
            dims=[stacked_name, "chi", "q"],
            coords={
                stacked_name: stacked.coords[stacked_name],
                "chi": reduced.coords["chi"].values,
            },
            attrs=common_attrs,
        )

        q_axes_are_same = self._allclose_1d(q_axes)
        if q_axes_are_same and not q_semantics_vary:
            result = result.assign_coords(q=q_axes[0])
            if q_perp_axis is not None and not np.allclose(q_axes[0], q_perp_axis, atol=0.0, rtol=0.0):
                result = result.assign_coords(q_perp=("q", q_perp_axis))
        else:
            result = result.assign_coords(q=np.arange(result.sizes["q"], dtype=np.int64))
            result = result.assign_coords(q_abs=((stacked_name, "q"), np.stack(q_axes, axis=0)))
            if q_perp_axis is not None:
                result = result.assign_coords(q_perp=("q", q_perp_axis))
            result.attrs["radial_coordinate_mode"] = "per_slice_q_abs"
            result.attrs["q_axis_note"] = (
                "The q dimension indexes radial bins. Exact detector-corrected q values are "
                "stored in the q_abs coordinate."
            )
            result.attrs.pop("energy_ev", None)

        if len(index_dims) > 1:
            result = result.unstack(stacked_name)
            result = result.transpose(*index_dims, "chi", "q")
        return result

    def _integrate_single_image(self, img, metadata):
        img_to_integ = np.asarray(img.values).squeeze()
        if img_to_integ.ndim != 2:
            raise ValueError(
                f"NRSSIntegrator expects a single detector image after squeezing, got shape {img_to_integ.shape!r}."
            )

        center_x = self._axis_center(img.qx, "qx")
        center_y = self._axis_center(img.qy, "qy")
        radius = np.sqrt((img_to_integ.shape[0] - center_x) ** 2 + (img_to_integ.shape[1] - center_y) ** 2)

        if self.MACHINE_HAS_CUDA:
            two_d = self.warp_polar_gpu(img_to_integ, center=(center_x, center_y), radius=radius)
        else:
            import skimage

            two_d = skimage.transform.warp_polar(img_to_integ, center=(center_x, center_y), radius=radius)

        q_perp_axis = self._q_perp_axis(img, int(two_d.shape[1]))
        radial_axis = q_perp_axis
        q_coord_name = "q"
        if metadata["nrss_semantic_mode"] == "3d_detector_aware":
            radial_axis = self._detector_corrected_q(q_perp_axis, metadata["energy_ev"])

        chi = np.linspace(-179.5, 179.5, 360)
        attrs = dict(img.attrs)
        attrs.update(
            {
                "radial_semantics": metadata["radial_semantics"],
                "source_integrator": "NRSSIntegrator",
                "nrss_semantic_mode": metadata["nrss_semantic_mode"],
                "phys_size_nm": metadata["phys_size_nm"],
                "z_dim": metadata["z_dim"],
            }
        )
        if metadata["shape_zyx"] is not None:
            attrs["shape_zyx"] = tuple(metadata["shape_zyx"])
        if metadata["energy_ev"] is not None:
            attrs["energy_ev"] = float(metadata["energy_ev"])

        result = xr.DataArray(two_d, dims=["chi", "q"], coords={q_coord_name: radial_axis, "chi": chi}, attrs=attrs)
        if metadata["nrss_semantic_mode"] == "3d_detector_aware":
            result = result.assign_coords(q_perp=("q", q_perp_axis))
        return result

    def _resolve_metadata(self, img, metadata_kwargs):
        fallback = dict(self._default_metadata)
        fallback.update(metadata_kwargs)

        shape_zyx = self._shape_from_attrs(img.attrs)
        if shape_zyx is None:
            shape_zyx = self._shape_from_value(fallback.get("shape_zyx"))

        z_dim = self._z_dim_from_attrs(img.attrs)
        if z_dim is None and shape_zyx is not None:
            z_dim = int(shape_zyx[0])
        if z_dim is None and fallback.get("z_dim") is not None:
            z_dim = int(fallback["z_dim"])
        if z_dim is None and fallback.get("shape_zyx") is not None:
            z_dim = int(self._shape_from_value(fallback["shape_zyx"])[0])

        projection_mode = self._projection_mode_from_attrs(img.attrs)
        if projection_mode is None and fallback.get("projection_mode") is not None:
            projection_mode = str(fallback["projection_mode"]).strip().lower()

        phys_size_nm = self._phys_size_from_attrs(img.attrs)
        if phys_size_nm is None and fallback.get("phys_size_nm") is not None:
            phys_size_nm = float(fallback["phys_size_nm"])

        validate_q_coords_against_phys_size = fallback.get("validate_q_coords_against_phys_size", "raise")
        if phys_size_nm is not None and validate_q_coords_against_phys_size:
            self._validate_q_axes_against_phys_size(
                img=img,
                phys_size_nm=phys_size_nm,
                mode=validate_q_coords_against_phys_size,
            )

        energy_ev = self._energy_from_attrs_or_coords(img)
        if energy_ev is None and fallback.get("energy_ev") is not None:
            energy_ev = float(fallback["energy_ev"])

        nrss_semantic_mode = self._semantic_mode(projection_mode=projection_mode, z_dim=z_dim)
        radial_semantics = "q_perp"
        if nrss_semantic_mode == "3d_detector_aware":
            radial_semantics = "q_abs_detector_corrected"
            if energy_ev is None:
                raise ValueError(
                    "NRSSIntegrator needs a scalar energy for detector-aware 3D reductions. "
                    "Populate img.attrs['energy_ev'], provide a scalar energy coordinate, or pass energy_ev=..."
                )

        return {
            "phys_size_nm": phys_size_nm,
            "shape_zyx": shape_zyx,
            "z_dim": z_dim,
            "energy_ev": energy_ev,
            "nrss_semantic_mode": nrss_semantic_mode,
            "radial_semantics": radial_semantics,
        }

    @staticmethod
    def _spatial_dims(img):
        missing = [name for name in ("qx", "qy") if name not in img.coords]
        if missing:
            raise ValueError(f"NRSSIntegrator requires qx/qy coordinates, missing {missing}.")
        dims = tuple(dim for dim in img.dims if dim in {"qx", "qy"})
        if set(dims) != {"qx", "qy"}:
            warnings.warn(
                "NRSSIntegrator found qx/qy coordinates but not both as dimensions. "
                "Reduction will proceed using the available qx/qy coordinates.",
                stacklevel=2,
            )
        return ("qx", "qy")

    @staticmethod
    def _axis_center(axis, name):
        return float(
            xr.DataArray(np.linspace(0, len(axis) - 1, len(axis)))
            .assign_coords({"dim_0": axis.values})
            .rename({"dim_0": name})
            .interp({name: 0})
            .data
        )

    @staticmethod
    def _q_perp_axis(img, n_points):
        q = np.sqrt(img.qy ** 2 + img.qx ** 2)
        return np.linspace(0.0, float(np.nanmax(q)), int(n_points), dtype=np.float64)

    @staticmethod
    def _detector_corrected_q(q_perp_axis, energy_ev):
        q_perp_axis = np.asarray(q_perp_axis, dtype=np.float64)
        wavelength_nm = 1239.84197 / float(energy_ev)
        k = 2.0 * np.pi / wavelength_nm
        val = k * k - q_perp_axis * q_perp_axis
        qz = np.full_like(q_perp_axis, np.nan, dtype=np.float64)
        q = np.full_like(q_perp_axis, np.nan, dtype=np.float64)
        valid = val >= 0.0
        qz[valid] = -k + np.sqrt(val[valid])
        q[valid] = np.sqrt(q_perp_axis[valid] * q_perp_axis[valid] + qz[valid] * qz[valid])
        return q

    @staticmethod
    def _expected_detector_axis(n_points, phys_size_nm):
        return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(int(n_points), d=float(phys_size_nm)))

    def _validate_q_axes_against_phys_size(self, img, phys_size_nm, mode):
        qx = np.asarray(img.qx.values, dtype=np.float64)
        qy = np.asarray(img.qy.values, dtype=np.float64)
        expected_qx = self._expected_detector_axis(qx.size, phys_size_nm)
        expected_qy = self._expected_detector_axis(qy.size, phys_size_nm)
        qx_ok = np.allclose(qx, expected_qx, atol=1e-12, rtol=0.0, equal_nan=True)
        qy_ok = np.allclose(qy, expected_qy, atol=1e-12, rtol=0.0, equal_nan=True)
        if qx_ok and qy_ok:
            return

        message = (
            "NRSSIntegrator detected inconsistent q coordinates for the provided phys_size_nm. "
            "qx/qy are authoritative for reduction, but they do not match the FFT-style detector "
            "axes implied by phys_size_nm."
        )
        if mode == "raise":
            raise ValueError(message)
        if mode == "warn":
            warnings.warn(message, stacklevel=2)
            return
        raise ValueError(
            "validate_q_coords_against_phys_size must be one of False, 'warn', or 'raise'. "
            f"Got {mode!r}."
        )

    def _semantic_mode(self, projection_mode, z_dim):
        if projection_mode is not None:
            mode = str(projection_mode).strip().lower()
            if mode in self._TWO_D_MODES:
                return "2d_reciprocal_plane"
            if mode in self._THREE_D_MODES:
                return "3d_detector_aware"
            raise ValueError(f"Unrecognized NRSS projection_mode {projection_mode!r}.")
        if z_dim is None:
            raise ValueError(
                "NRSSIntegrator could not determine whether the input is 2D reciprocal-plane or "
                "3D detector-aware. Provide z_dim, shape_zyx, or projection_mode metadata."
            )
        return "2d_reciprocal_plane" if int(z_dim) == 1 else "3d_detector_aware"

    @staticmethod
    def _shape_from_attrs(attrs):
        for key in ("shape_zyx", "num_zyx", "NumZYX"):
            if key in attrs:
                return NRSSIntegrator._shape_from_value(attrs[key])
        return None

    @staticmethod
    def _z_dim_from_attrs(attrs):
        if "z_dim" in attrs:
            return int(attrs["z_dim"])
        shape_zyx = NRSSIntegrator._shape_from_attrs(attrs)
        if shape_zyx is None:
            return None
        return int(shape_zyx[0])

    @staticmethod
    def _phys_size_from_attrs(attrs):
        for key in ("phys_size_nm", "PhysSize"):
            if key in attrs and attrs[key] is not None:
                return float(attrs[key])
        return None

    @staticmethod
    def _projection_mode_from_attrs(attrs):
        for key in ("nrss_output_semantics", "nrss_semantic_mode", "projection_mode"):
            value = attrs.get(key)
            if value is not None:
                return str(value).strip().lower()
        return None

    @staticmethod
    def _energy_from_attrs_or_coords(img):
        for key in ("energy_ev", "Energy"):
            value = img.attrs.get(key)
            if value is not None:
                return float(value)

        if "energy" not in img.coords:
            return None

        energy_values = np.asarray(img.coords["energy"].values, dtype=np.float64).reshape(-1)
        if energy_values.size == 1:
            return float(energy_values[0])
        return None

    @staticmethod
    def _shape_from_value(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        shape = tuple(int(v) for v in value)
        if len(shape) != 3:
            raise ValueError(f"shape_zyx/num_zyx must contain exactly 3 entries, got {shape!r}.")
        return shape

    @staticmethod
    def _allclose_1d(arrays):
        if len(arrays) <= 1:
            return True
        ref = np.asarray(arrays[0], dtype=np.float64)
        for arr in arrays[1:]:
            candidate = np.asarray(arr, dtype=np.float64)
            if candidate.shape != ref.shape:
                return False
            if not np.allclose(candidate, ref, equal_nan=True, rtol=0.0, atol=1e-12):
                return False
        return True
