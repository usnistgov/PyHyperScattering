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
        result = self.integrateImageStack_batched(img, **metadata_kwargs)
        squeeze_dims = [dim for dim in result.dims if dim not in {"chi", "q"} and result.sizes[dim] == 1]
        if squeeze_dims:
            result = result.squeeze(squeeze_dims, drop=True)
        return result

    def integrateImageStack(self, img_stack, method=None, chunksize=None, **metadata_kwargs):
        if (self.use_chunked_processing and method is None) or method == "dask":
            raise NotImplementedError(
                "NRSSIntegrator does not support dask-backed reduction yet because "
                "detector-corrected q coordinates can vary between slices."
            )
        if method is None or method == "legacy" or method == "batched":
            return self.integrateImageStack_batched(img_stack, **metadata_kwargs)
        raise NotImplementedError(f"unsupported integration method {method}")

    def integrateImageStack_legacy(self, data, **metadata_kwargs):
        return self.integrateImageStack_batched(data, **metadata_kwargs)

    def integrateImageStack_batched(self, data, **metadata_kwargs):
        stacked, stacked_name, index_dims, spatial_dims = self._prepare_batched_input(data)
        values = np.asarray(stacked.values)
        if values.ndim != 3:
            raise ValueError(f"NRSSIntegrator expected stacked data with 3 dims, got shape {values.shape!r}.")

        metadata = [self._resolve_metadata(stacked.isel({stacked_name: i}, drop=False), metadata_kwargs) for i in range(values.shape[0])]
        reduced_values, q_axes, q_perp_axis, chi, common_attrs, common_mode, q_semantics_vary = (
            self._integrate_batched_array(stacked, values, metadata, spatial_dims)
        )

        result = xr.DataArray(
            reduced_values,
            dims=[stacked_name, "chi", "q"],
            coords={
                stacked_name: stacked.coords[stacked_name],
                "chi": chi,
            },
            attrs=common_attrs,
        )

        result = self._assign_output_q_coords(result, stacked_name, q_axes, q_perp_axis, common_mode, q_semantics_vary)

        if len(index_dims) == 0:
            return result.isel({stacked_name: 0}, drop=True)
        if len(index_dims) > 1:
            result = result.unstack(stacked_name)
            result = result.transpose(*index_dims, "chi", "q")
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
    def _prepare_batched_input(data):
        spatial_dims = NRSSIntegrator._spatial_dims(data)
        index_dims = [dim for dim in data.dims if dim not in spatial_dims]
        spatial_dims_in_order = tuple(dim for dim in data.dims if dim in {"qx", "qy"})

        if len(index_dims) == 0:
            stacked_name = "pyhyper_internal_batch"
            stacked = data.expand_dims({stacked_name: [0]})
        elif len(index_dims) == 1:
            stacked_name = index_dims[0]
            stacked = data
        else:
            stacked_name = "pyhyper_internal_multiindex"
            stacked = data.stack({stacked_name: index_dims})

        stacked = stacked.transpose(stacked_name, *spatial_dims_in_order)
        return stacked, stacked_name, index_dims, spatial_dims_in_order

    def _integrate_batched_array(self, stacked, values, metadata, spatial_dims):
        center_x = self._axis_center(stacked.qx, "qx")
        center_y = self._axis_center(stacked.qy, "qy")
        center_lookup = {
            "qx": center_x,
            "qy": center_y,
        }
        center = tuple(center_lookup[dim] for dim in spatial_dims)
        radius = np.sqrt((values.shape[1] - center[0]) ** 2 + (values.shape[2] - center[1]) ** 2)
        reduced_values = self._warp_polar_batched(values, center=center, radius=radius)

        q_perp_axis = self._q_perp_axis(stacked, int(reduced_values.shape[-1]))
        q_axes, common_mode, q_semantics_vary = self._batched_q_axes(q_perp_axis, metadata)
        chi = np.linspace(-179.5, 179.5, reduced_values.shape[1])
        common_attrs = self._build_batched_attrs(stacked.attrs, metadata[0])
        return reduced_values, q_axes, q_perp_axis, chi, common_attrs, common_mode, q_semantics_vary

    def _assign_output_q_coords(self, result, stacked_name, q_axes, q_perp_axis, common_mode, q_semantics_vary):
        q_axes_are_same = self._allclose_1d(q_axes)
        if q_axes_are_same and not q_semantics_vary:
            result = result.assign_coords(q=q_axes[0])
            if q_perp_axis is not None and not np.allclose(q_axes[0], q_perp_axis, atol=0.0, rtol=0.0):
                result = result.assign_coords(q_perp=("q", q_perp_axis))
            return result

        if common_mode == "3d_detector_aware" and not q_semantics_vary:
            q_common = self._shared_q_grid(q_axes)
            if q_common is not None:
                interpolated = self._interp_stack_to_common_q(np.asarray(result.values), q_axes, q_common)
                result = xr.DataArray(
                    interpolated,
                    dims=result.dims,
                    coords={
                        stacked_name: result.coords[stacked_name],
                        "chi": result.coords["chi"].values,
                        "q": q_common,
                        "q_abs": ((stacked_name, "q"), np.stack(q_axes, axis=0)),
                    },
                    attrs=dict(result.attrs),
                )
                result.attrs["radial_coordinate_mode"] = "shared_q_grid_interpolated"
                result.attrs["q_axis_note"] = (
                    "The q dimension is a shared detector-corrected q grid spanning the overlap "
                    "of all slices. Exact per-slice q values before interpolation remain in q_abs."
                )
                result.attrs.pop("energy_ev", None)
                return result

        return self._attach_per_slice_q_index(result, stacked_name, q_axes, q_perp_axis)

    @staticmethod
    def _build_batched_attrs(base_attrs, metadata):
        attrs = dict(base_attrs)
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
        return attrs

    @staticmethod
    def _batched_q_axes(q_perp_axis, metadata):
        q_axes = []
        common_mode = None
        q_semantics_vary = False
        three_d_indices = []
        three_d_energies = []

        for i, md in enumerate(metadata):
            mode = md["nrss_semantic_mode"]
            if common_mode is None:
                common_mode = mode
            elif mode != common_mode:
                q_semantics_vary = True
            q_axes.append(np.asarray(q_perp_axis, dtype=np.float64))
            if mode == "3d_detector_aware":
                three_d_indices.append(i)
                three_d_energies.append(md["energy_ev"])

        if three_d_indices:
            corrected = NRSSIntegrator._detector_corrected_q_batch(q_perp_axis, np.asarray(three_d_energies, dtype=np.float64))
            for idx, q_axis in zip(three_d_indices, corrected):
                q_axes[idx] = q_axis

        return q_axes, common_mode, q_semantics_vary

    def _warp_polar_batched(self, values, center, radius):
        if self.MACHINE_HAS_CUDA:
            try:
                import cupy as cp
            except ImportError:  # pragma: no cover
                return self._warp_polar_batched_numpy(values, center=center, radius=radius)

            values_xp = cp.asarray(values)
            reduced = self._warp_polar_batched_xp(values_xp, center=center, radius=radius, xp=cp)
            if self.return_cupy:
                return reduced
            return cp.asnumpy(reduced)

        return self._warp_polar_batched_numpy(values, center=center, radius=radius)

    def _warp_polar_batched_numpy(self, values, center, radius):
        reduced = self._warp_polar_batched_xp(np.asarray(values), center=center, radius=radius, xp=np)
        return np.asarray(reduced)

    @staticmethod
    def _warp_polar_batched_xp(values, center, radius, xp):
        values = xp.asarray(values)
        n_images, n_rows, n_cols = values.shape
        n_theta = 360
        n_radius = int(np.ceil(radius))
        if n_radius <= 0:
            raise ValueError(f"NRSSIntegrator computed a non-positive polar radius {radius!r}.")

        center_row, center_col = center
        theta = xp.deg2rad(xp.arange(n_theta, dtype=xp.float64))
        radial = xp.arange(n_radius, dtype=xp.float64) * (float(radius) / n_radius)
        radial_grid, theta_grid = xp.meshgrid(radial, theta)

        row_coords = radial_grid * xp.sin(theta_grid) + center_row
        col_coords = radial_grid * xp.cos(theta_grid) + center_col

        row0 = xp.floor(row_coords).astype(xp.int64)
        col0 = xp.floor(col_coords).astype(xp.int64)
        row1 = row0 + 1
        col1 = col0 + 1

        row_weight = row_coords - row0
        col_weight = col_coords - col0

        def sample(row_idx, col_idx):
            valid = (row_idx >= 0) & (row_idx < n_rows) & (col_idx >= 0) & (col_idx < n_cols)
            row_clip = xp.clip(row_idx, 0, n_rows - 1)
            col_clip = xp.clip(col_idx, 0, n_cols - 1)
            sampled = values[:, row_clip, col_clip]
            return sampled * valid[None, :, :]

        top_left = sample(row0, col0)
        top_right = sample(row0, col1)
        bottom_left = sample(row1, col0)
        bottom_right = sample(row1, col1)

        return (
            top_left * (1.0 - row_weight)[None, :, :] * (1.0 - col_weight)[None, :, :]
            + top_right * (1.0 - row_weight)[None, :, :] * col_weight[None, :, :]
            + bottom_left * row_weight[None, :, :] * (1.0 - col_weight)[None, :, :]
            + bottom_right * row_weight[None, :, :] * col_weight[None, :, :]
        )

    @staticmethod
    def _detector_corrected_q_batch(q_perp_axis, energy_ev):
        q_perp_axis = np.asarray(q_perp_axis, dtype=np.float64)[None, :]
        energy_ev = np.asarray(energy_ev, dtype=np.float64).reshape(-1, 1)
        wavelength_nm = 1239.84197 / energy_ev
        k = 2.0 * np.pi / wavelength_nm
        val = k * k - q_perp_axis * q_perp_axis
        valid = val >= 0.0
        qz = -k + np.sqrt(val, where=valid, out=np.full_like(val, np.nan, dtype=np.float64))
        q = np.full_like(val, np.nan, dtype=np.float64)
        q_perp_broadcast = np.broadcast_to(q_perp_axis, val.shape)
        q[valid] = np.sqrt(q_perp_broadcast[valid] * q_perp_broadcast[valid] + qz[valid] * qz[valid])
        return q

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

    @staticmethod
    def _shared_q_grid(q_axes):
        lower_bounds = []
        upper_bounds = []
        n_points = None

        for axis in q_axes:
            axis = np.asarray(axis, dtype=np.float64)
            finite = axis[np.isfinite(axis)]
            if finite.size < 2:
                return None
            lower_bounds.append(float(np.min(finite)))
            upper_bounds.append(float(np.max(finite)))
            n_points = axis.size if n_points is None else min(n_points, axis.size)

        q_min = max(lower_bounds)
        q_max = min(upper_bounds)
        if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min or n_points is None or n_points < 2:
            return None
        return np.linspace(q_min, q_max, int(n_points), dtype=np.float64)

    @staticmethod
    def _interp_stack_to_common_q(values, q_axes, q_common):
        values = np.asarray(values)
        output = np.full((values.shape[0], values.shape[1], q_common.size), np.nan, dtype=values.dtype)

        for i, q_axis in enumerate(q_axes):
            q_axis = np.asarray(q_axis, dtype=np.float64)
            valid = np.isfinite(q_axis)
            q_valid = q_axis[valid]
            if q_valid.size < 2:
                continue

            q_valid, unique_idx = np.unique(q_valid, return_index=True)
            slice_values = values[i][:, valid][:, unique_idx]
            for chi_idx in range(values.shape[1]):
                output[i, chi_idx, :] = np.interp(
                    q_common,
                    q_valid,
                    slice_values[chi_idx],
                    left=np.nan,
                    right=np.nan,
                )

        return output

    @staticmethod
    def _attach_per_slice_q_index(result, stacked_name, q_axes, q_perp_axis):
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
        return result
