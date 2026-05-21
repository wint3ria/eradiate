from __future__ import annotations

from typing import Literal

import attrs
import numpy as np
import pandas as pd
import xarray as xr

from eradiate import traverse

from ._core import PhaseFunction
from ...attrs import define, documented
from ...grid import GridCoords, PlaneParallelGridCoords
from ...kernel import SceneParameter
from ...scenes.atmosphere import AtmosphericMedium, ParticleLayer
from ...scenes.geometry import PlaneParallelGeometry
from ...scenes.phase import ParticlePhase
from ...spectral.index import MonoSpectralIndex, SpectralIndex
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


def format_particles_dataset(
    iprt_ds: xr.Dataset,
    v_eff: np.ndarray,
    particle_shape: Literal["spherical", "spheroidal"] = "spherical",
) -> xr.Dataset:
    """
    Reindex an IPRT standard Mie particles property dataset and return a compact :class:`xarray.Dataset`.

    The 4-component phase matrix is expanded to the full :math:`4 \\times 4`
    Mueller form.  Scattering angles and phase data are stored as contiguous
    1-D arrays (``theta_native``, ``phase_native``) together with per-entry
    ``start`` / ``n_pts`` index arrays, so that the data for entry
    ``(iw, ir, iv)`` occupies a zero-copy, NaN-free slice.

    Parameters
    ----------
    iprt_ds : xr.Dataset
        IPRT standard Mie particles property dataset with dimensions
        ``nlam``, ``nreff``, ``nthetamax``, and ``nphamat``, and variables
        ``phase``, ``theta``, ``ext``, ``ssa``, ``wavelen``, and ``reff``.
    v_eff : array-like
        Effective variances for the size distribution :math:`[\\mu m^2]`.
    particle_shape : {"spherical", "spheroidal"}, optional
        Symmetry class of the scatterer.

        * ``"spherical"`` — 4 independent Mueller elements
          :math:`[m_{11}, m_{12}, m_{33}, m_{34}]`, with
          :math:`m_{22} = m_{11}` and :math:`m_{44} = m_{33}`.
        * ``"spheroidal"`` — 6 independent Mueller elements
          :math:`[m_{11}, m_{12}, m_{22}, m_{33}, m_{34}, m_{44}]`.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates ``w``, ``r_eff``, ``v_eff``, ``rho``, and
        ``alpha``; and data variables ``theta_native``, ``phase_native``,
        ``start``, ``n_pts``, ``m_extinction``, and ``albedo``.

    Raises
    ------
    NotImplementedError
        If *particle_shape* is not ``"spherical"`` or ``"spheroidal"``.
    """
    v_eff = np.asarray(v_eff)
    nveff = len(v_eff)
    nlam = len(iprt_ds.nlam)
    nreff = len(iprt_ds.nreff)
    ntheta = len(iprt_ds.nthetamax)
    E = nlam * nreff * nveff

    phase_raw = iprt_ds.phase.values  # (nlam, nreff, 4, ntheta)
    phase_np = np.zeros((nlam, nreff, 4, 4, ntheta))

    if particle_shape == "spherical":
        for k in range(nlam):
            phase_np[k, :, 0, 0, :] = phase_raw[k, :, 0]
            phase_np[k, :, 1, 1, :] = phase_raw[k, :, 0]
            phase_np[k, :, 0, 1, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 0, :] = phase_raw[k, :, 1]
            phase_np[k, :, 2, 2, :] = phase_raw[k, :, 2]
            phase_np[k, :, 3, 3, :] = phase_raw[k, :, 2]
            phase_np[k, :, 2, 3, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 2, :] = phase_raw[k, :, 3]
    elif particle_shape == "spheroidal":
        for k in range(nlam):
            phase_np[k, :, 0, 0, :] = phase_raw[k, :, 0]
            phase_np[k, :, 0, 1, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 0, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 1, :] = phase_raw[k, :, 4]
            phase_np[k, :, 2, 2, :] = phase_raw[k, :, 2]
            phase_np[k, :, 2, 3, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 2, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 3, :] = phase_raw[k, :, 5]
    else:
        raise NotImplementedError(
            f"Particle shape '{particle_shape}' is not implemented. "
            "Use 'spherical' or 'spheroidal'."
        )

    theta_raw = iprt_ds.theta.isel(nphamat=0).values.reshape(nlam, nreff, nveff, ntheta)
    phase_4d = phase_np.reshape(nlam, nreff, nveff, 4, 4, ntheta)

    theta_flat = theta_raw.reshape(E, ntheta)
    phase_flat = phase_4d.reshape(E, 16, ntheta)

    valid_mask = ~np.isnan(theta_flat)
    valid_counts = valid_mask.sum(axis=1)
    total_pts = int(valid_counts.sum())

    start_flat = np.zeros(E, dtype=np.int64)
    start_flat[1:] = np.cumsum(valid_counts[:-1])

    theta_native = np.empty(total_pts, dtype=np.float64)
    phase_native = np.empty((total_pts, 16), dtype=np.float64)

    for e in range(E):
        nc = int(valid_counts[e])
        s = int(start_flat[e])
        valid_idx = np.where(valid_mask[e])[0]
        sidx = np.argsort(theta_flat[e, valid_idx])[::-1]
        sorted_idx = valid_idx[sidx]
        theta_native[s : s + nc] = theta_flat[e, sorted_idx]
        phase_native[s : s + nc, :] = phase_flat[e, :, sorted_idx]

    start = start_flat.reshape(nlam, nreff, nveff).astype(np.int64)
    n_pts = valid_counts.reshape(nlam, nreff, nveff).astype(np.int32)

    return xr.Dataset(
        coords=dict(
            w=(
                ["w"],
                iprt_ds.wavelen.values,
                {"long_name": "wavelength", "units": "micron"},
            ),
            r_eff=(
                ["r_eff"],
                iprt_ds.reff.values,
                {"long_name": "effective_radius", "units": "micron"},
            ),
            v_eff=(
                ["v_eff"],
                v_eff,
                {"long_name": "effective_variance", "units": "micron ** 2"},
            ),
            rho=(
                ["rho"],
                np.atleast_1d(
                    to_quantity(iprt_ds.rho.isel(nlam=0, nreff=0)).m_as("g/cm^3")
                ),
                {"long_name": "density", "units": "g/cm^3"},
            ),
            alpha=(["v_eff"], 1.0 / v_eff - 2.0),
        ),
        data_vars=dict(
            theta_native=(["total_pts"], theta_native, iprt_ds.theta.attrs.copy()),
            phase_native=(
                ["total_pts", "ch16"],
                phase_native,
                iprt_ds.phase.attrs.copy(),
            ),
            start=(["w", "r_eff", "v_eff"], start),
            n_pts=(["w", "r_eff", "v_eff"], n_pts),
            m_extinction=(
                ["w", "r_eff", "v_eff"],
                iprt_ds.ext.values.reshape(nlam, nreff, nveff),
                iprt_ds.ext.attrs.copy(),
            ),
            albedo=(
                ["w", "r_eff", "v_eff"],
                iprt_ds.ssa.values.reshape(nlam, nreff, nveff),
                dict(units="dimensionless"),
            ),
        ),
        attrs=iprt_ds.attrs.copy(),
    ).squeeze(dim="rho")


def interp_properties_wavelength(
    properties: xr.Dataset,
    wavelengths: ureg.Quantity,
) -> xr.Dataset:
    """Interpolate a ragged-array phase properties dataset along the wavelength dimension."""

    assert wavelengths.check(ucc.get("length"))
    wavelengths = np.atleast_1d(wavelengths)

    properties_wavelengths = to_quantity(properties.w)
    w_unit = properties_wavelengths.units

    if len(wavelengths) != len(np.unique(wavelengths.m_as(w_unit))):
        raise ValueError("Duplicate wavelengths in interpolation target.")

    if properties_wavelengths.size <= 1:
        assert np.all(
            np.abs(
                (properties_wavelengths - wavelengths.to(w_unit))
                / wavelengths.to(w_unit)
            )
            < 0.001
        )
        return properties

    if not np.all(wavelengths >= properties_wavelengths[0]):
        raise ValueError("Interpolation wavelength below properties spectral range.")
    if not np.all(wavelengths <= properties_wavelengths[-1]):
        raise ValueError("Interpolation wavelength above properties spectral range.")

    properties_w_values = properties_wavelengths.m_as(w_unit)
    target_w_values = wavelengths.m_as(w_unit)

    lower_bracket_indices = np.clip(
        np.searchsorted(properties_w_values, target_w_values, side="left"),
        0,
        len(properties_wavelengths) - 2,
    )
    upper_bracket_indices = lower_bracket_indices + 1

    bracket_table = pd.DataFrame(
        {
            "lower_bracket_index": lower_bracket_indices,
            "upper_bracket_index": upper_bracket_indices,
            "lower_bracket_w": properties_w_values[lower_bracket_indices],
            "upper_bracket_w": properties_w_values[upper_bracket_indices],
            "target_w": target_w_values,
        }
    )

    n_r_eff = properties.sizes["r_eff"]
    n_v_eff = properties.sizes["v_eff"]
    n_channels = properties.sizes["ch16"]

    accumulated_theta = []
    accumulated_phase = []
    output_start = np.zeros((len(wavelengths), n_r_eff, n_v_eff), dtype=np.int64)
    output_n_pts = np.zeros((len(wavelengths), n_r_eff, n_v_eff), dtype=np.int32)

    target_w_to_output_index = {w: i for i, w in enumerate(target_w_values)}
    current_flat_index = 0

    for (lower_wi, upper_wi), bracket_group in bracket_table.groupby(
        ["lower_bracket_index", "upper_bracket_index"]
    ):
        lower_w = properties_w_values[lower_wi]
        upper_w = properties_w_values[upper_wi]

        for r_eff_index in range(n_r_eff):
            for v_eff_index in range(n_v_eff):
                lower_start = properties.start.values[
                    lower_wi, r_eff_index, v_eff_index
                ]
                lower_n_pts = properties.n_pts.values[
                    lower_wi, r_eff_index, v_eff_index
                ]
                lower_theta = properties.theta_native.values[
                    lower_start : lower_start + lower_n_pts
                ]
                lower_phase = properties.phase_native.values[
                    lower_start : lower_start + lower_n_pts, :
                ]

                upper_start = properties.start.values[
                    upper_wi, r_eff_index, v_eff_index
                ]
                upper_n_pts = properties.n_pts.values[
                    upper_wi, r_eff_index, v_eff_index
                ]
                upper_theta = properties.theta_native.values[
                    upper_start : upper_start + upper_n_pts
                ]
                upper_phase = properties.phase_native.values[
                    upper_start : upper_start + upper_n_pts, :
                ]

                merged_theta = np.union1d(lower_theta, upper_theta)[::-1]
                assert merged_theta[0] < merged_theta[-1]
                n_merged_pts = len(merged_theta)

                lower_phase_on_merged_theta = np.stack(
                    [
                        np.interp(merged_theta, lower_theta, lower_phase[:, c])
                        for c in range(n_channels)
                    ],
                    axis=-1,
                )
                upper_phase_on_merged_theta = np.stack(
                    [
                        np.interp(merged_theta, upper_theta, upper_phase[:, c])
                        for c in range(n_channels)
                    ],
                    axis=-1,
                )

                for _, target_row in bracket_group.iterrows():
                    target_w = target_row["target_w"]
                    wavelength_weight = (target_w - lower_w) / (upper_w - lower_w)
                    interpolated_phase = (
                        (1.0 - wavelength_weight) * lower_phase_on_merged_theta
                        + wavelength_weight * upper_phase_on_merged_theta
                    )

                    output_index = target_w_to_output_index[target_w]
                    output_start[output_index, r_eff_index, v_eff_index] = (
                        current_flat_index
                    )
                    output_n_pts[output_index, r_eff_index, v_eff_index] = n_merged_pts
                    accumulated_theta.append(merged_theta)
                    accumulated_phase.append(interpolated_phase)
                    current_flat_index += n_merged_pts

    output_theta = np.concatenate(accumulated_theta)
    output_phase = np.concatenate(accumulated_phase, axis=0)

    scalar_fields_interpolated = properties[["m_extinction", "albedo"]].interp(
        w=target_w_values, method="linear"
    )

    output_dataset = xr.Dataset(
        {
            "theta_native": xr.DataArray(
                output_theta, dims=["total_pts"], attrs=properties.attrs.copy()
            ),
            "phase_native": xr.DataArray(output_phase, dims=["total_pts", "ch16"]),
            "start": xr.DataArray(output_start, dims=["w", "r_eff", "v_eff"]),
            "n_pts": xr.DataArray(output_n_pts, dims=["w", "r_eff", "v_eff"]),
            "m_extinction": scalar_fields_interpolated.m_extinction,
            "albedo": scalar_fields_interpolated.albedo,
        },
        coords={
            "w": xr.DataArray(
                target_w_values,
                dims=["w"],
                attrs={"units": str(w_unit)},
            ),
            "r_eff": properties.r_eff,
            "v_eff": properties.v_eff,
            "rho": properties.rho,
            "alpha": properties.alpha,
        },
    )

    return output_dataset


def _compute_overlap(
    src_x0: float,
    src_x1: float,
    src_y0: float,
    src_y1: float,
    tgt_edges_x: np.ndarray,
    tgt_edges_y: np.ndarray,
) -> np.ndarray:
    overlap_x = np.maximum(
        0.0, np.minimum(src_x1, tgt_edges_x[1:]) - np.maximum(src_x0, tgt_edges_x[:-1])
    ) / (tgt_edges_x[1:] - tgt_edges_x[:-1])
    overlap_y = np.maximum(
        0.0, np.minimum(src_y1, tgt_edges_y[1:]) - np.maximum(src_y0, tgt_edges_y[:-1])
    ) / (tgt_edges_y[1:] - tgt_edges_y[:-1])
    return np.outer(overlap_x, overlap_y)


@define(eq=False, slots=False)
class ParticleField(AtmosphericMedium):
    """
    Atmospheric medium representing a cloud field with spatially heterogeneous
    physical properties.

    The cloud field is defined by a sparse volumetric *profile* (voxel-indexed
    microphysical properties) and a pre-computed *properties* lookup dataset
    (libRadtran standard Mie particles property dataset).

    At render time the profile is resampled onto the render grid, the per-voxel
    phase function is interpolated from the lookup table, and the result is
    handed to a :class:`.ParticlePhaseFunction` kernel plugin.

    Notes
    -----
    * The profile may have an irregular z-grid, but its x and y grids must
      be regular.
    * The render grid must be contained within the z-extent of the profile.
      A warning is emitted when the render z-resolution is more than twice
      coarser than the profile z-resolution.
    """

    profile: xr.Dataset = documented(
        attrs.field(kw_only=True),
        doc="Sparse volumetric cloud profile dataset.  Must contain 1-D "
        "variables ``r_eff``, ``v_eff``, and ``mass_density`` indexed by a "
        "flat ``index`` dimension, together with integer voxel coordinates "
        "``i_x``, ``i_y``, and ``i_z`` (1-based) and level-edge coordinates "
        "``x_levels``, ``y_levels``, and ``z_levels``. "
        "This parameter has no default.",
        type="xr.Dataset",
    )

    @profile.validator
    def _validate_profile(self, attribute, value):
        if "mass_density" not in value.data_vars:
            raise ValueError(
                "The profile dataset must contain a 'mass_density' variable."
            )

    properties: xr.Dataset = documented(
        attrs.field(kw_only=True),
        doc="Mie particles property dataset. Contains per-wavelength "
        "extinction coefficients and phase function data used as the "
        "lookup table for per-voxel optical property interpolation. "
        "This parameter has no default.",
        type="xr.Dataset",
    )

    z_interp_method: str = documented(
        attrs.field(kw_only=True, default="linear"),
        doc="Profile interpolation method against Z axis. profiles "
        "must be resampled to a regular grid before being used for "
        "inferring optical properties. Default method is linear.",
        type="str",
        default="linear",
    )

    has_absorption: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, the medium contributes an absorption coefficient.",
        type="bool",
        init_type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, the medium contributes a scattering coefficient.",
        type="bool",
        init_type="bool",
        default="True",
    )

    _phase: PhaseFunction = None

    def eval_sigma_t(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the extinction coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Extinction coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        wavelength = np.atleast_1d(si.w)
        grid = grid or self.geometry.grid
        resampled_profile = self._resample_profile_to_grid(grid, wavelength)
        properties = self._interpolate_properties(wavelength, grid)

        ix = resampled_profile.i_x.values
        iy = resampled_profile.i_y.values
        iz = resampled_profile.i_z.values

        r_eff_pts = xr.DataArray(
            np.clip(
                resampled_profile.r_eff.values,
                properties.r_eff.values.min(),
                properties.r_eff.values.max(),
            ),
            dims="index",
        )
        v_eff_pts = xr.DataArray(
            np.clip(
                resampled_profile.v_eff.values,
                properties.v_eff.values.min(),
                properties.v_eff.values.max(),
            ),
            dims="index",
        )
        w_value = wavelength.m_as(properties.w.attrs["units"])

        m_extinction = properties.m_extinction
        interp_coords = {}

        if m_extinction.sizes["w"] > 1:
            interp_coords["w"] = w_value
        else:
            m_extinction = m_extinction.squeeze("w")

        if m_extinction.sizes["r_eff"] > 1:
            interp_coords["r_eff"] = r_eff_pts
        else:
            m_extinction = m_extinction.squeeze("r_eff")

        if m_extinction.sizes["v_eff"] > 1:
            interp_coords["v_eff"] = v_eff_pts
        else:
            m_extinction = m_extinction.squeeze("v_eff")

        if interp_coords:
            m_extinction = m_extinction.interp(interp_coords, method="linear")

        extinction = (
            to_quantity(m_extinction) * to_quantity(resampled_profile.mass_density)
        ).to("1/km")

        sigma_t_values = np.zeros(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z),
            dtype=np.float32,
        )
        sigma_t_values[ix, iy, iz] = extinction.m
        sigma_t = ureg.Quantity(sigma_t_values, "1/km")

        if self.has_absorption and self.has_scattering:
            return sigma_t
        if self.has_scattering:
            return sigma_t - self.eval_sigma_a(si, grid)
        if self.has_absorption:
            return sigma_t - self.eval_sigma_s(si, grid)

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    def _build_r_eff_v_eff_grids(
        self, grid: GridCoords
    ) -> tuple[np.ndarray, np.ndarray]:
        resampled = self._resample_profile_to_grid(
            grid, np.atleast_1d(to_quantity(self.properties.wavelen.isel(nlam=0)))
        )

        r_eff_grid = np.full(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), np.nan, dtype=np.float32
        )
        v_eff_grid = np.full(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), np.nan, dtype=np.float32
        )

        ix = resampled.i_x.values
        iy = resampled.i_y.values
        iz = resampled.i_z.values
        r_eff_grid[ix, iy, iz] = resampled.r_eff.values
        v_eff_grid[ix, iy, iz] = resampled.v_eff.values

        return r_eff_grid, v_eff_grid

    def _eval_phase_data(self, si: SpectralIndex) -> xr.Dataset:
        return self._interpolate_properties(np.atleast_1d(si.w), self.geometry.grid)

    @property
    def phase(self) -> "ParticlePhase":
        if self._phase:
            return self._phase

        grid = self.geometry.grid
        r_eff_volume, v_eff_volume = self._build_r_eff_v_eff_grids(grid)
        properties = self._interpolate_properties(
            np.atleast_1d(to_quantity(self.properties.wavelen.isel(nlam=0))),
            grid,
        )
        return ParticlePhase(
            grid=grid,
            r_eff_volume=r_eff_volume,
            v_eff_volume=v_eff_volume,
            r_eff_grid=properties.r_eff.values.astype(np.float32),
            v_eff_grid=properties.v_eff.values.astype(np.float32),
            phase_data=lambda ctx: self._eval_phase_data(ctx.si),
            wrap_mode=str(self.geometry.wrap_mode),
            filter_type="nearest",
            blending_method="stochastic",
        )

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        _, result = traverse(self.phase)
        return result.data

    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the single-scattering albedo on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Single-scattering albedo array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, dimensionless.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        if self.has_absorption and self.has_scattering:
            wavelength = np.atleast_1d(si.w)
            grid = grid or self.geometry.grid
            resampled_profile = self._resample_profile_to_grid(grid, wavelength)
            properties = self._interpolate_properties(wavelength, grid)

            ix = resampled_profile.i_x.values
            iy = resampled_profile.i_y.values
            iz = resampled_profile.i_z.values

            r_eff_pts = xr.DataArray(
                np.clip(
                    resampled_profile.r_eff.values,
                    properties.r_eff.values.min(),
                    properties.r_eff.values.max(),
                ),
                dims="index",
            )
            v_eff_pts = xr.DataArray(
                np.clip(
                    resampled_profile.v_eff.values,
                    properties.v_eff.values.min(),
                    properties.v_eff.values.max(),
                ),
                dims="index",
            )
            w_value = wavelength.m_as(properties.w.attrs["units"])

            albedo_da = properties.albedo
            interp_coords = {}

            if albedo_da.sizes["w"] > 1:
                interp_coords["w"] = w_value
            else:
                albedo_da = albedo_da.squeeze("w")

            if albedo_da.sizes["r_eff"] > 1:
                interp_coords["r_eff"] = r_eff_pts
            else:
                albedo_da = albedo_da.squeeze("r_eff")

            if albedo_da.sizes["v_eff"] > 1:
                interp_coords["v_eff"] = v_eff_pts
            else:
                albedo_da = albedo_da.squeeze("v_eff")

            if interp_coords:
                albedo_da = albedo_da.interp(interp_coords, method="linear")

            albedo_values = np.zeros(
                (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float32
            )
            albedo_values[ix, iy, iz] = to_quantity(albedo_da).m
            return ureg.Quantity(albedo_values, "dimensionless")

        if self.has_absorption:
            return ureg.Quantity(0.0, "dimensionless")
        if self.has_scattering:
            return ureg.Quantity(1.0, "dimensionless")

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the scattering coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Scattering coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        if not self.has_scattering:
            return ureg.Quantity(
                np.zeros(grid.shape, dtype=np.float32),
                "1/km",
            )
        return self.eval_sigma_t(si, grid) * self.eval_albedo(si, grid).m_as(
            ureg.dimensionless
        )

    def eval_sigma_a(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the absorption coefficient on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Absorption coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.
        """
        grid = grid or self.geometry.grid
        if not self.has_absorption:
            return ureg.Quantity(
                np.zeros(grid.shape, dtype=np.float32),
                "1/km",
            )
        return self.eval_sigma_t(si, grid) * (
            1.0 - self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        )

    def eval_mfp(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> ureg.Quantity:
        """
        Evaluate the mean free path on the render grid.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.
        grid : :class:`.GridCoords`, optional
            Target render grid.  Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`pint.Quantity`
            Mean free path array of shape ``(n_cells_x, n_cells_y, n_cells_z)``,
            in metres.
        """
        grid = grid or self.geometry.grid
        sigma_t = self.eval_sigma_t(si, grid)
        sigma_t_values = sigma_t.m_as("1/m")
        mfp_values = np.full(sigma_t_values.shape, np.inf, dtype=np.float32)
        np.divide(1.0, sigma_t_values, where=sigma_t_values != 0, out=mfp_values)
        return ureg.Quantity(mfp_values, "m")

    @cache_by_id
    def _interpolate_properties(
        self,
        wavelengths: ureg.Quantity,
        grid: GridCoords,
    ) -> xr.Dataset:
        """
        Interpolate the cloud-particle properties dataset at the given wavelengths.

        The result is memoised by argument identity via :func:`.cache_by_id`.

        Parameters
        ----------
        wavelengths : :class:`pint.Quantity`
            Query wavelengths.
        grid : :class:`.GridCoords`
            Target render grid (used only for cache keying).

        Returns
        -------
        xr.Dataset
            Interpolated cloud-particle properties dataset in the layout
            produced by :func:`.interp_properties_wavelength`.
        """
        alpha = self.properties.attrs["param_alpha"]
        v_eff = 1 / np.asarray([alpha + 2.0])
        properties = format_particles_dataset(self.properties, v_eff)

        return interp_properties_wavelength(properties, wavelengths)

    @cache_by_id
    def _resample_profile_to_grid(
        self,
        grid: GridCoords,
        w: ureg.Quantity,
    ) -> xr.Dataset:
        """
        Resample the sparse cloud profile onto the cells of *grid*.
    
        The resampling strategy is controlled by ``self.z_interp_method``:
    
        ``"nearest"``
            Each target cell is assigned to the nearest source cell centre.
            Entries are deduplicated.
    
        ``"linear"``
            Each target cell receives a linear interpolation between the two
            bracketing source cell centres.
    
        In both modes, target cells that fall outside the source z range are
        treated as invalid and absent from the sparse output.
    
        Parameters
        ----------
        grid : :class:`.GridCoords`
            Target render grid.
        w : :class:`pint.Quantity`
            Query wavelengths (used only for cache keying).
    
        Returns
        -------
        xr.Dataset
            Sparse dataset indexed by a flat ``index`` dimension with voxel
            coordinates ``i_x``, ``i_y``, ``i_z`` (render-grid, 0-based) and
            cloud microphysical properties.
    
        Raises
        ------
        ValueError
            If the profile x/y grid does not match the render grid, or if
            ``z_interp_method`` is not one of ``"nearest"`` or ``"linear"``.
        """
        z_levels_src  = to_quantity(self.profile.z_levels).m_as(ureg.meter)
        z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])
        z_centers_tgt = grid.layers.m_as(ureg.meter)
        n_z_src       = len(z_centers_src)
    
        profile_nx = len(self.profile.x_levels.values) - 1
        profile_ny = len(self.profile.y_levels.values) - 1
        if profile_nx != grid.n_cells_x or profile_ny != grid.n_cells_y:
            raise ValueError(
                f"Profile x/y grid ({profile_nx} x {profile_ny}) does not match "
                f"render grid ({grid.n_cells_x} x {grid.n_cells_y}). "
                "Resample the profile to the render grid first."
            )
    
        ix = self.profile.i_x.values
        iy = self.profile.i_y.values
        iz = self.profile.i_z.values
        n_x = grid.n_cells_x
        n_y = grid.n_cells_y
    
        r_eff_src        = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        v_eff_src        = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        mass_density_src = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        valid_src        = np.zeros((n_x, n_y, n_z_src), dtype=bool)
    
        valid_src[ix, iy, iz]        = True
        r_eff_src[ix, iy, iz]        = self.profile.r_eff.values
        v_eff_src[ix, iy, iz]        = self.profile.v_eff.values
        mass_density_prof            = to_quantity(self.profile.mass_density)
        mass_density_src[ix, iy, iz] = mass_density_prof.m
    
        # Bracket each target centre between two source centres.
        # il / iu are the lower / upper indices; in_range masks out targets
        # that fall outside the source z extent.
        il       = np.searchsorted(z_centers_src, z_centers_tgt, side="right") - 1
        iu       = il + 1
        in_range = (il >= 0) & (iu < n_z_src)
        il_safe  = np.clip(il, 0, n_z_src - 2)
        iu_safe  = il_safe + 1
        iz_src_tgt = None
    
        if self.z_interp_method == "nearest":
            # Pick the closer of the two bracketing centres.
            mid = 0.5 * (z_centers_src[il_safe] + z_centers_src[iu_safe])
            inn = np.where(z_centers_tgt < mid, il_safe, iu_safe)
    
            r_eff_tgt        = r_eff_src[:, :, inn]
            v_eff_tgt        = v_eff_src[:, :, inn]
            mass_density_tgt = mass_density_src[:, :, inn]
            valid_tgt        = valid_src[:, :, inn] & in_range
            iz_src_tgt       = inn
    
        elif self.z_interp_method == "linear":
            # Interpolation weight t ∈ [0, 1]:
            #   value(z_t) = (1 - t) * value[il] + t * value[iu]
            dz = z_centers_src[iu_safe] - z_centers_src[il_safe]
            t  = np.where(
                in_range,
                (z_centers_tgt - z_centers_src[il_safe]) / dz,
                0.0,
            ).astype(np.float32)
    
            r_eff_tgt        = (1.0 - t) * r_eff_src[:, :, il_safe]        + t * r_eff_src[:, :, iu_safe]
            v_eff_tgt        = (1.0 - t) * v_eff_src[:, :, il_safe]        + t * v_eff_src[:, :, iu_safe]
            mass_density_tgt = (1.0 - t) * mass_density_src[:, :, il_safe] + t * mass_density_src[:, :, iu_safe]
            valid_tgt        = valid_src[:, :, il_safe] & valid_src[:, :, iu_safe] & in_range
    
        else:
            raise ValueError(
                f"Unsupported z_interp_method {self.z_interp_method!r}: "
                "expected 'nearest' or 'linear'."
            )
    
        ix_tgt, iy_tgt, iz_tgt = np.where(valid_tgt)
    
        data_vars = dict(
            i_x         =(["index"], ix_tgt),
            i_y         =(["index"], iy_tgt),
            i_z         =(["index"], iz_tgt),
            r_eff       =(["index"], r_eff_tgt[ix_tgt, iy_tgt, iz_tgt]),
            v_eff       =(["index"], v_eff_tgt[ix_tgt, iy_tgt, iz_tgt]),
            mass_density=(
                ["index"],
                mass_density_tgt[ix_tgt, iy_tgt, iz_tgt],
                {"units": str(mass_density_prof.units)},
            ),
        )
        if iz_src_tgt is not None:
            data_vars["i_z_src"] = (["index"], iz_src_tgt[iz_tgt])
    
        return xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                z_levels=(["z"], grid.levels.m_as(ureg.kilometer)),
                x_levels=(["x"], grid.edges_x.m_as(ureg.kilometer)),
                y_levels=(["y"], grid.edges_y.m_as(ureg.kilometer)),
            ),
        )

    def _profile_regular_grid(self) -> PlaneParallelGridCoords:
        """
        Return a regular :class:`.PlaneParallelGridCoords` aligned with
        and fine enough to resolve the profile's z-levels.
        """

        def _regularize(arr: np.ndarray) -> np.ndarray:
            min_gap = np.diff(arr).min()
            n = int(round((arr[-1] - arr[0]) / min_gap)) * 2 + 1
            return np.linspace(arr[0], arr[-1], n)

        x_levels = self.profile.x_levels.values * ureg.km
        y_levels = self.profile.y_levels.values * ureg.km
        z_levels = _regularize(self.profile.z_levels.values) * ureg.km

        return PlaneParallelGridCoords(
            edges_x=x_levels,
            edges_y=y_levels,
            levels=z_levels,
        ).centered()

    def to_profile_regular_grid(self):
        """
        Return a copy of this :class:`ParticleField` with its geometry set to
        a regular grid aligned with the profile.
        """
        grid = self._profile_regular_grid()
        geometry = PlaneParallelGeometry(
            grid=grid,
            toa_altitude=grid.levels[-1],
            width=self.geometry.width,
        )
        return attrs.evolve(self, geometry=geometry)

    def to_particle_layer_from_profile_point(
        self,
        profile_point: xr.Dataset,
        w: ureg.Quantity,
        grid: GridCoords | None = None,
    ) -> "ParticleLayer":
        """
        Build a :class:`.ParticleLayer` from a single profile point.

        Parameters
        ----------
        profile_point : xr.Dataset
            A single-point subset of a resampled profile dataset, e.g.
            obtained via ``pfield._resample_profile_to_grid(...).sel(index=0)``.
            Must contain exactly one point along the ``index`` dimension.
        w : :class:`pint.Quantity`
            Evaluation wavelength.
        grid : :class:`.GridCoords`, optional
            Target render grid. Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`.ParticleLayer`
        """
        if profile_point.sizes.get("index", 1) != 1:
            raise ValueError(
                "profile_point must contain exactly one point along the 'index' dimension."
            )

        grid = grid or self.geometry.grid
        wavelength = np.atleast_1d(w)

        ix = int(profile_point.i_x.values.squeeze())
        iy = int(profile_point.i_y.values.squeeze())
        iz = int(profile_point.i_z.values.squeeze())
        r_eff_val = float(profile_point.r_eff.values.squeeze())
        v_eff_val = float(profile_point.v_eff.values.squeeze())
        mass_density_quantity = to_quantity(profile_point.mass_density).squeeze()

        v_eff_single = np.array([v_eff_val])
        single_voxel_properties = format_particles_dataset(
            self.properties, v_eff_single
        )
        interpolated_properties = interp_properties_wavelength(
            single_voxel_properties, wavelength
        )

        r_eff_pt = xr.DataArray([r_eff_val], dims="index")
        v_eff_pt = xr.DataArray([v_eff_val], dims="index")
        w_value = wavelength.m_as(interpolated_properties.w.attrs["units"])

        def _interp_da(da):
            saved_attrs = da.attrs
            interp_coords = {}
            if da.sizes["w"] > 1:
                interp_coords["w"] = w_value
            else:
                da = da.squeeze("w")
            if da.sizes["r_eff"] > 1:
                interp_coords["r_eff"] = r_eff_pt
            else:
                da = da.squeeze("r_eff")
            if da.sizes["v_eff"] > 1:
                interp_coords["v_eff"] = v_eff_pt
            else:
                da = da.squeeze("v_eff")
            if interp_coords:
                da = da.interp(**interp_coords, method="linear")
                da.attrs = saved_attrs
            return da

        m_extinction = to_quantity(
            _interp_da(interpolated_properties.m_extinction)
        ).squeeze()
        albedo = to_quantity(_interp_da(interpolated_properties.albedo)).squeeze()
        sigma_t = (m_extinction * mass_density_quantity).to("1/km")

        r_eff_index = int(
            np.argmin(np.abs(interpolated_properties.r_eff.values - r_eff_val))
        )
        start = int(interpolated_properties.start.values[0, r_eff_index, 0])
        n_pts = int(interpolated_properties.n_pts.values[0, r_eff_index, 0])
        theta = interpolated_properties.theta_native.values[start : start + n_pts]
        phase_vals = interpolated_properties.phase_native.values[
            start : start + n_pts, :
        ]

        mu = np.cos(np.deg2rad(theta))
        sort_idx = np.argsort(mu)
        mu = mu[sort_idx]
        phase_vals = phase_vals[sort_idx, :]

        phase_matrix = np.zeros((1, len(mu), 4, 4))
        for k in range(16):
            phase_matrix[0, :, k // 4, k % 4] = phase_vals[:, k]

        layer_height = to_quantity(
            xr.DataArray(np.diff(grid.levels.m_as(ureg.km)), attrs={"units": "km"})
        )[iz]

        # Keep the OT per covered XY area constant
        tau_voxel = (sigma_t * layer_height).m_as(ureg.dimensionless)
        area_profile = grid.cell_width * grid.cell_length
        area_layer = (grid.edges_x[-1] - grid.edges_x[0]) * (
            grid.edges_y[-1] - grid.edges_y[0]
        )
        tau_ref = float(
            tau_voxel * (area_profile / area_layer).m_as(ureg.dimensionless)
        )
        tgt_edges_x = grid.edges_x.m_as(ureg.km)
        tgt_edges_y = grid.edges_y.m_as(ureg.km)
        src_x0, src_x1 = float(tgt_edges_x[ix]), float(tgt_edges_x[ix + 1])
        src_y0, src_y1 = float(tgt_edges_y[iy]), float(tgt_edges_y[iy + 1])

        overlap = _compute_overlap(
            src_x0, src_x1, src_y0, src_y1, tgt_edges_x, tgt_edges_y
        )
        layer_height = to_quantity(
            xr.DataArray(np.diff(grid.levels.m_as(ureg.km)), attrs={"units": "km"})
        )[iz]
        tau_ref = ureg.Quantity(
            (sigma_t * layer_height).m_as(ureg.dimensionless) * overlap,
            "dimensionless",
        )

        phase_ds = xr.Dataset(
            {
                "sigma_t": xr.DataArray(
                    sigma_t.m.reshape(1),
                    dims=["w"],
                    attrs={"units": "1/km"},
                ),
                "albedo": xr.DataArray(
                    albedo.m.reshape(1),
                    dims=["w"],
                    attrs={"units": "dimensionless"},
                ),
                "phase": xr.DataArray(
                    phase_matrix,
                    dims=["w", "mu", "i", "j"],
                    attrs={"units": "sr^-1"},
                ),
            },
            coords={
                "w": xr.DataArray(
                    wavelength.m_as(interpolated_properties.w.attrs["units"]).reshape(
                        1
                    ),
                    dims=["w"],
                    attrs={"units": interpolated_properties.w.attrs["units"]},
                ),
                "mu": xr.DataArray(mu, dims=["mu"], attrs={"units": "dimensionless"}),
                "i": xr.DataArray(np.arange(4), dims=["i"]),
                "j": xr.DataArray(np.arange(4), dims=["j"]),
            },
        )

        return ParticleLayer(
            bottom=grid.levels[iz],
            top=grid.levels[iz + 1],
            distribution="uniform",
            w_ref=wavelength[0],
            tau_ref=tau_ref,
            dataset=phase_ds,
            geometry=self.geometry,
        )

    def to_particle_layer(
        self,
        w: ureg.Quantity,
        ix: int,
        iy: int,
        iz: int,
        grid: GridCoords | None = None,
    ) -> "ParticleLayer":
        """
        Build a :class:`.ParticleLayer` representing a single voxel of the
        cloud field at the given grid location.

        Parameters
        ----------
        w : :class:`pint.Quantity`
            Evaluation wavelength.
        ix, iy, iz : int
            Voxel indices in the render grid.
        grid : :class:`.GridCoords`, optional
            Target render grid. Defaults to ``self.geometry.grid``.

        Returns
        -------
        :class:`.ParticleLayer`
        """
        grid = grid or self.geometry.grid
        wavelength = np.atleast_1d(w)

        resampled_profile = self._resample_profile_to_grid(grid, wavelength)

        voxel_mask = (
            (resampled_profile.i_x.values == ix)
            & (resampled_profile.i_y.values == iy)
            & (resampled_profile.i_z.values == iz)
        )
        if not np.any(voxel_mask):
            raise ValueError(f"No cloud voxel found at ({ix}, {iy}, {iz}).")

        index = int(np.where(voxel_mask)[0][0])
        profile_point = resampled_profile.isel(index=index)

        return self.to_particle_layer_from_profile_point(profile_point, w, grid)

    @classmethod
    def from_particle_layer(
        cls,
        particle_layer: "ParticleLayer",
        mass_density: ureg.Quantity,
        r_eff: ureg.Quantity,
        v_eff: float,
        grid: GridCoords | None = None,
    ) -> "ParticleField":
        """
        Build a :class:`.ParticleField` from a :class:`.ParticleLayer`.

        Parameters
        ----------
        particle_layer : :class:`.ParticleLayer`
            Source particle layer.
        mass_density : :class:`pint.Quantity`
            Mass density used to derive the mass extinction coefficient.
        r_eff : :class:`pint.Quantity`
            Effective radius of the particle species.
        v_eff : float
            Effective variance of the particle species.
        grid : :class:`.GridCoords`, optional
            Target render grid for the output :class:`.ParticleField`.
            If provided, the :class:`.ParticleLayer` is re-evaluated on it.

        Returns
        -------
        :class:`.ParticleField`
        """
        if grid is not None:
            geometry = attrs.evolve(particle_layer.geometry, grid=grid)
            particle_layer = attrs.evolve(particle_layer, geometry=geometry)

        target_grid = particle_layer.geometry.grid
        w_ref = particle_layer.w_ref
        ds = particle_layer.dataset
        w_unit = ds.w.attrs["units"]
        w_value = w_ref.m_as(w_unit)

        if ds.sizes["w"] > 1:
            sigma_t_at_wref = to_quantity(ds.sigma_t.interp(w=float(w_value)))
            albedo_at_wref = to_quantity(ds.albedo.interp(w=float(w_value)))
        else:
            sigma_t_at_wref = to_quantity(ds.sigma_t.squeeze("w"))
            albedo_at_wref = to_quantity(ds.albedo.squeeze("w"))

        m_extinction = (sigma_t_at_wref / mass_density).to(
            str(sigma_t_at_wref.units / mass_density.units)
        )

        si_ref = MonoSpectralIndex(w=w_ref)
        sigma_t_grid = particle_layer.eval_sigma_t(si_ref, target_grid)
        mass_density_grid = (sigma_t_grid / m_extinction).to(str(mass_density.units))

        valid_mask = mass_density_grid.m > 0
        ix_arr, iy_arr, iz_arr = np.where(valid_mask)

        level_units = dict(units="kilometer")
        profile_ds = xr.Dataset(
            data_vars=dict(
                i_x=(["index"], ix_arr),
                i_y=(["index"], iy_arr),
                i_z=(["index"], iz_arr),
                r_eff=(
                    ["index"],
                    np.full(len(ix_arr), r_eff.m_as("micron")),
                    {"units": "micron"},
                ),
                v_eff=(
                    ["index"],
                    np.full(len(ix_arr), float(v_eff)),
                    {"units": "dimensionless"},
                ),
                mass_density=(
                    ["index"],
                    mass_density_grid.m[ix_arr, iy_arr, iz_arr],
                    {"units": str(mass_density_grid.units)},
                ),
            ),
            coords=dict(
                x_levels=(["x"], target_grid.edges_x.m_as(ureg.kilometer), level_units),
                y_levels=(["y"], target_grid.edges_y.m_as(ureg.kilometer), level_units),
                z_levels=(["z"], target_grid.levels.m_as(ureg.kilometer), level_units),
            ),
        )

        mu = ds.mu.values
        theta = np.rad2deg(np.pi - np.arccos(mu))
        sort_idx = np.argsort(theta)
        theta_sorted = theta[sort_idx]
        phase_full = ds.phase.values[0, sort_idx, :, :]  # (n_theta, 4, 4)
        n_theta = len(theta_sorted)

        phase_4comp = np.stack(
            [
                phase_full[:, 0, 0],
                phase_full[:, 0, 1],
                phase_full[:, 2, 2],
                phase_full[:, 2, 3],
            ],
            axis=0,
        ).reshape(1, 1, 4, n_theta)

        theta_padded = np.full((1, 1, 4, n_theta), np.nan, dtype=np.float32)
        theta_padded[0, 0, :, :] = theta_sorted[np.newaxis, :]

        alpha = 1.0 / v_eff - 2.0

        properties_ds = xr.Dataset(
            {
                "ext": xr.DataArray(
                    m_extinction.m.reshape(1, 1),
                    dims=["nlam", "nreff"],
                    attrs={"units": str(m_extinction.units)},
                ),
                "ssa": xr.DataArray(
                    albedo_at_wref.m.reshape(1, 1),
                    dims=["nlam", "nreff"],
                    attrs={"units": "dimensionless"},
                ),
                "theta": xr.DataArray(
                    theta_padded,
                    dims=["nlam", "nreff", "nphamat", "nthetamax"],
                    attrs=dict(units="degree"),
                ),
                "phase": xr.DataArray(
                    phase_4comp.astype(np.float32),
                    dims=["nlam", "nreff", "nphamat", "nthetamax"],
                ),
                "ntheta": xr.DataArray(
                    np.full((1, 1, 4), n_theta, dtype=np.int32),
                    dims=["nlam", "nreff", "nphamat"],
                ),
                "wavelen": xr.DataArray(
                    np.array([w_value], dtype=np.float64),
                    dims=["nlam"],
                    attrs=dict(units=w_unit),
                ),
                "reff": xr.DataArray(
                    np.array([r_eff.m_as("micron")], dtype=np.float64),
                    dims=["nreff"],
                    attrs=dict(units="micron"),
                ),
                "rho": xr.DataArray(
                    np.asarray(mass_density.m_as("g/cm^3"), dtype=np.float64).reshape(
                        1, 1
                    ),
                    dims=["nlam", "nreff"],
                    attrs=dict(units="g/cm^3"),
                ),
            },
            attrs={"param_alpha": alpha},
        )

        return cls(
            profile=profile_ds,
            properties=properties_ds,
            geometry=particle_layer.geometry,
            has_absorption=particle_layer.has_absorption,
            has_scattering=particle_layer.has_scattering,
            z_interp_method="nearest",
        )
