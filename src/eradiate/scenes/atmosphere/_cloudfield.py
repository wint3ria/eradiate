from __future__ import annotations

import warnings
from typing import Literal

import attrs
import numpy as np
import xarray as xr

from eradiate import traverse

from ...attrs import define, documented
from ...grid import GridCoords, PlaneParallelGridCoords
from ...kernel import SceneParameter
from ...util.misc import cache_by_id
from ...scenes.atmosphere import AtmosphericMedium
from ...scenes.geometry import PlaneParallelGeometry
from ...scenes.phase import CloudPhaseFunction, interpolate_cloudparticles_profile
from ...spectral.index import SpectralIndex
from ...units import unit_registry as ureg


def _validate_particles_interp_method(instance, attribute, value):
    if value not in ("linear", "nearest"):
        raise NotImplementedError(
            f"Interpolation method '{value}' is not supported. "
            "Use 'linear' or 'nearest'."
        )


@define(eq=False, slots=False)
class CloudField(AtmosphericMedium):
    """
    Atmospheric medium representing a cloud field with spatially heterogeneous
    optical properties.

    The cloud field is defined by a sparse volumetric *profile* (voxel-indexed
    microphysical properties) and a pre-computed *properties* lookup dataset
    (IPRT standard Mie particles property dataset as returned by
    :func:`.format_cloudparticles_dataset`).  At render time the profile is
    resampled onto the render grid, the per-voxel phase function is
    interpolated from the lookup table, and the result is handed to a
    :class:`.CloudPhaseFunction` kernel plugin.

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
        "variables ``r_eff``, ``v_eff``, and ``extinction`` indexed by a "
        "flat ``index`` dimension, together with integer voxel coordinates "
        "``i_x``, ``i_y``, and ``i_z`` (1-based) and level-edge coordinates "
        "``x_levels``, ``y_levels``, and ``z_levels``. "
        "This parameter has no default.",
        type="xr.Dataset",
    )

    properties: xr.Dataset = documented(
        attrs.field(kw_only=True),
        doc="IPRT standard Mie particles property dataset as returned by "
        ":func:`.format_cloudparticles_dataset`.  Contains per-wavelength "
        "extinction coefficients and phase function data used as the "
        "lookup table for per-voxel optical property interpolation. "
        "This parameter has no default.",
        type="xr.Dataset",
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

    particles_interp_method: Literal["linear", "nearest"] = documented(
        attrs.field(
            default="linear",
            kw_only=True,
            validator=_validate_particles_interp_method,
        ),
        doc="Interpolation strategy for the particle property lookup. "
        "``\"linear\"`` uses trilinear interpolation in the "
        "``(w, r_eff, v_eff)`` parameter space; ``\"nearest\"`` "
        "uses nearest-neighbour lookup in ``(r_eff, v_eff)`` with "
        "linear interpolation in wavelength.",
        type="str",
        init_type='{"linear", "nearest"}',
        default='"linear"',
    )

    def eval_sigma_t(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> np.ndarray:
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
        ndarray
            Extinction coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, in units of
            :math:`\\mathrm{km}^{-1}`.

        Raises
        ------
        ValueError
            If neither absorption nor scattering is enabled.
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        grid = grid or self.geometry.grid
        profile = self._resample_profile_to_grid(grid)

        ix = profile.i_x.values - 1
        iy = profile.i_y.values - 1
        iz = profile.i_z.values - 1

        sigma_t = np.zeros(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float32
        )
        sigma_t[ix, iy, iz] = profile.extinction.values[np.newaxis, :]
        sigma_t = sigma_t * ureg.Unit("1/km")

        if self.has_absorption and self.has_scattering:
            return sigma_t

        if self.has_scattering:
            return sigma_t - self.eval_sigma_a(si, grid)

        if self.has_absorption:
            return sigma_t - self.eval_sigma_s(si, grid)

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> np.ndarray:
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
        ndarray
            Scattering coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``.
        """
        grid = grid or self.geometry.grid

        if not self.has_scattering:
            return np.zeros(grid.shape)

        return self.eval_sigma_t(si, grid) * self.eval_albedo(si, grid).m_as(
            ureg.dimensionless
        )

    def eval_sigma_a(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> np.ndarray:
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
        ndarray
            Absorption coefficient array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``.
        """
        grid = grid or self.geometry.grid

        if not self.has_absorption:
            return np.zeros(grid.shape)

        return self.eval_sigma_t(si, grid) * (
            1.0 - self.eval_albedo(si, grid).m_as(ureg.dimensionless)
        )

    def eval_mfp(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> np.ndarray:
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
        ndarray
            Mean free path in metres.
        """
        grid = grid or self.geometry.grid
        sigma_s = self.eval_sigma_s(si, grid)
        out = np.full(sigma_s.shape, np.inf)
        np.divide(1.0, sigma_s, where=sigma_s != 0, out=out)
        return out * ureg.m

    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> np.ndarray:
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
        ndarray
            Single-scattering albedo array of shape
            ``(n_cells_x, n_cells_y, n_cells_z)``, dimensionless.

        Raises
        ------
        RuntimeError
            If both :attr:`has_absorption` and :attr:`has_scattering` are
            ``False``.
        """
        grid = grid or self.geometry.grid

        if self.has_absorption and self.has_scattering:
            wavelengths = np.atleast_1d(si.w)
            profile = self._resample_profile_to_grid(grid)
            properties = self._interpolate_profile(wavelengths, grid, profile)

            ix = profile.i_x.values - 1
            iy = profile.i_y.values - 1
            iz = profile.i_z.values - 1

            albedo = np.zeros(
                (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float32
            )
            albedo[ix, iy, iz] = properties.albedo.values.squeeze()[np.newaxis, :]
            return albedo * ureg.dimensionless

        if self.has_absorption:
            return 0.0 * ureg.dimensionless

        if self.has_scattering:
            return 1.0 * ureg.dimensionless

        raise RuntimeError(
            "At least one of 'has_absorption' or 'has_scattering' must be True."
        )

    @property
    def _template_phase(self) -> dict:
        return traverse(self.phase)[0].data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        return traverse(self.phase)[1].data

    @cache_by_id
    def _resample_profile_to_grid(
        self,
        grid: GridCoords,
        method: Literal["nearest", "linear"] = "nearest",
    ) -> xr.Dataset:
        """
        Resample the sparse cloud profile onto the cells of *grid*.

        Parameters
        ----------
        grid : :class:`.GridCoords`
            Target render grid.
        method : {"nearest", "linear"}, optional
            Resampling strategy along the z-axis.

        Returns
        -------
        xr.Dataset
            Sparse resampled dataset indexed by a flat ``index`` dimension
            with 1-based voxel coordinates and cloud microphysical
            properties.
        """
        z_levels_src = self.profile.z_levels.values * 1e3
        z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])
        z_centers_tgt = grid.layers.m_as(ureg.meter)
        n_z_src = len(z_centers_src)

        ix = self.profile.i_x.values - 1
        iy = self.profile.i_y.values - 1
        iz = self.profile.i_z.values - 1
        n_x = grid.n_cells_x
        n_y = grid.n_cells_y

        r_eff_src = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        v_eff_src = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        ext_src = np.zeros((n_x, n_y, n_z_src), dtype=np.float32)
        valid_src = np.zeros((n_x, n_y, n_z_src), dtype=bool)
        r_eff_src[ix, iy, iz] = self.profile.r_eff.values
        v_eff_src[ix, iy, iz] = self.profile.v_eff.values
        ext_src[ix, iy, iz] = self.profile.extinction.values
        valid_src[ix, iy, iz] = True

        il = np.clip(
            np.searchsorted(z_centers_src, z_centers_tgt) - 1, 0, n_z_src - 2
        )
        iu = il + 1

        if method == "nearest":
            mid = 0.5 * (z_centers_src[il] + z_centers_src[iu])
            inn = np.where(z_centers_tgt < mid, il, iu)
            _, unique_iz = np.unique(inn, return_index=True)
            mask = np.isin(np.arange(len(inn)), unique_iz)
            r_eff_tgt = r_eff_src[:, :, inn]
            v_eff_tgt = v_eff_src[:, :, inn]
            ext_tgt = ext_src[:, :, inn]
            valid_tgt = valid_src[:, :, inn] & mask
        else:
            dz = z_centers_src[iu] - z_centers_src[il]
            t = (z_centers_tgt - z_centers_src[il]) / np.where(dz == 0, 1.0, dz)
            r_eff_tgt = r_eff_src[:, :, il] * (1 - t) + r_eff_src[:, :, iu] * t
            v_eff_tgt = v_eff_src[:, :, il] * (1 - t) + v_eff_src[:, :, iu] * t
            ext_tgt = ext_src[:, :, il] * (1 - t) + ext_src[:, :, iu] * t
            valid_tgt = valid_src[:, :, il] | valid_src[:, :, iu]

        ix_tgt, iy_tgt, iz_tgt = np.where(valid_tgt)
        return xr.Dataset(
            data_vars=dict(
                i_x=(["index"], ix_tgt + 1),
                i_y=(["index"], iy_tgt + 1),
                i_z=(["index"], iz_tgt + 1),
                extinction=(["index"], ext_tgt[ix_tgt, iy_tgt, iz_tgt]),
                r_eff=(["index"], r_eff_tgt[ix_tgt, iy_tgt, iz_tgt]),
                v_eff=(["index"], v_eff_tgt[ix_tgt, iy_tgt, iz_tgt]),
            ),
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

    def to_profile_regular_grid(self) -> CloudField:
        """
        Return a copy of this :class:`CloudField` with its geometry set to
        a regular grid aligned with the profile.
        """
        grid = self._profile_regular_grid()
        geometry = PlaneParallelGeometry(
            grid=grid,
            toa_altitude=grid.levels[-1],
            width=self.geometry.width,
        )
        return attrs.evolve(self, geometry=geometry)

    @cache_by_id
    def _interpolate_profile(
        self,
        wavelengths: np.ndarray,
        grid: GridCoords,
        profile: xr.Dataset,
    ) -> xr.Dataset:
        """
        Interpolate the cloud-particle properties dataset onto the resampled
        profile entries at the given wavelengths.

        This method is a thin wrapper around
        :func:`.interpolate_cloudparticles_profile` that feeds the resampled
        *profile* (as returned by :meth:`_resample_profile_to_grid`) directly
        and passes :attr:`particles_interp_method` as the ``rv_mode``
        argument.  The result is memoised by argument identity via
        :func:`.cache_by_id`.

        Parameters
        ----------
        wavelengths : ndarray
            Query wavelengths as a :class:`pint.Quantity`.
        grid : :class:`.GridCoords`
            Target render grid (used only for cache keying).
        profile : xr.Dataset
            Resampled profile as returned by :meth:`_resample_profile_to_grid`.

        Returns
        -------
        xr.Dataset
            Interpolated cloud-particle properties dataset in the layout
            produced by :func:`.interpolate_cloudparticles_profile`.
        """
        rv_mode_map = {"linear": "trilinear", "nearest": "nearest_rv_linear_w"}
        return interpolate_cloudparticles_profile(
            self.properties,
            profile,
            wavelengths,
            rv_mode=rv_mode_map[self.particles_interp_method],
        )

    def _check_grid_profile_compatibility(self, grid: GridCoords) -> None:
        z_levels_src = self.profile.z_levels.values * 1e3
        z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])
        z_centers_tgt = grid.layers.m_as(ureg.meter)

        n_x = grid.n_cells_x
        n_y = grid.n_cells_y
        n_z_src = len(z_centers_src)

        ix = self.profile.i_x.values - 1
        iy = self.profile.i_y.values - 1
        iz = self.profile.i_z.values - 1

        if z_centers_tgt[0] < z_centers_src[0] or z_centers_tgt[-1] > z_centers_src[-1]:
            raise ValueError(
                f"Target grid z-range [{z_centers_tgt[0]:.2f}, {z_centers_tgt[-1]:.2f}] m "
                f"extends beyond source z-range "
                f"[{z_centers_src[0]:.2f}, {z_centers_src[-1]:.2f}] m."
            )

        if grid.n_cells_x != n_x or grid.n_cells_y != n_y:
            raise ValueError(
                f"Target grid x/y shape ({grid.n_cells_x}, {grid.n_cells_y}) "
                f"is incompatible with profile x/y shape ({n_x}, {n_y})."
            )

        tgt_min_gap = np.diff(z_centers_tgt).min()
        src_min_gap = np.diff(z_centers_src).min()
        if tgt_min_gap > 2 * src_min_gap:
            warnings.warn(
                f"Target z-resolution ({tgt_min_gap:.2f} m) is coarser than "
                f"twice the source minimum gap ({src_min_gap:.2f} m). "
                "Cloud features may be lost.",
                stacklevel=2,
            )

        if ix.max() >= n_x or iy.max() >= n_y or iz.max() >= n_z_src:
            raise ValueError(
                "Profile contains voxel indices out of bounds for the given grid."
            )

    @cache_by_id
    def _eval_cloud_phase_data(
        self, si: SpectralIndex
    ) -> tuple[xr.Dataset, np.ndarray]:
        """
        Return the interpolated cloud-particle properties dataset and the
        spatial index volume for *si*.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index at which to evaluate.

        Returns
        -------
        interp_ds : xr.Dataset
            Interpolated cloud-particle properties dataset.
        index_volume : ndarray of int32, shape (n_cells_x, n_cells_y, n_cells_z)
            Per-voxel index into *interp_ds*, or ``-1`` for empty voxels.
        """
        grid = self.geometry.grid
        profile = self._resample_profile_to_grid(grid)
        interp_ds = self._interpolate_profile(np.atleast_1d(si.w), grid, profile)

        ix = profile.i_x.values - 1
        iy = profile.i_y.values - 1
        iz = profile.i_z.values - 1
        index_volume = np.full(grid.shape, -1, dtype=np.int32)
        index_volume[ix, iy, iz] = np.arange(len(ix), dtype=np.int32)

        return interp_ds, index_volume

    @property
    def phase(self) -> CloudPhaseFunction:
        return CloudPhaseFunction(
            grid=self.geometry.grid,
            interpolated_cloudproperties=lambda ctx: self._eval_cloud_phase_data(ctx.si)[0],
            spatial_index=lambda ctx: self._eval_cloud_phase_data(ctx.si)[1],
        )
