from __future__ import annotations

import warnings

import attrs
import numpy as np
import xarray as xr

from eradiate import traverse
from eradiate.attrs import define
from eradiate.grid import GridCoords, PlaneParallelGridCoords
from eradiate.kernel import SceneParameter
from eradiate.scenes.atmosphere import AtmosphericMedium
from eradiate.scenes.geometry import (
    PlaneParallelGeometry,
)
from eradiate.scenes.phase import (
    CloudPhaseFunction,
    interpolate_cloudparticles_profile,
)
from eradiate.spectral.index import (
    SpectralIndex,
)
from eradiate.units import unit_registry as ureg
from eradiate.util.misc import cache_by_id

# TBD:
# has_absorption/has_scattering
# proper documented fields
# check grid and profile compatibility
# pint and datasets integration
# document: profile may have irregular zgrid, but shall have regular x and y grids


@define(eq=False, slots=False)
class CloudField(AtmosphericMedium):
    profile = attrs.field(kw_only=True)
    properties = attrs.field(kw_only=True)

    has_absorption = attrs.field(default=True)
    has_scattering = attrs.field(default=True)

    particles_interp_method = attrs.field(default="bilinear")

    def eval_sigma_t(self, si: SpectralIndex, grid: GridCoords | None = None):
        grid = grid or self.geometry.grid

        profile = self._resample_profile_to_grid(grid)

        ix = profile.i_x.values - 1
        iy = profile.i_y.values - 1
        iz = profile.i_z.values - 1
        ext = profile.extinction.values

        sigma_t = np.zeros(
            (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float32
        )
        sigma_t[ix, iy, iz] = ext[np.newaxis, :]
        sigma_t = sigma_t * ureg.Unit("1/km")

        if self.has_absorption and self.has_scattering:
            return sigma_t

        if self.has_scattering:
            return sigma_t - self.eval_sigma_a(si, grid)

        if self.has_absorption:
            raise ValueError
            return sigma_t - self.eval_sigma_s(si, grid)

        raise RuntimeError()

    def eval_sigma_s(self, si: SpectralIndex, grid: GridCoords | None = None):
        grid = grid or self.geometry.grid

        if not self.has_scattering:
            return np.zeros(grid.shape)

        albedo = self.eval_albedo(si, grid)
        sigma_t = self.eval_sigma_t(si, grid)

        return sigma_t * albedo.m_as(ureg.dimensionless)

    def eval_sigma_a(self, si: SpectralIndex, grid: GridCoords | None = None):
        grid = grid or self.geometry.grid

        if not self.has_absorption:
            return np.zeros(grid.shape)

        albedo = self.eval_albedo(si, grid)
        sigma_t = self.eval_sigma_t(si, grid)

        return sigma_t * (1.0 - albedo.m_as(ureg.dimensionless))

    def eval_mfp(self, si: SpectralIndex, grid: GridCoords | None = None):
        min_sigma_s = self.eval_sigma_s(ctx.si).min(axis=-1)
        out = np.full(min_sigma_s.shape, np.inf)
        np.divide(
            np.ones(min_sigma_s.shape[:-1]),
            min_sigma_s,
            where=min_sigma_s != 0,
            out=out,
        )
        return out * ureg.m

    def eval_albedo(self, si: SpectralIndex, grid: GridCoords | None = None):
        grid = grid or self.geometry.grid

        if self.has_absorption and self.has_scattering:
            wavelengths = np.atleast_1d(si.w)

            profile = self._resample_profile_to_grid(grid)
            properties = self._interpolate_profile(wavelengths, grid, profile)

            ix = profile.i_x.values - 1
            iy = profile.i_y.values - 1
            iz = profile.i_z.values - 1
            alb = properties.albedo.values.squeeze()
            albedo = np.zeros(
                (grid.n_cells_x, grid.n_cells_y, grid.n_cells_z), dtype=np.float32
            )
            albedo[ix, iy, iz] = alb[np.newaxis, :]

            return albedo * ureg.dimensionless

        if self.has_absorption:
            return 0.0 * ureg.dimensionless

        if self.has_scattering:
            return 1.0 * ureg.dimensionless

        raise RuntimeError()

    @property
    def _template_phase(self):
        return traverse(self.phase)[0].data

    @property
    def _params_phase(self) -> dict[std, SceneParameter]:
        return traverse(self.phase)[1].data

    @cache_by_id
    def _resample_profile_to_grid(self, grid, method="nearest"):
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

        il = np.clip(np.searchsorted(z_centers_src, z_centers_tgt) - 1, 0, n_z_src - 2)
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

    def _profile_regular_grid(self):
        def regularize_sorted_array(arr):
            min_gap = np.diff(arr).min()
            n_samples = int(round((arr[-1] - arr[0]) / min_gap)) * 2 + 1
            resampled = np.linspace(arr[0], arr[-1], n_samples)
            return resampled

        x_levels = self.profile.x_levels.values * ureg.km
        y_levels = self.profile.y_levels.values * ureg.km
        z_levels = regularize_sorted_array(self.profile.z_levels.values) * ureg.km

        grid = PlaneParallelGridCoords(
            edges_x=x_levels,
            edges_y=y_levels,
            levels=z_levels,
        ).centered()

        return grid

    def to_profile_regular_grid(self) -> CloudField:
        grid = self._profile_regular_grid()
        geometry = PlaneParallelGeometry(
            grid=grid,
            toa_altitude=grid.levels[-1],
            width=self.geometry.width,
        )

        return attrs.evolve(self, geometry=geometry)

    @cache_by_id
    def _interpolate_profile(self, wavelengths, grid, profile):
        profile = self._resample_profile_to_grid(grid)

        properties = interpolate_cloudparticles_profile(
            self.properties,
            profile,
            wavelengths,
            rv_mode=self.particles_interp_method,
        )

        return properties

    def _check_grid_profile_compatibility(self, grid):
        z_levels_src = self.profile.z_levels.values * 1e3
        z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])
        z_centers_tgt = grid.layers.m_as(ureg.meter)

        n_x = grid.n_cells_x
        n_y = grid.n_cells_y
        n_z_src = len(z_centers_src)

        ix = self.profile.i_x.values - 1
        iy = self.profile.i_y.values - 1
        iz = self.profile.i_z.values - 1

        # target z-range must be contained within source z-range
        if z_centers_tgt[0] < z_centers_src[0] or z_centers_tgt[-1] > z_centers_src[-1]:
            raise ValueError(
                f"Target grid z-range [{z_centers_tgt[0]:.2f}, {z_centers_tgt[-1]:.2f}] m "
                f"extends beyond source z-range [{z_centers_src[0]:.2f}, {z_centers_src[-1]:.2f}] m."
            )

        # x/y grid must be compatible
        if grid.n_cells_x != n_x or grid.n_cells_y != n_y:
            raise ValueError(
                f"Target grid x/y shape ({grid.n_cells_x}, {grid.n_cells_y}) "
                f"is incompatible with profile x/y shape ({n_x}, {n_y})."
            )

        # warn if target z-resolution is coarser than source
        tgt_min_gap = np.diff(z_centers_tgt).min()
        src_min_gap = np.diff(z_centers_src).min()
        if tgt_min_gap > 2 * src_min_gap:
            warnings.warn(
                f"Target z-resolution ({tgt_min_gap:.2f} m) is coarser than twice the "
                f"source minimum gap ({src_min_gap:.2f} m). Cloud features may be lost."
            )

        # profile indices must be within grid bounds
        if ix.max() >= n_x or iy.max() >= n_y or iz.max() >= n_z_src:
            raise ValueError(
                "Profile contains voxel indices out of bounds for the given grid."
            )

    @cache_by_id
    def _eval_cloud_phase_data(self, si):
        w = si.w

        grid = self.geometry.grid
        profile = self._resample_profile_to_grid(grid)
        interp_ds = self._interpolate_profile(w, grid, profile)

        ix = profile.i_x.values - 1
        iy = profile.i_y.values - 1
        iz = profile.i_z.values - 1
        index_volume = np.full(grid.shape, -1, dtype=np.int32)
        for i in range(len(interp_ds.grid_start)):
            index_volume[ix[i], iy[i], iz[i]] = i

        return interp_ds, index_volume

    @property
    def phase(self):
        grid = self.geometry.grid
        return CloudPhaseFunction(
            grid=grid,
            arrays=lambda ctx: self._eval_cloud_phase_data(ctx.si)[0],
            index=lambda ctx: self._eval_cloud_phase_data(ctx.si)[1],
        )
