"""
Particle profiles.

Particle profiles describe the spatial distribution of a particle field:
the local particle size distribution parameters and mass concentration at
every voxel of a regular grid. Voxels with no particles carry a mass
concentration of zero.
"""

from __future__ import annotations

from functools import singledispatchmethod
from os import PathLike
from typing import Any

import attrs
import numpy as np
import xarray as xr

from ...attrs import define, documented
from ...converters import load_dataset
from ...grid import SpatialGrid, CartesianGrid, SphericalGrid
from ...units import to_quantity
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id


def _clip_and_interp(
    property_grid: xr.DataArray, reff_pts: xr.DataArray, veff_pts: xr.DataArray
) -> xr.DataArray:
    """Interpolate a ``(reff, veff)``-indexed property grid at *reff_pts*/*veff_pts*, clipped to its own coordinate range."""
    interp_coords = {}

    if property_grid.sizes.get("reff", 1) > 1:
        reff_vals = property_grid["reff"].values
        interp_coords["reff"] = reff_pts.clip(reff_vals.min(), reff_vals.max())
    elif "reff" in property_grid.dims:
        property_grid = property_grid.squeeze("reff")

    if property_grid.sizes.get("veff", 1) > 1:
        veff_vals = property_grid["veff"].values
        interp_coords["veff"] = veff_pts.clip(veff_vals.min(), veff_vals.max())
    elif "veff" in property_grid.dims:
        property_grid = property_grid.squeeze("veff")

    if interp_coords:
        property_grid = property_grid.interp(interp_coords, method="linear")

    return property_grid


@define
class ParticleProfile:
    """
    Dense, per-voxel description of a spatially varying particle field, in
    the :ref:`Pfd-Core v1 <sec-data-formats-pfd_core_v1>` format.
    """

    data: xr.Dataset = documented(
        attrs.field(validator=attrs.validators.instance_of(xr.Dataset)),
        doc="Particle profile dataset, in Pfd-Core v1 format.",
        type="xarray.Dataset",
        init_type="xarray.Dataset",
    )

    @classmethod
    def convert(cls, value: Any) -> ParticleProfile:
        """
        Conversion logic:

        * If value is a ParticleProfile instance, return it.
        * If value is a Dataset, initialize a ParticleProfile with it.
        * If value is a path-like, load the dataset and initialize a
          ParticleProfile with it.
        """
        if isinstance(value, ParticleProfile):
            return value

        if isinstance(value, xr.Dataset):
            return ParticleProfile(value)

        if isinstance(value, (str, PathLike)):
            return ParticleProfile(load_dataset(value))

        raise TypeError(
            f"could not convert value of type {type(value).__name__} to ParticleProfile"
        )

    @data.validator
    def _validate_data(self, attribute, value):
        missing = [
            name
            for name in (
                "reff",
                "veff",
                "mass_concentration",
                "x_levels",
                "y_levels",
                "z_levels",
            )
            if name not in value
        ]
        if missing:
            raise ValueError(
                f"Particle profile dataset is missing required variable(s): {missing}"
            )

    @staticmethod
    def _check_regular(name: str, values: np.ndarray) -> None:
        """Raise if ``values`` isn't evenly spaced."""
        step = np.diff(values)
        if not np.allclose(step, step[0]):
            raise ValueError(f"'{name}' must be regularly spaced")

    @singledispatchmethod
    def resample_regular(self, grid: SpatialGrid) -> ParticleProfile:
        """
        Resample the profile onto a regular grid.

        The profile's own ``x_levels``/``y_levels`` must already be regularly
        spaced and must match ``grid``'s extent and resolution exactly. ``z``
        is resampled onto ``grid``'s layer centres by nearest-neighbour
        lookup.

        Parameters
        ----------
        grid : .SpatialGrid
            Target regular grid.

        Returns
        -------
        ParticleProfile
            A new profile, regular on all three axes.

        Raises
        ------
        ValueError
            If the profile's ``x_levels``/``y_levels`` are not regularly
            spaced, or do not match ``grid``'s extent and resolution.
        """
        raise NotImplementedError(
            f"resample_regular() is not implemented for grid type {type(grid).__name__}"
        )

    @resample_regular.register(CartesianGrid)
    def _(self, grid) -> ParticleProfile:
        x_levels = to_quantity(self.data["x_levels"]).m_as(grid.edges_x.units)
        y_levels = to_quantity(self.data["y_levels"]).m_as(grid.edges_y.units)
        self._check_regular("x_levels", x_levels)
        self._check_regular("y_levels", y_levels)

        if (
            x_levels.shape != grid.edges_x.shape
            or y_levels.shape != grid.edges_y.shape
            or not np.allclose(x_levels, grid.edges_x.m)
            or not np.allclose(y_levels, grid.edges_y.m)
        ):
            raise ValueError(
                "Profile 'x_levels'/'y_levels' do not match the target "
                "grid's extent and resolution."
            )

        return self._resample_z(grid, grid.edges_x, grid.edges_y)

    @resample_regular.register(SphericalGrid)
    def _(self, grid) -> ParticleProfile:
        x_levels = to_quantity(self.data["x_levels"]).m_as(grid.azimuths.units)
        y_levels = to_quantity(self.data["y_levels"]).m_as(grid.colatitudes.units)
        self._check_regular("x_levels", x_levels)
        self._check_regular("y_levels", y_levels)

        if (
            x_levels.shape != grid.azimuths.shape
            or y_levels.shape != grid.colatitudes.shape
            or not np.allclose(x_levels, grid.azimuths.m)
            or not np.allclose(y_levels, grid.colatitudes.m)
        ):
            raise ValueError(
                "Profile 'x_levels'/'y_levels' do not match the target "
                "grid's extent and resolution."
            )

        return self._resample_z(grid, grid.azimuths, grid.colatitudes)

    @cache_by_id
    def _resample_z(
        self,
        grid: SpatialGrid,
        x_levels_out: ureg.Quantity,
        y_levels_out: ureg.Quantity,
    ) -> ParticleProfile:
        """
        Resample this profile onto ``grid``'s ``z`` layer centres by
        nearest-neighbour lookup.

        Parameters
        ----------
        grid : .SpatialGrid
            Target grid providing ``z`` layer centres.
        x_levels_out, y_levels_out : :class:`pint.Quantity`
            Horizontal level edges to store on the output profile, as-is.

        Returns
        -------
        ParticleProfile
            A new profile resampled onto ``grid``'s vertical layers.
        """
        z_levels_src = to_quantity(self.data["z_levels"]).m_as(ureg.meter)
        z_centers_src = 0.5 * (z_levels_src[:-1] + z_levels_src[1:])
        z_centers_tgt = grid.layers.m_as(ureg.meter)
        n_z_src = len(z_centers_src)

        mass_concentration_unit = str(
            to_quantity(self.data["mass_concentration"]).units
        )
        reff_unit = str(self.data["reff"].attrs.get("units", "micron"))
        veff_unit = str(self.data["veff"].attrs.get("units", "dimensionless"))

        reff_src = to_quantity(self.data["reff"]).m_as(reff_unit)  # (z, y, x)
        veff_src = to_quantity(self.data["veff"]).m_as(veff_unit)  # (z, y, x)
        mass_concentration_src = to_quantity(
            self.data["mass_concentration"]
        ).m_as(mass_concentration_unit)  # (z, y, x)

        # Nearest-neighbour lookup
        il = np.searchsorted(z_centers_src, z_centers_tgt, side="right") - 1
        in_range = (z_centers_tgt >= z_centers_src[0]) & (
            z_centers_tgt <= z_centers_src[-1]
        )
        il_safe = np.clip(il, 0, n_z_src - 2)
        iu_safe = il_safe + 1

        mid = 0.5 * (z_centers_src[il_safe] + z_centers_src[iu_safe])
        inn = np.where(z_centers_tgt < mid, il_safe, iu_safe)

        reff_tgt = reff_src[inn, :, :]
        veff_tgt = veff_src[inn, :, :]
        mass_concentration_tgt = mass_concentration_src[inn, :, :]

        out_of_range = ~in_range
        if np.any(out_of_range):
            mass_concentration_tgt = mass_concentration_tgt.copy()
            mass_concentration_tgt[out_of_range, :, :] = 0.0

        data_vars = {
            "reff": (["z", "y", "x"], reff_tgt, {"units": reff_unit}),
            "veff": (["z", "y", "x"], veff_tgt, {"units": veff_unit}),
            "mass_concentration": (
                ["z", "y", "x"],
                mass_concentration_tgt,
                {"units": mass_concentration_unit},
            ),
            "x_levels": (
                ["x_levels"],
                x_levels_out.m,
                {"units": str(x_levels_out.units)},
            ),
            "y_levels": (
                ["y_levels"],
                y_levels_out.m,
                {"units": str(y_levels_out.units)},
            ),
            "z_levels": (
                ["z_levels"],
                grid.levels.m,
                {"units": str(grid.levels.units)},
            ),
        }

        return ParticleProfile(data=xr.Dataset(data_vars=data_vars))

    def interp_reff_veff(self, property_grid: xr.DataArray) -> xr.DataArray:
        """
        Continuously interpolate a ``(reff, veff)``-indexed property grid
        (e.g. an ext/ssa grid from :class:`.ParticleProperties`) at this
        profile's own per-voxel ``reff``/``veff`` (dims ``z``, ``y``, ``x``).

        Points are clipped to *property_grid*'s own ``reff``/``veff``
        coordinate range, so that voxels sitting exactly on (or just past,
        from floating-point drift) the properties grid boundary do not
        turn into NaNs.
        """
        reff_pts = xr.DataArray(self.data["reff"].values, dims=["z", "y", "x"])
        veff_pts = xr.DataArray(self.data["veff"].values, dims=["z", "y", "x"])
        return _clip_and_interp(property_grid, reff_pts, veff_pts)

    def to_reff_veff_volumes(self) -> tuple[ureg.Quantity, ureg.Quantity]:
        """
        Return this profile's per-voxel ``reff``/``veff`` as dense
        ``(n_x, n_y, n_z)`` volumes. NaN marks empty cells.
        """
        mass_concentration = self.data["mass_concentration"].values  # (z, y, x)
        empty = mass_concentration <= 0

        reff_zyx = np.where(empty, np.nan, self.data["reff"].values)
        veff_zyx = np.where(empty, np.nan, self.data["veff"].values)

        reff_volume = np.transpose(reff_zyx, (2, 1, 0))
        veff_volume = np.transpose(veff_zyx, (2, 1, 0))

        return (
            ureg.Quantity(reff_volume, self.data["reff"].attrs["units"]),
            ureg.Quantity(veff_volume, self.data["veff"].attrs["units"]),
        )
