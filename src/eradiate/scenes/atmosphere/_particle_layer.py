"""
Particle layers.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import singledispatchmethod
from typing import Callable, Literal

import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr

from ._core import AtmosphericMedium
from ._particle_dist import ParticleDistribution, particle_distribution_factory
from ..core import traverse
from ..geometry import PlaneParallelGeometry, SceneGeometry, SphericalShellGeometry
from ..phase import TabulatedPhaseFunction
from ... import converters
from ...attrs import define, documented
from ...contexts import KernelContext
from ...grid import GridCoords
from ...kernel import SceneParameter
from ...spectral.index import (
    CKDSpectralIndex,
    MonoSpectralIndex,
    SpectralIndex,
)
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id, summary_repr
from ...validators import is_positive


def _particle_layer_distribution_converter(value):
    if isinstance(value, str):
        if value == "uniform":
            return particle_distribution_factory.convert({"type": "uniform"})
        elif value == "gaussian":
            return particle_distribution_factory.convert({"type": "gaussian"})
        elif value == "exponential":
            return particle_distribution_factory.convert({"type": "exponential"})

    return particle_distribution_factory.convert(value)


def _extent_converter_x(value, self_):
    if not (value is None or isinstance(value, dict)):
        return value
    if self_.geometry is None:
        raise ValueError("Could not infer default extent for a None geometry")
    elif isinstance(self_.geometry, PlaneParallelGeometry):
        if isinstance(value, dict):
            return LengthExtent(**value)
        return LengthExtent(
            extent_min=self_.geometry.grid.edges_x.min(),
            extent_max=self_.geometry.grid.edges_x.max(),
        )
    elif isinstance(self_.geometry, SphericalShellGeometry):
        if isinstance(value, dict):
            return AngleExtent(**value)
        return AngleExtent(
            extent_min=self_.geometry.grid.azimuths.min(),
            extent_max=self_.geometry.grid.azimuths.max(),
        )
    else:
        raise NotImplementedError(
            f"Geometric type {type(self_.geometry)} is not supported"
        )


def _extent_converter_y(value, self_):
    if not (value is None or isinstance(value, dict)):
        return value
    if self_.geometry is None:
        raise ValueError("Could not infer default extent for a None geometry")
    elif isinstance(self_.geometry, PlaneParallelGeometry):
        if isinstance(value, dict):
            return LengthExtent(**value)
        return LengthExtent(
            extent_min=self_.geometry.grid.edges_y.min(),
            extent_max=self_.geometry.grid.edges_y.max(),
        )
    elif isinstance(self_.geometry, SphericalShellGeometry):
        if isinstance(value, dict):
            return AngleExtent(**value)
        return AngleExtent(
            extent_min=self_.geometry.grid.colatitudes.min(),
            extent_max=self_.geometry.grid.colatitudes.max(),
        )
    else:
        raise NotImplementedError(
            f"Geometric type {type(self_.geometry)} is not supported"
        )


@define(eq=False, slots=False)
class HorizontalExtent:
    """
    Abstract horizontal extent class. Handles unit
    conversion from plane parallel extents in lengths
    to spherical extents in angles, depending on the
    considered radius.
    """

    @property
    def length(self):
        return self.extent_max - self.extent_min

    @property
    def extent_min(self) -> pint.Quantity:
        return self._extent_min

    @property
    def extent_max(self) -> pint.Quantity:
        return self._extent_max

    @abstractmethod
    def to_geometry(self, g: SceneGeometry) -> HorizontalExtent:
        pass

    def eval_geometry(self, g: SceneGeometry, op: Callable):
        extent = self.to_geometry(g)
        try:
            return op(extent)
        except pint.DimensionalityError as e:
            raise RuntimeError(
                f"Could not evaluate operation on extent {self}. "
                f"Attempt to convert extent to the provided {type(g)} geometry "
                f"yields the extent {extent} that is not compatible with the operation."
            ) from e

    def eval_length(self, g: SceneGeometry) -> pint.Quantity:
        return self.eval_geometry(g, lambda e: e.length)

    def eval_min(self, g: SceneGeometry) -> pint.Quantity:
        return self.eval_geometry(g, lambda e: e.extent_min)

    def eval_max(self, g: SceneGeometry) -> pint.Quantity:
        return self.eval_geometry(g, lambda e: e.extent_max)


@define(eq=False, slots=False)
class LengthExtent(HorizontalExtent):
    """
    Horizontal extent of a ParticleLayer expressed on a local
    plane parellel atmospheric coordinates grid.
    """

    _extent_min: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(-1.0, ureg.km),
            validator=[pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Min limit of the horizontal extent in a plane parallel grid.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="float or quantity",
        default="-1 km",
    )

    _extent_max: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(1.0, ureg.km),
            validator=[pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Max limit of the horizontal extent in a plane parallel grid.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="float or quantity",
        default="1 km.",
    )

    @_extent_min.validator
    @_extent_max.validator
    def _extent_validator(self, attribute, value):
        assert self._extent_min < self._extent_max

    def to_geometry(self, g: SceneGeometry | None) -> HorizontalExtent:
        """
        Attempt to convert a plane parallel horizontal extent to a compatible
        geometric type.

        Return ``self`` if ``g`` is a ``PlaneParallelGeometry`` or g is ``None``,
        without any conversion performed.

        If ``g`` is a SphericalShellGeometry, the following conversion is performed:
        * The zero position of the plane parallel grid is positioned at (colatitude=0°, azimuth=0°).
        * The source extent min and max lengths are mapped to the sphere angular positions from this frame.
        using the ``planet_radius`` of g.
        * The return type is then an ``AngleExtent``.

        Emits a warning if the conversion is performed.
        """

        if g is None:
            return self
        if isinstance(g, PlaneParallelGeometry):
            return self
        if isinstance(g, SphericalShellGeometry):
            warnings.warn(
                "Performing automatic conversion of particles horizontal extent "
                "from plane parallel to spherical geometry at runtime. The converter "
                "placed the plane parallel coordinate reference frame at the arbitrary "
                "point (colatitude=0°, azimuth=0°), which might not correspond to any of"
                "the measure's observation target. Particles location on the spherical "
                "coordinate grid should be defined explicitely by an angular extent "
                "instead."
            )
            radius = g.planet_radius
            extent_min = pint.Quantity(
                self._extent_min.m_as(ureg.m) / radius.m_as(ureg.m), ureg.rad
            )
            extent_max = pint.Quantity(
                self._extent_max.m_as(ureg.m) / radius.m_as(ureg.m), ureg.rad
            )
            return AngleExtent(extent_min=extent_min, extent_max=extent_max)

        raise NotImplementedError("Not supported geometry type ({type(g)})")


@define(eq=False, slots=False)
class AngleExtent(HorizontalExtent):
    """
    Horizontal extent of a ParticleLayer expressed on a geocentric
    spherical atmospheric coordinates grid.
    """

    _extent_min: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.degree),
            validator=[pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Min limit of the horizontal extent in a spherical grid.\n"
        "\n"
        "Unit-enabled field (default: ucc[angle])",
        type="quantity",
        init_type="float or quantity",
        default="0 degree",
    )

    _extent_max: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(1.0, ureg.degree),
            validator=[pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Max limit of the horizontal extent in a spherical grid.\n"
        "\n"
        "Unit-enabled field (default: ucc[angle]).",
        type="quantity",
        init_type="float or quantity",
        default="1 degree",
    )

    @_extent_min.validator
    @_extent_max.validator
    def _extent_validator(self, attribute, value):
        assert self._extent_min < self._extent_max

    def to_geometry(self, g: Geometry) -> HorizontalExtent:
        """
        Attempt to convert an angular horizontal extent to a compatible
        geometric type.

        Return ``self`` if ``g`` is a ``SphericalShellGeometry`` or g is ``None``,
        without any conversion performed.

        Raises ValueError If ``g`` is a PlaneParallelGeometry.
        """

        if g is None:
            return self
        if isinstance(g, SphericalShellGeometry):
            return self
        if isinstance(g, PlaneParallelGeometry):
            # Cannot convert to plane parallel without a planet radius information,
            raise ValueError(
                "Cannot automatically convert an extent defined on a spherical "
                "coordinate grid to a plane parallel coordinate grid."
            )
        raise NotImplementedError("Not supported geometry type ({type(g)})")


@define(eq=False, slots=False)
class ParticleLayer(AtmosphericMedium):
    """
    Particle layer scene element [``particle_layer``].

    The particle layer has a vertical extension specified by a bottom altitude
    (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the layer, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available
    distribution types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow to describe the variations of the particles number
    with altitude.
    The particle density in the layer is adjusted so that the particle layer's
    optical thickness at a specified reference wavelength (``w_ref``) meets a
    specified value (``tau_ref``).
    The particles radiative properties are specified by a data set
    (``dataset``).

    Notes
    -----
    If the optical property dataset contains only one spectral data point, the
    data is considered uniform throughout the entire spectrum.
    """

    bottom: pint.Quantity = documented(
        pinttrs.field(
            default=ureg.Quantity(0.0, ureg.km),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="float or quantity",
        default="0 km",
    )

    top: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="float or quantity",
        default="1 km.",
    )

    y_extent: HorizontalExtent = documented(
        attrs.field(
            default=None,
            validator=[
                attrs.validators.optional(
                    attrs.validators.instance_of(HorizontalExtent)
                )
            ],
            converter=attrs.Converter(_extent_converter_y, takes_self=True),
        ),
        doc="Horizontal extent of the particles in the Y dimension of the atmosphere "
        "coordinate system.",
        init_type="`NoneType`",
        type=":class:`.HorizontalExtent`",
        default="LengthExtent()",
    )

    x_extent: HorizontalExtent = documented(
        attrs.field(
            default=None,
            validator=[
                attrs.validators.optional(
                    attrs.validators.instance_of(HorizontalExtent)
                )
            ],
            converter=attrs.Converter(_extent_converter_x, takes_self=True),
        ),
        doc="Horizontal extent of the particles in the X dimension of the atmosphere "
        "coordinate system.",
        init_type="`NoneType`",
        type=":class:`.HorizontalExtent`",
        default="LengthExtent()",
    )

    @y_extent.validator
    @x_extent.validator
    def _extent_validator(self, attribute, value):
        # Use type() rather than isinstance() to enforce that both extents
        # are the same concrete type, not just compatible supertypes.
        if type(self.x_extent) is not type(self.y_extent):
            raise ValueError(
                f"while validating '{attribute.name}': x_extent and y_extent must "
                f"be of the same type (got {type(self.x_extent).__name__} and "
                f"{type(self.y_extent).__name__})"
            )

    @bottom.validator
    @top.validator
    def _bottom_top_validator(self, attribute, value):
        if np.any(self.bottom >= self.top):
            raise ValueError(
                f"while validating '{attribute.name}': bottom altitude must be "
                "lower than top altitude "
                f"(got bottom={self.bottom}, top={self.top})"
            )

    distribution: ParticleDistribution = documented(
        attrs.field(
            default="uniform",
            converter=_particle_layer_distribution_converter,
            validator=attrs.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution in the Z dimension of the coords grid. "
        "Simple defaults can be set using a string: "
        '``"uniform"`` (resp. ``"gaussian"``, ``"exponential"``) is converted to '
        ":class:`UniformParticleDistribution() <.UniformParticleDistribution>` "
        "(resp. :class:`GaussianParticleDistribution() <.GaussianParticleDistribution>`, "
        ":class:`ExponentialParticleDistribution() <.ExponentialParticleDistribution>`).",
        init_type=":class:`.ParticleDistribution` or dict or "
        '{"uniform", "gaussian", "exponential"}, optional',
        type=":class:`.ParticleDistribution`",
        default='"uniform"',
    )

    w_ref: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("wavelength"),
            default=550.0 * ureg.nm,
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Reference wavelength at which the extinction optical thickness is "
        "specified. To minimize the uncertainty on the computed extinction "
        "coefficient, it is recommended that this wavelength is included in the "
        "provided radiative property dataset.\n"
        "\n"
        "Unit-enabled field (default: ucc['wavelength']).",
        type="quantity",
        init_type="quantity or float",
        default="550.0 nm",
    )

    @w_ref.validator
    def _w_ref_validator(self, attribute, value):
        w_units = self.dataset["w"].attrs["units"]
        if not np.any(np.isclose(value.m_as(w_units), self.dataset["w"].values)):
            warnings.warn(
                "While initializing ParticleLayer: the provided aerosol "
                "single-scattering property dataset does not contain the selected "
                f"reference wavelength (w_ref = {value})"
            )

    tau_ref: pint.Quantity = documented(
        pinttrs.field(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[is_positive, pinttrs.validators.has_compatible_units],
        ),
        doc="Extinction optical thickness at the reference wavelength.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    @tau_ref.validator
    def _tau_ref_all_positive(self, attribute, value):
        if np.any(value < 0.0):
            raise ValueError(
                "While initialising ParticleLayer: reference "
                "extinction optical thickness must be positive"
            )

    @tau_ref.validator
    @bottom.validator
    @top.validator
    def _xy_shape_validator(self, attribute, value):
        if self.geometry is None:
            return
        if np.size(value) == 1:
            return
        if value.shape != self.geometry.grid.shape[:2]:
            raise ValueError(
                "While initialising ParticleLayer: the shape of the "
                f"{attribute.name} attribute is inconsistent with the "
                "scene geometry. Expected a scalar value or a "
                f"{self.geometry.grid.shape[:2]} sized array, "
                f"received a {value.shape} sized array."
            )

    dataset: xr.Dataset = documented(
        attrs.field(
            default="govaerts_2021-continental",
            converter=converters.passthrough_type(xr.Dataset)(
                attrs.converters.pipe(
                    converters.resolve_keyword(lambda x: f"aerosol/{x}.nc"),
                    converters.resolve_path,
                    converters.load_dataset,
                )
            ),
            validator=attrs.validators.instance_of(xr.Dataset),
            repr=summary_repr,
        ),
        doc="Particle radiative property data set. "
        "If an xarray dataset is passed, the dataset is used as is "
        "(refer to the data guide for the format requirements of this dataset). "
        "If a path is passed, the converter looks it up on the hard drive, using "
        "the file resolver. "
        "If a string is passed, it is interpreted as a particle radiative "
        "property dataset identifier.",
        type="Dataset",
        init_type="Dataset or path-like or str",
        default='"govaerts_2021-continental"',
    )

    has_absorption: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="Absorption bypass switch. If ``True``, the absorption coefficient "
        "is computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(default=True, converter=bool),
        doc="Scattering bypass switch. If ``True``, the scattering coefficient "
        "is computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    @has_absorption.validator
    @has_scattering.validator
    def _switch_validator(self, attribute, value):
        if not self.has_absorption and not self.has_scattering:
            raise ValueError(
                f"while validating {attribute.name}: at least one of "
                "'has_absorption' and 'has_scattering' must be True"
            )

    force_polarized_phase: bool = documented(
        attrs.field(default=False, converter=bool),
        doc="Force the use of a polarized phase function implementation, even"
        "when no polarization information is available.",
        type="bool",
        default="False",
    )

    particle_shape: Literal["spherical", "spheroidal"] = documented(
        attrs.field(default="spherical", kw_only=True),
        doc="Defines the shape of the particle. Only used in polarized mode.\n\n"
        '* ``"spherical"``: 4 coefficients considered [m11, m12, m33, m34].\n'
        '* ``"spheroidal"``: 6 coefficients considered [m11, m12, m22, m33, m34, m44].',
        type="str",
        init_type='{"spherical", "spheroidal"}',
        default='"spherical"',
    )

    _phase: TabulatedPhaseFunction | None = attrs.field(default=None, init=False)

    def update(self) -> None:
        self._phase = TabulatedPhaseFunction(
            id=self.phase_id,
            data=self.dataset.phase,
            force_polarized_phase=self.force_polarized_phase,
            particle_shape=self.particle_shape,
        )

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    def eval_fractions(self, grid: GridCoords) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer in 3D.

        Each dimension of the fractions grid is evaluated according to their
        respective particle distribution.

        The Z dimension (vertical extent) also supports spatially varying
        parameters for its top, bottom, and tau_ref parameters. In these cases
        each value of the parameter is evaluated independently for the corresponding
        1D Z-column of particles on the uniform 3D grid.

        Returns
        -------
        ndarray
            Particle number fractions as a (n_x, n_y, n_layers,)-shaped array.
        """

        # Variable bottom and top altitude
        bottom = np.atleast_1d(self.bottom)
        layers = np.atleast_3d(np.ones(bottom.shape)) @ np.atleast_2d(grid.layers)

        def eval_dim(grid_dim):
            return lambda ext: (grid_dim - ext.extent_min) / ext.length

        x = self.x_extent.eval_geometry(self.geometry, eval_dim(grid.cells_x))
        y = self.y_extent.eval_geometry(self.geometry, eval_dim(grid.cells_y))
        z = (layers.T - bottom.T).T / (self.top - self.bottom)

        # Only the z dimension (altitude) supports variable extents wrt to x and y dims.
        if x.squeeze().ndim > 1 or y.squeeze().ndim > 1:
            raise ValueError("Multidimensional horizontal extents are not supported")

        fractions = self.distribution(z.m_as(ureg.dimensionless))

        out = np.zeros(fractions.shape)
        fractions_sum = fractions.sum(axis=-1).T
        res = np.divide(fractions.T, fractions_sum, out=out, where=fractions_sum > 0.0)

        return out.T

    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        min_sigma_s = self.eval_sigma_s(ctx.si).min(axis=-1)
        out = np.full(min_sigma_s.shape, np.inf)
        np.divide(
            np.ones(min_sigma_s.shape[:-1]),
            min_sigma_s,
            where=min_sigma_s != 0,
            out=out,
        )
        return out * ureg.m

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @cache_by_id
    def _eval_albedo_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return albedo from dataset (without accounting for bypass switches)
        # This routine returns an array of shape (n_wavelengths, [x, y, ]n_layers)
        ds = self.dataset
        wavelengths = w.m_as(ds.w.attrs["units"])

        if len(ds["w"]) == 1:
            interpolated = to_quantity(ds.albedo.sel(w=wavelengths, method="nearest"))
        else:
            interpolated = to_quantity(ds.albedo.interp(w=np.atleast_1d(wavelengths)))

        assert interpolated.ndim == 1 and interpolated.size == np.size(wavelengths)
        fractions = self.eval_fractions(grid)
        where_present = fractions > 0
        where_present = where_present.reshape(
            *np.ones(3 - fractions.ndim, dtype=int), *fractions.shape, 1
        )
        interpolated = interpolated.reshape(1, 1, interpolated.size, 1)

        albedo = np.transpose(where_present @ interpolated, (3, 0, 1, 2))
        return albedo

    @cache_by_id
    def _eval_sigma_t_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return extinction coefficient from dataset (without accounting
        # for bypass switches). Returns an array of shape (n_wavelengths, [x, y, ]n_layers)

        ds = self.dataset
        ds_w_units = ureg(ds.w.attrs["units"])
        wavelengths = np.atleast_1d(w.m_as(ds_w_units))

        if len(ds["w"]) == 1:
            sigma_t_star = to_quantity(ds.sigma_t.sel(w=wavelengths, method="nearest"))
            sigma_t_star_ref = to_quantity(
                ds.sigma_t.sel(w=self.w_ref.m_as(ds_w_units), method="nearest")
            )
        else:
            sigma_t_star = to_quantity(ds.sigma_t.interp(w=wavelengths))
            sigma_t_star_ref = to_quantity(
                ds.sigma_t.interp(w=np.atleast_1d(self.w_ref.m_as(ds_w_units)))
            )

        # Compute target optical thickness value
        tau = (
            np.atleast_3d(self.tau_ref)
            @ np.reshape(sigma_t_star, (1, 1, -1))
            / sigma_t_star_ref
        )

        # Scatter this total OT to all layers
        # TODO: Make sure that axis order is consistent with other vectorized
        #  routines
        fractions = self.eval_fractions(grid)

        tau_layers = np.transpose(
            tau[..., np.newaxis] @ fractions[..., np.newaxis, :], [2, 0, 1, 3]
        )

        # Compute corresponding average coefficient
        sigma_t = tau_layers / grid.layer_height

        return sigma_t

    def _eval_sigma_a_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return absorption coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_layers)
        albedo = self._eval_albedo_impl(w, grid)
        return self._eval_sigma_t_impl(w, grid) * (1.0 - albedo.m)

    def _eval_sigma_s_impl(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Return scattering coefficient from dataset (without accounting for
        # bypass switches). This routine is vectorized and returns an array of
        # shape (n_wavelengths, n_layers)
        albedo = self._eval_albedo_impl(w, grid)
        return self._eval_sigma_t_impl(w, grid) * albedo.m

    @singledispatchmethod
    def eval_albedo(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_albedo.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_albedo_mono(
            w=si.w,
            grid=self.geometry.grid if grid is None else grid,
        )

    @eval_albedo.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_albedo_ckd(
            w=si.w,
            g=si.g,
            grid=self.geometry.grid if grid is None else grid,
        )

    def eval_albedo_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        if self.has_absorption and self.has_scattering:
            albedo = self._eval_albedo_impl(w, grid).squeeze(axis=0)

        elif self.has_absorption and not self.has_scattering:
            albedo = 0.0 * ureg.dimensionless

        elif self.has_scattering and not self.has_absorption:
            albedo = 1.0 * ureg.dimensionless

        else:
            raise RuntimeError

        # Albedo is constant vs spatial dimension
        return np.full_like(grid.layers, albedo)

    def eval_albedo_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_albedo_mono(w=w, grid=grid)

    @singledispatchmethod
    def eval_sigma_t(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_t.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_t_mono(
            w=si.w,
            grid=self.geometry.grid if grid is None else grid,
        )

    @eval_sigma_t.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_t_ckd(
            w=si.w,
            g=si.g,
            grid=self.geometry.grid if grid is None else grid,
        )

    def eval_sigma_t_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        result = self._eval_sigma_t_impl(w, grid).squeeze(axis=0)

        if self.has_absorption and self.has_scattering:
            return result

        elif not self.has_absorption and self.has_scattering:
            return result - self._eval_sigma_a_impl(w, grid)

        elif self.has_absorption and not self.has_scattering:
            return result - self._eval_sigma_s_impl(w, grid)

        raise RuntimeError

    def eval_sigma_t_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        return self.eval_sigma_t_mono(w=w, grid=grid)

    @singledispatchmethod
    def eval_sigma_a(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_a.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_a_mono(
            w=si.w,
            grid=self.geometry.grid if grid is None else grid,
        )

    @eval_sigma_a.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_a_ckd(
            w=si.w,
            g=si.g,
            grid=self.geometry.grid if grid is None else grid,
        )

    def eval_sigma_a_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        value = self._eval_sigma_a_impl(w, grid).squeeze(axis=0)
        return value if self.has_absorption else np.zeros_like(value) * value.units

    def eval_sigma_a_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_a_mono(w, grid)

    @singledispatchmethod
    def eval_sigma_s(
        self, si: SpectralIndex, grid: GridCoords | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        raise NotImplementedError

    @eval_sigma_s.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_s_mono(
            w=si.w,
            grid=self.geometry.grid if grid is None else grid,
        )

    @eval_sigma_s.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords | None = None) -> pint.Quantity:
        return self.eval_sigma_s_ckd(
            w=si.w,
            g=si.g,
            grid=self.geometry.grid if grid is None else grid,
        )

    def eval_sigma_s_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        value = self._eval_sigma_s_impl(w, grid).squeeze(axis=0)
        return value if self.has_scattering else np.zeros_like(value) * value.units

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_s_mono(w, grid)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> TabulatedPhaseFunction:
        # Inherit docstring
        return self._phase

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict[str, SceneParameter]:
        _, result = traverse(self.phase)
        return result.data
