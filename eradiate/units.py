__all__ = [
    "symbol",
    "to_quantity",
]


import enum
from functools import lru_cache
from typing import Union

import pint
import pinttr
import xarray

# -- Global data members -------------------------------------------------------

#: Unit registry common to all Eradiate components. All units used in Eradiate
#: must be created using this registry.
unit_registry = pint.UnitRegistry()

unit_registry.define("dobson_unit = 2.687e20 * meter^-2 = du = dobson = dobson_units")


class PhysicalQuantity(enum.Enum):
    """An enumeration defining physical quantities known to Eradiate."""

    ALBEDO = "albedo"
    ANGLE = "angle"
    COLLISION_COEFFICIENT = "collision_coefficient"
    DIMENSIONLESS = "dimensionless"
    IRRADIANCE = "irradiance"
    LENGTH = "length"
    MASS = "mass"
    RADIANCE = "radiance"
    REFLECTANCE = "reflectance"
    SPEED = "speed"
    TIME = "time"
    TRANSMITTANCE = "transmittance"
    WAVELENGTH = "wavelength"

    @classmethod
    @lru_cache(maxsize=32)
    def spectrum(cls):
        """
        Return a tuple containing a subset of :class:`PhysicalQuantity`
        members suitable for :class:`.Spectrum` initialisation. This function
        caches its results for improved efficiency.
        """
        return (
            cls.ALBEDO,
            cls.COLLISION_COEFFICIENT,
            cls.DIMENSIONLESS,
            cls.IRRADIANCE,
            cls.RADIANCE,
            cls.REFLECTANCE,
            cls.TRANSMITTANCE,
        )


def _make_unit_context():
    uctx = pinttr.UnitContext(
        interpret_str=True, ureg=unit_registry, key_converter=PhysicalQuantity
    )

    # fmt: off
    for key, value in {
        # We allow for dimensionless quantities
        PhysicalQuantity.DIMENSIONLESS: pinttr.UnitGenerator(unit_registry.dimensionless),
        # Basic quantities must be named after their SI name
        # https://en.wikipedia.org/wiki/International_System_of_Units
        PhysicalQuantity.LENGTH: pinttr.UnitGenerator(unit_registry.m),
        PhysicalQuantity.TIME: pinttr.UnitGenerator(unit_registry.s),
        PhysicalQuantity.MASS: pinttr.UnitGenerator(unit_registry.kg),
        # Derived quantity names are more flexible
        PhysicalQuantity.ALBEDO: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.ANGLE: pinttr.UnitGenerator(unit_registry.deg),
        PhysicalQuantity.REFLECTANCE: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.TRANSMITTANCE: pinttr.UnitGenerator(unit_registry.dimensionless),
        PhysicalQuantity.WAVELENGTH: pinttr.UnitGenerator(unit_registry.nm),
    }.items():
        uctx.register(key, value)
    # fmt: on

    # The following quantities will update automatically based on their parent units
    uctx.register(
        PhysicalQuantity.COLLISION_COEFFICIENT,
        pinttr.UnitGenerator(lambda: uctx.get(PhysicalQuantity.LENGTH) ** -1),
    )
    uctx.register(
        PhysicalQuantity.IRRADIANCE,
        pinttr.UnitGenerator(
            lambda: unit_registry.watt
            / uctx.get(PhysicalQuantity.LENGTH) ** 2
            / uctx.get(PhysicalQuantity.WAVELENGTH),
        ),
    )
    uctx.register(
        PhysicalQuantity.RADIANCE,
        pinttr.UnitGenerator(
            lambda: unit_registry.watt
            / uctx.get(PhysicalQuantity.LENGTH) ** 2
            / unit_registry.steradian
            / uctx.get(PhysicalQuantity.WAVELENGTH),
        ),
    )

    return uctx


#: Unit context used when interpreting config dictionaries
unit_context_config = _make_unit_context()

#: Unit context used when building kernel dictionaries
unit_context_kernel = _make_unit_context()


# -- Public functions ----------------------------------------------------------


def symbol(units: Union[pint.Unit, str]) -> str:
    """
    Normalise a string or Pint units to a symbol string.

    Parameter ``units`` (:class:`pint.Unit` or srt):
        Value to convert to a symbol string.

    Returns → str:
        Symbol string (*e.g.* 'm' for 'metre', 'W / m ** 2' for 'W/m^2', etc.).
    """
    units = unit_registry.Unit(units)
    return format(units, "~")


def to_quantity(da: xarray.DataArray) -> pint.Quantity:
    """
    Converts a :class:`~xarray.DataArray` to a :class:`~pint.Quantity`.
    The array's ``attrs`` metadata mapping must contain a ``units`` field.

    .. note:: This function can also be used on coordinate variables.

    Parameter ``da`` (:class:`~xarray.DataArray`):
        :class:`~xarray.DataArray` instance which will be converted.

    Returns → :class:`pint.Quantity`:
        The corresponding Pint quantity.

    Raises → ValueError:
        If the array's metadata do not contain a ``units`` field.
    """
    try:
        units = da.attrs["units"]
    except KeyError as e:
        raise ValueError("this DataArray has no 'units' metadata field") from e
    else:
        return unit_registry.Quantity(da.values, units)
