"""
Atmospheric thermophysical properties profiles models according to
:cite:`Anderson1986AtmosphericConstituentProfiles`.
"""
import typing as t

import pint
import xarray as xr

import eradiate

from .util import compute_scaling_factors, interpolate, rescale_concentration
from ..data import open
from .._mode import ModeFlags


def make_profile(
    model_id: str = "us_standard",
    levels: t.Optional[pint.Quantity] = None,
    concentrations: t.Optional[t.MutableMapping[str, pint.Quantity]] = None,
) -> xr.Dataset:
    """
    Makes the atmospheric profiles from the AFGL's 1986 technical report
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    Parameters
    ----------
    model_id : {"us_standard", "midlatitude_summer", "midlatitude_winter", "subarctic_summer", "subarctic_winter", "tropical"}, default: "us_standard"
        Model identifier.

    levels : quantity or array, optional
        Altitude levels. The array must contain at least two values.
        If not provided, the atmospheric profile is built using the data set's
        altitude levels.

    concentrations : dict, optional
        Molecules concentrations as a {str: quantity} mapping.

    Returns
    -------
    Dataset
        Atmospheric profile.

    Notes
    -----
    :cite:`Anderson1986AtmosphericConstituentProfiles` defines six models,
    listed in the table below.

    .. list-table:: AFGL (1986) atmospheric thermophysical properties profiles models
       :widths: 2 4 4
       :header-rows: 1

       * - Model number
         - Model identifier
         - Model name
       * - 1
         - ``tropical``
         - Tropic (15N Annual Average)
       * - 2
         - ``midlatitude_summer``
         - Mid-Latitude Summer (45N July)
       * - 3
         - ``midlatitude_winter``
         - Mid-Latitude Winter (45N Jan)
       * - 4
         - ``subarctic_summer``
         - Sub-Arctic Summer (60N July)
       * - 5
         - ``subarctic_winter``
         - Sub-Arctic Winter (60N Jan)
       * - 6
         - ``us_standard``
         - U.S. Standard (1976)

    .. attention::
       The original altitude mesh specified by
       :cite:`Anderson1986AtmosphericConstituentProfiles` is a piece-wise
       regular altitude mesh with an altitude step of 1 km from 0 to 25 km,
       2.5 km from 25 km to 50 km and 5 km from 50 km to 120 km.
       Since the Eradiate kernel only supports regular altitude mesh, the
       original atmospheric thermophysical properties profiles were
       interpolated on the regular altitude mesh with an altitude step of 1 km
       from 0 to 120 km.

    Although the altitude meshes of the interpolated
    :cite:`Anderson1986AtmosphericConstituentProfiles` profiles is fixed,
    this function lets you define a custom altitude mesh (regular or irregular).

    All six models include the following six absorbing molecular species:
    H2O, CO2, O3, N2O, CO, CH4 and O2.
    The concentrations of these species in the atmosphere is fixed by
    :cite:`Anderson1986AtmosphericConstituentProfiles`.
    However, this function allows you to rescale the concentrations of each
    individual molecular species to custom concentration values.
    Custom concentrations can be provided in different units.
    For more information about rescaling process and the supported
    concentration units, refer to the documentation of
    :func:`~eradiate.thermoprops.util.compute_scaling_factors`.
    """
    if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        if model_id != "us_standard":
            raise NotImplementedError(
                "In CKD mode, only the 'us_standard' model is supported."
            )
        species = set(concentrations.keys()) if concentrations else set()
        unhandled = species - {"H2O", "O3"}

        if unhandled:
            raise NotImplementedError(
                f"species '{unhandled}' cannot be rescaled in ckd mode"
            )

    thermoprops = open(category="thermoprops_profiles", id="afgl1986-" + model_id)

    if levels is not None:
        thermoprops = interpolate(ds=thermoprops, z_level=levels, conserve_columns=True)

    if concentrations is not None:
        factors = compute_scaling_factors(ds=thermoprops, concentration=concentrations)
        thermoprops = rescale_concentration(ds=thermoprops, factors=factors)

    return thermoprops
