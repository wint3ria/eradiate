"""Radiative property profile definitions.

.. admonition:: Atmospheric radiative properties data set specification (1D)

   The data structure is a :class:`~xarray.Dataset` with specific data
   variables, dimensions and data coordinates.

   Data variables must be:

   - ``sigma_a``: absorption coefficient [m^-1],
   - ``sigma_s``: scattering coefficient [m^-1],
   - ``sigma_t``: extinction coefficient [m^-1],
   - ``albedo``: albedo [dimensionless]

   The dimensions are ``z_layer`` and ``z_level``. All data variables depend
   on ``z_layer``.

   The data coordinates are:

   - ``z_layer``: layer altitude [m]. The layer altitude is an altitude
     representative of the given layer, e.g. the middle of the layer.
   - ``z_level``: level altitude [m]. The sole purpose of this data coordinate
     is to store the information on the layers sizes.

   In addition, the data set must include the following metadata attributes:
   ``convention``, ``title``, ``history``, ``source`` and ``reference``.
   Please refer to the `NetCDF Climate and Forecast (CF) Metadata Conventions <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#description-of-file-contents>`_
   for a description of these attributes.
   Additional attributes are allowed.
"""

import datetime
from abc import ABC, abstractmethod

import attr
import numpy as np
import pinttr
import xarray as xr

import eradiate
from .absorption import compute_sigma_a
from .rayleigh import compute_sigma_s_air
from .. import data
from .._units import unit_context_default as ucd
from .._units import unit_registry as ureg
from ..data.absorption_spectra import find_dataset
from ..thermoprops import us76
from ..util.attrs import (
    documented,
    parse_docs,
    validator_all_positive,
    validator_is_positive
)
from ..util.exceptions import ModeError
from ..util.factory import BaseFactory


@ureg.wraps(ret=None, args=("m", "m", "m^-1", "m^-1", "m^-1", None), strict=False)
def make_dataset(z_level, z_layer=None, sigma_a=None, sigma_s=None, sigma_t=None, albedo=None):
    r"""Makes an atmospheric radiative properties data set.

    Parameter ``z_level`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Level altitudes [m].

    Parameter ``z_layer`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Layer altitudes [m].

        If ``None``, the layer altitudes are computed automatically, so that
        they are halfway between the adjacent altitude levels.

    Parameter ``sigma_a`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Absorption coefficient values [m^-1].

    Parameter ``sigma_s`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Scattering coefficient values [m^-1].

    Parameter ``sigma_t`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Extinction coefficient values [m^-1].

    Parameter ``sigma_s`` (:class:numpy.ndarray or :class:`pint.Quantity`):
        Albedo values [/].

    Parameter ``profile`` (:class:`~xr.Dataset`):
        Atmospheric vertical profile.

    Returns → :class:`~xarray.Dataset`:
        Data set.
    """
    if z_layer is None:
        z_layer = (z_level[1:] + z_level[:-1]) / 2.

    if sigma_a is not None and sigma_s is not None:
        sigma_t = sigma_a + sigma_s
        albedo = sigma_s / sigma_t
    elif sigma_t is not None and albedo is not None:
        sigma_s = albedo * sigma_t
        sigma_a = sigma_t - sigma_s
    else:
        raise ValueError(
            "You must provide either one of the two pairs of arguments 'sigma_a' and 'sigma_s' or "
            "'sigma_t' and 'albedo'.")

    return xr.Dataset(
        data_vars={
            "sigma_a": (
                "z_layer",
                sigma_a,
                {
                    "units": "m^-1",
                    "standard_name": "absorption_coefficient"
                }
            ),
            "sigma_s": (
                "z_layer",
                sigma_s,
                {
                    "units": "m^-1",
                    "standard_name": "scattering_coefficient"
                }
            ),
            "sigma_t": (
                "z_layer",
                sigma_t,
                {
                    "units": "m^-1",
                    "standard_name": "extinction_coefficient"
                }
            ),
            "albedo": (
                "z_layer",
                albedo,
                {
                    "units": "",
                    "standard_name": "albedo"
                }
            )
        },
        coords={
            "z_level": (
                "z_level",
                z_level,
                {
                    "units": "m",
                    "standard_name": "level_altitude",
                }
            ),
            "z_layer": (
                "z_layer",
                z_layer,
                {
                    "units": "m",
                    "standard_name": "layer_altitude",
                }
            )
        },
        attrs={
            "convention": "CF-1.8",
            "title": "Atmospheric monochromatic radiative properties",
            "history":
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                f"data set creation - "
                f"{__name__}.make_dataset",
            "source": f"eradiate, version {eradiate.__version__}",
            "references": "",
        }
    )


@attr.s
class RadProfile(ABC):
    """An abstract base class for radiative property profiles. Classes deriving
    from this one must implement methods which return the albedo and collision
    coefficients as Pint-wrapped 3D Numpy arrays.

    .. warning::

       Arrays returned by the :data:`albedo`, :data:`sigma_a`, :data:`sigma_s`
       and :data:`sigma_t` properties **must** be 3D. Should the profile
       be one-dimensional, the invariant dimensions can be set to 1.

    .. seealso::

       :class:`.RadProfileFactory`

    """

    @classmethod
    def from_dict(cls, d):
        """Initialise a :class:`RadPropsProfile` from a dictionary."""
        d_copy = pinttr.interpret_units(d)
        return cls(**d_copy)

    @property
    @abstractmethod
    def albedo(self):
        """Return albedo.

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        pass

    @property
    @abstractmethod
    def sigma_t(self):
        """Return extinction coefficient.

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        pass

    @property
    @abstractmethod
    def sigma_a(self):
        """Return absorption coefficient.

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        pass

    @property
    @abstractmethod
    def sigma_s(self):
        pass

    @abstractmethod
    def to_dataset(self):
        """Return a dataset that holds the radiative properties of the
        corresponding atmospheric profile.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        pass


class RadProfileFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`RadProfile`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: RadProfileFactory
    """
    _constructed_type = RadProfile
    registry = {}


@RadProfileFactory.register("array")
@parse_docs
@attr.s
class ArrayRadProfile(RadProfile):
    """A flexible radiative property profile whose albedo and extinction
    coefficient are specified as numpy arrays. The underlying altitude
    mesh is assumed regular and specified by a single ``height`` parameter,
    which corresponds to the height of the atmosphere.

    .. warning::

       The ``albedo_values`` and ``sigma_t_values`` parameters must be 3D
       arrays even if the profile is 1D, and have the same shape.

    .. admonition:: Example

       The following creates a radiative property profile with 3 layers between
       0 and 5 kilometers, corresponding to a purely scattering atmosphere
       (albedo = 1) with scattering coefficient values of :code:`9e-6`,
       :code:`5e-6` and :code:`1e-6` in units of :code:`cdu[length]^-1`:

        .. code:: python

            import numpy as np

            rad_profile = ArrayRadProfile(
                sigma_t_values=np.array([9e-6, 5e-6, 1e-6]).reshape(1, 1, 3),
                albedo_values=np.ones((1, 1, 3)),
                height=ureg.Quantity(5, "km")
            )

        Note that the shape of the ``sigma_t_values`` and ``albedo_values``
        arrays is :code:`(1, 1, 3)`, where the last dimension corresponds
        to the ``z`` dimension.
    """

    albedo_values = documented(
        pinttr.ib(
            default=None,
            validator=validator_all_positive,
            units=ureg.dimensionless,
        ),
        doc="An array specifying albedo values. **Required, no default**.\n"
            "\n"
            "Unit-enabled field (dimensionless).",
        type="array",
    )

    height = documented(
        pinttr.ib(
            default=ureg.Quantity(100, "km"),
            validator=validator_is_positive,
            units=ucd.deferred("length")
        ),
        doc="Height of the atmosphere. Default: 100 km.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    sigma_t_values = documented(
        pinttr.ib(
            default=None,
            validator=validator_all_positive,
            units=ucd.deferred("collision_coefficient"),
        ),
        doc="An array specifying extinction coefficient values. **Required, no "
            "default**.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]^-1).",
        type="array",
    )

    @albedo_values.validator
    @sigma_t_values.validator
    def _validator_values(instance, attribute, value):
        if value.ndim != 3:
            raise ValueError(f"while setting {attribute.name}: "
                             f"must have 3 dimensions "
                             f"(got shape {value.shape})")

        if instance.albedo_values.shape != instance.sigma_t_values.shape:
            raise ValueError(f"while setting {attribute.name}: "
                             f"'albedo_values' and 'sigma_t_values' must have "
                             f"the same length")

    @property
    def albedo(self):
        """Return albedo.

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        return self.albedo_values

    @property
    def sigma_t(self):
        """Return extinction coefficient.

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        return self.sigma_t_values

    @property
    def sigma_a(self):
        """Return absorption coefficient.

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        return self.sigma_t * (1. - self.albedo)

    @property
    def sigma_s(self):
        """Return scattering coefficient.

        Returns → :class:`pint.Quantity`:
            Profile scattering coefficient.
        """
        return self.sigma_t * self.albedo

    def to_dataset(self):
        """Return a dataset that holds the radiative properties of the
        corresponding atmospheric profile.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        n_layers = self.sigma_t.size
        return make_dataset(
            z_level=np.linspace(0., self.height.to("m"), n_layers + 1),
            sigma_t=self.sigma_t.flatten(),
            albedo=self.albedo.flatten()
        )


@RadProfileFactory.register("us76_approx")
@parse_docs
@attr.s
class US76ApproxRadProfile(RadProfile):
    """A US76-approximation radiative profile.

    .. note::

       Instantiating this class requires to download the absorption dataset
       ``spectra-us76_u86_4`` and place it in ``$ERADIATE_DIR/resources/data/``.

    The radiative properties are computed based upon the so-called US76
    atmospheric vertical profile.
    The scattering coefficient is computed with
    :func:`sigma_s_air<eradiate.radprops.rayleigh.sigma_s_air>`
    using the total number density from the US76 atmospheric vertical
    profile.
    The absorption coefficient is computed in two steps. First, the
    absorption cross section is computed by interpolating the
    absorption cross section datasets for the :ref:`us76_u86_4 <sec-user_guide-molecular_absorption_datasets-spectra_us76_u86_4>`
    mixture at the wavelength specified in the eradiate mode and at the pressure values
    corresponding to the US76 atmospheric vertical profile. The second step
    consists in multiplying these cross sections by the total number density
    values from the US76 atmospheric vertical profile, in the corresponding
    atmospheric layers.
    """
    n_layers = documented(
        attr.ib(
            default=50,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Number of atmospheric layers. Default: 50",
        type="int",
    )

    height = documented(
        pinttr.ib(
            default=ureg.Quantity(100., ureg.km),
            validator=validator_is_positive,
            units=ucd.deferred("length"),
        ),
        doc="Atmosphere's height. Default: 100 km.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float"
    )

    _sigma_s_values = pinttr.ib(
        default=None,
        units=ucd.deferred("collision_coefficient"),
        init=False,
        repr=False,
    )

    _sigma_a_values = pinttr.ib(
        default=None,
        units=ucd.deferred("collision_coefficient"),
        init=False,
        repr=False,
    )

    _thermo_profile = attr.ib(
        default=None,
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self):
        # Compute total number density and pressure values
        altitude_mesh = np.linspace(
            start=0.,
            stop=self.height,
            num=self.n_layers + 1
        )

        # make US76 atmospheric profile
        profile = us76.make_profile(altitude_mesh)
        self._thermo_profile = profile

        self.update()

    def update(self):
        """Update internal variables. An update is required to recompute the
        internal state when the wavelength is changed.
        """
        mode = eradiate.mode()

        if not mode.is_monochromatic():
            raise ModeError(f"unsupported mode {mode.id}")
        else:
            wavelength = mode.wavelength

        # Compute scattering coefficient
        profile = self._thermo_profile
        self._sigma_s_values = compute_sigma_s_air(
            wavelength=wavelength,
            number_density=ureg.Quantity(profile.n_tot.values, profile.n_tot.units),
        )

        # find the absorption dataset
        wavenumber = (1.0 / wavelength).to("cm^-1")
        dataset_id = find_dataset(
            wavenumber=wavenumber.magnitude,
            absorber="us76_u86_4",
            engine="spectra",
        )

        dataset = data.open(category="absorption_spectrum", id=dataset_id)
        # Compute absorption coefficient
        self._sigma_a_values = compute_sigma_a(
            dataset,
            wl=wavelength,
            p=profile.p.values,
            n=profile.n_tot.values,
            p_fill_value=0.,  # us76_u86_4 dataset is limited to pressures above
            # 0.101325 Pa, but us76 thermophysical profile goes below that
            # value for altitudes larger than 93 km. At these altitudes, the
            # number density is so small compared to that at the sea level that
            # we assume it is negligible.
        )

    @property
    def albedo(self):
        """Return albedo.

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        return (self.sigma_s / self.sigma_t).to(ureg.dimensionless)

    @property
    def sigma_a(self):
        """Return absorption coefficient.

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        return self._sigma_a_values[np.newaxis, np.newaxis, ...]

    @property
    def sigma_s(self):
        """Return scattering coefficient.

        Returns → :class:`pint.Quantity`:
            Profile scattering coefficient.
        """
        return self._sigma_s_values[np.newaxis, np.newaxis, ...]

    @property
    def sigma_t(self):
        """Return extinction coefficient.

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        return self.sigma_a + self.sigma_s

    def to_dataset(self):
        """Return a dataset that holds the radiative properties of the
        corresponding atmospheric profile.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        return make_dataset(
            z_level=self._thermo_profile.z_level.values,
            z_layer=self._thermo_profile.z_layer.values,
            sigma_a=self.sigma_a.flatten(),
            sigma_s=self.sigma_s.flatten(),
        )
