"""Homogeneous atmosphere scene elements."""

import attr

import eradiate
from .base import Atmosphere, AtmosphereFactory
from ..spectra import Spectrum, SpectrumFactory, UniformSpectrum
from ...radprops.rayleigh import compute_sigma_s_air
from ...util.attrs import (
    converter_or_auto,
    documented,
    parse_docs,
    validator_has_quantity,
    validator_or_auto
)
from ...util.collections import onedict_value
from ..._units import unit_context_kernel as uck


@AtmosphereFactory.register("homogeneous")
@parse_docs
@attr.s()
class HomogeneousAtmosphere(Atmosphere):
    """Homogeneous atmosphere scene element [:factorykey:`homogeneous`].

    This class builds an atmosphere consisting of a homogeneous medium.
    Scattering uses the Rayleigh phase function.
   """

    sigma_s = documented(
        attr.ib(
            default="auto",
            converter=converter_or_auto(
                SpectrumFactory.converter("collision_coefficient")
            ),
            validator=validator_or_auto(
                attr.validators.instance_of(Spectrum),
                validator_has_quantity("collision_coefficient")
            ),
        ),
        doc="Atmosphere scattering coefficient value. If set to ``\"auto\"``, "
            "the scattering coefficient will be computed based on the current "
            "operational mode configuration using the :func:`sigma_s_air` "
            "function.\n"
            "\n"
            "Can be initialised with a dictionary processed by "
            ":class:`.SpectrumFactory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum` or \"auto\"",
        default="``\"auto\"``",
    )

    sigma_a = documented(
        attr.ib(
        default=0.,
        converter=SpectrumFactory.converter("collision_coefficient"),
        validator=[attr.validators.instance_of(Spectrum),
                   validator_has_quantity("collision_coefficient")]
    ),
        doc="Atmosphere absorption coefficient value. Defaults disable "
            "absorption.\n"
            "\n"
            "Can be initialised with a dictionary processed by "
            ":class:`.SpectrumFactory`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.0 cdu[collision_coefficient]"
    )

    @property
    def kernel_width(self):
        """Width of the kernel object delimiting the atmosphere."""
        if self.width == "auto":
            return 10. / self._sigma_s.value
        else:
            return self.width

    @property
    def _albedo(self):
        """Return albedo."""
        return UniformSpectrum(
            quantity="albedo",
            value=self._sigma_s.value / (self._sigma_s.value + self.sigma_a.value)
        )

    @property
    def _sigma_s(self):
        """Return scattering coefficient based on configuration."""
        if self.sigma_s == "auto":
            return UniformSpectrum(
                quantity="collision_coefficient",
                value=compute_sigma_s_air(wavelength=eradiate.mode().wavelength)
            )
        else:
            return self.sigma_s

    @property
    def _sigma_t(self):
        """Return extinction coefficient."""
        return UniformSpectrum(
            quantity="collision_coefficient",
            value=self.sigma_a.value + self._sigma_s.value
        )

    def phase(self):
        return {f"phase_{self.id}": {"type": "rayleigh"}}

    def media(self, ref=False):
        if ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = self.phase()[f"phase_{self.id}"]

        return {
            f"medium_{self.id}": {
                "type": "homogeneous",
                "phase": phase,
                "sigma_t": onedict_value(self._sigma_t.kernel_dict()),
                "albedo": onedict_value(self._albedo.kernel_dict()),
            }
        }

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f

        if ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.media(ref=False)[f"medium_{self.id}"]

        k_length = uck.get("length")
        k_width = self.kernel_width.to(k_length).magnitude
        k_height = self.kernel_height.to(k_length).magnitude
        k_offset = self.kernel_offset.to(k_length).magnitude

        return {
            f"shape_{self.id}": {
                "type":
                    "cube",
                "to_world":
                    ScalarTransform4f([
                        [0.5 * k_width, 0., 0., 0.],
                        [0., 0.5 * k_width, 0., 0.],
                        [0., 0., 0.5 * k_height, 0.5 * k_height - k_offset],
                        [0., 0., 0., 1.],
                    ]),
                "bsdf": {
                    "type": "null"
                },
                "interior":
                    medium
            }
        }
