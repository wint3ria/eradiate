from typing import MutableMapping, Optional

import attr

import eradiate

from ._core import PhaseFunction, PhaseFunctionFactory
from ..spectra import Spectrum, SpectrumFactory, UniformSpectrum
from ... import validators
from ..._util import onedict_value
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...exceptions import UnsupportedModeError


@PhaseFunctionFactory.register("hg")
@parse_docs
@attr.s
class HenyeyGreensteinPhaseFunction(PhaseFunction):
    """
    The Henyey-Greenstein phase function :cite:`Henyey1941Diffuse` models
    scattering in an isotropic medium. The scattering pattern is controlled by
    its :math:`g` parameter, which is equal to the phase function's asymmetry
    parameter (the mean cosine of the scattering angle): a positive (resp.
    negative) value corresponds to predominant forward (resp. backward)
    scattering.
    """

    g = documented(
        attr.ib(
            default=0.0,
            converter=SpectrumFactory.converter("dimensionless"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("dimensionless"),
            ],
        ),
        doc="Asymmetry parameter. Must be dimensionless. "
        "Must be in :math:`]-1, 1[`.",
        type=":class:`.Spectrum`",
        default="0.0",
    )

    def kernel_dict(self, ctx: Optional[KernelDictContext] = None) -> MutableMapping:
        if eradiate.mode().is_monochromatic():
            # TODO: This is a workaround until the hg plugin accepts spectra for
            #  its g parameter
            g = float(onedict_value(self.g.kernel_dict(ctx=ctx))["value"])
            return {
                self.id: {
                    "type": "hg",
                    "g": g,
                }
            }
        else:
            raise UnsupportedModeError(supported="monochromatic")
