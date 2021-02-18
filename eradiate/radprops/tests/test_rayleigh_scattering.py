import numpy as np

from eradiate.radprops.rayleigh import (
    _LOSCHMIDT,
    air_refractive_index,
    compute_sigma_s_air
)
from eradiate import unit_registry as ureg

_Q = ureg.Quantity


def test_sigma_s_air():
    """Test computation of Rayleigh scattering coefficient for air with
    default values."""

    ref_cross_section = _Q(4.513e-27, "cm**2")
    ref_sigmas = ref_cross_section * _LOSCHMIDT
    expected = ref_sigmas

    # Compare to reference value computed from scattering cross section in
    # Bates (1984) Planetary and Space Science, Volume 32, No. 6.
    assert np.allclose(compute_sigma_s_air(number_density=_LOSCHMIDT), expected, rtol=1e-2)


def test_sigma_s_air_wavelength_dependence():
    """Test that the Rayleigh scattering coefficient scales with the 4th power
    of wavelength."""

    wavelength = np.linspace(240., 2400.)
    sigma_s = compute_sigma_s_air(wavelength)
    prod = sigma_s.magnitude * np.power(wavelength, 4)
    assert np.allclose(prod, prod[0], rtol=0.2)


def test_sigma_s_air_optical_thickness():
    """We compute the total optical thickness due to Rayleigh scattering by the
    air in a 100km high atmosphere. We compare the obtained result to the value
    given by :cite:`Hansen1974LightScatteringPlanetary` at page 544.
    """
    from eradiate.thermoprops.us76 import make_profile

    profile = make_profile(levels=np.linspace(0., 100000., 1001))
    n = profile.n_tot.values
    z = profile.z_level.values
    dz = z[1:] - z[:-1]
    sigma_s = compute_sigma_s_air(
        number_density=_Q(n, "m^-3"),
        depolarisation_ratio=0.031
    )
    optical_thickness = np.sum(sigma_s.to("m^-1").magnitude * dz)

    assert np.isclose(optical_thickness, 0.0973, rtol=1e-2)


def test_air_refractive_index():
    """Test computation of the air refractive index for different wavelength
    values. We compare the results with the values given in Table III of
    :cite:`Peck1972DispersionAir`."""

    wavelength = _Q(np.array([
        1.6945208,
        1.01425728,
        0.64402492,
        0.54622707,
        0.3889751,
        0.230289
    ]), "micrometer")

    indices = air_refractive_index(wavelength)

    # compute the refractivities in parts per 1e8
    results = (indices - 1) * 1e8

    expected = np.array([
        27314.19,
        27410.90,
        27638.092,
        27789.843,
        28336.843,
        30787.68
    ])
    assert np.allclose(results, expected, rtol=1e-5)
