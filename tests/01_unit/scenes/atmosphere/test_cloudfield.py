from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from eradiate.scenes.atmosphere._cloudfield import CloudField, _validate_particles_interp_method
from eradiate.units import unit_registry as ureg


def _make_profile(nx=2, ny=2, nz=4):
    rng = np.random.default_rng(0)
    x_levels = np.linspace(0.0, 10.0, nx + 1)
    y_levels = np.linspace(0.0, 10.0, ny + 1)
    z_levels = np.linspace(0.0, 4.0, nz + 1)

    ix = np.array([1, 1, 2])
    iy = np.array([1, 2, 1])
    iz = np.clip(np.array([1, 2, 3]), 1, nz)

    return xr.Dataset(
        data_vars=dict(
            i_x=(["index"], ix),
            i_y=(["index"], iy),
            i_z=(["index"], iz),
            r_eff=(["index"], rng.uniform(1.0, 10.0, len(ix)).astype(np.float32)),
            v_eff=(["index"], rng.uniform(0.01, 0.1, len(ix)).astype(np.float32)),
            extinction=(["index"], rng.uniform(0.01, 1.0, len(ix)).astype(np.float32)),
        ),
        coords=dict(
            x_levels=(["x"], x_levels),
            y_levels=(["y"], y_levels),
            z_levels=(["z"], z_levels),
        ),
    )


def _make_grid(nx=2, ny=2, nz=4):
    from eradiate.grid import PlaneParallelGridCoords
    return PlaneParallelGridCoords(
        edges_x=np.linspace(0.0, 10.0, nx + 1) * ureg.km,
        edges_y=np.linspace(0.0, 10.0, ny + 1) * ureg.km,
        levels=np.linspace(0.0, 4.0, nz + 1) * ureg.km,
    ).centered()



def _make_cloud_field_from_profile(profile, **kwargs):
    from eradiate.scenes.phase._cloudphase import format_cloudparticles_dataset
    from eradiate.scenes.geometry import PlaneParallelGeometry
    rng = np.random.default_rng(1)
    nlam, nreff, nveff, ntheta = 2, 3, 1, 8
    iprt_ds = xr.Dataset(
        coords={
            "nlam": ("nlam", np.arange(nlam)),
            "nreff": ("nreff", np.arange(nreff)),
            "nveff": ("nveff", np.array([0.05])),
            "nthetamax": ("nthetamax", np.arange(ntheta)),
            "nphamat": ("nphamat", np.arange(4)),
        },
        data_vars={
            "wavelen": ("nlam", np.linspace(0.5, 1.0, nlam)),
            "reff": ("nreff", np.linspace(1.0, 10.0, nreff)),
            "theta": (
                ["nlam", "nreff", "nveff", "nphamat", "nthetamax"],
                np.tile(np.linspace(0.0, 180.0, ntheta), (nlam, nreff, nveff, 4, 1)),
            ),
            "phase": (
                ["nlam", "nreff", "nveff", "nphamat", "nthetamax"],
                rng.uniform(0.1, 1.0, (nlam, nreff, nveff, 4, ntheta)),
            ),
            "ext": (["nlam", "nreff", "nveff"], rng.uniform(0.01, 0.1, (nlam, nreff, nveff))),
            "ssa": (["nlam", "nreff", "nveff"], rng.uniform(0.5, 1.0, (nlam, nreff, nveff))),
        },
    )
    properties = format_cloudparticles_dataset(iprt_ds, v_eff=[0.05])
    nz = len(profile.z_levels) - 1
    grid = _make_grid(nz=nz)
    geometry = PlaneParallelGeometry(
        grid=grid,
        toa_altitude=grid.levels[-1],
        width=10.0 * ureg.km,
    )
    return CloudField(profile=profile, properties=properties, geometry=geometry, **kwargs)


def _make_cloud_field(**kwargs):
    from eradiate.scenes.phase._cloudphase import format_cloudparticles_dataset
    rng = np.random.default_rng(1)
    nlam, nreff, nveff, ntheta = 2, 3, 1, 8
    iprt_ds = xr.Dataset(
        coords={
            "nlam": ("nlam", np.arange(nlam)),
            "nreff": ("nreff", np.arange(nreff)),
            "nveff": ("nveff", np.array([0.05])),
            "nthetamax": ("nthetamax", np.arange(ntheta)),
            "nphamat": ("nphamat", np.arange(4)),
        },
        data_vars={
            "wavelen": ("nlam", np.linspace(0.5, 1.0, nlam)),
            "reff": ("nreff", np.linspace(1.0, 10.0, nreff)),
            "theta": (
                ["nlam", "nreff", "nveff", "nphamat", "nthetamax"],
                np.tile(np.linspace(0.0, 180.0, ntheta), (nlam, nreff, nveff, 4, 1)),
            ),
            "phase": (
                ["nlam", "nreff", "nveff", "nphamat", "nthetamax"],
                rng.uniform(0.1, 1.0, (nlam, nreff, nveff, 4, ntheta)),
            ),
            "ext": (["nlam", "nreff", "nveff"], rng.uniform(0.01, 0.1, (nlam, nreff, nveff))),
            "ssa": (["nlam", "nreff", "nveff"], rng.uniform(0.5, 1.0, (nlam, nreff, nveff))),
        },
    )
    properties = format_cloudparticles_dataset(iprt_ds, v_eff=[0.05])
    profile = _make_profile()
    grid = _make_grid()

    from eradiate.scenes.geometry import PlaneParallelGeometry
    geometry = PlaneParallelGeometry(
        grid=grid,
        toa_altitude=grid.levels[-1],
        width=10.0 * ureg.km,
    )
    return CloudField(profile=profile, properties=properties, geometry=geometry, **kwargs)


def test_validate_particles_interp_method_raises():
    with pytest.raises(NotImplementedError, match="not supported"):
        _validate_particles_interp_method(None, None, "cubic")


def test_validate_particles_interp_method_valid():
    for val in ("linear", "nearest"):
        _validate_particles_interp_method(None, None, val)


def test_resample_profile_to_grid_output_variables():
    cf = _make_cloud_field()
    grid = _make_grid()
    out = cf._resample_profile_to_grid(grid)
    for var in ("i_x", "i_y", "i_z", "extinction", "r_eff", "v_eff"):
        assert var in out


def test_resample_profile_to_grid_indices_in_bounds():
    cf = _make_cloud_field()
    grid = _make_grid(nx=2, ny=2, nz=4)
    out = cf._resample_profile_to_grid(grid)
    assert np.all(out.i_x.values >= 1) and np.all(out.i_x.values <= grid.n_cells_x)
    assert np.all(out.i_y.values >= 1) and np.all(out.i_y.values <= grid.n_cells_y)
    assert np.all(out.i_z.values >= 1) and np.all(out.i_z.values <= grid.n_cells_z)


def test_check_grid_profile_compatibility_z_range_raises():
    cf = _make_cloud_field()
    from eradiate.grid import PlaneParallelGridCoords
    grid_out = PlaneParallelGridCoords(
        edges_x=np.linspace(0.0, 10.0, 3) * ureg.km,
        edges_y=np.linspace(0.0, 10.0, 3) * ureg.km,
        levels=np.linspace(0.0, 10.0, 6) * ureg.km,
    ).centered()
    with pytest.raises(ValueError, match="extends beyond"):
        cf._check_grid_profile_compatibility(grid_out)


def test_check_grid_profile_compatibility_coarse_z_warns():
    from eradiate.grid import PlaneParallelGridCoords
    fine_profile = _make_profile(nx=2, ny=2, nz=20)
    cf = _make_cloud_field_from_profile(fine_profile)
    grid_coarse = PlaneParallelGridCoords(
        edges_x=np.linspace(0.0, 10.0, 3) * ureg.km,
        edges_y=np.linspace(0.0, 10.0, 3) * ureg.km,
        levels=np.array([0.0, 2.0, 4.0]) * ureg.km,
    ).centered()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cf._check_grid_profile_compatibility(grid_coarse)
        assert any("coarser" in str(warning.message) for warning in w)


def test_profile_regular_grid_covers_profile_extent():
    cf = _make_cloud_field()
    grid = cf._profile_regular_grid()
    z_min = grid.levels[0].m_as(ureg.km)
    z_max = grid.levels[-1].m_as(ureg.km)
    assert np.isclose(z_min, cf.profile.z_levels.values[0])
    assert np.isclose(z_max, cf.profile.z_levels.values[-1])


def test_to_profile_regular_grid_returns_cloud_field():
    cf = _make_cloud_field()
    cf2 = cf.to_profile_regular_grid()
    assert isinstance(cf2, CloudField)
    assert cf2.geometry is not cf.geometry
