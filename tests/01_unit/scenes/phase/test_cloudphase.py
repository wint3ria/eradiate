from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eradiate.scenes.phase._cloudphase import (
    CloudPhaseFunction,
    _corners_on_union,
    _gather_corners,
    _interp_nearest_rv_linear_w,
    _interp_trilinear,
    _w_brackets,
    format_cloudparticles_dataset,
    interpolate_cloudparticles_profile,
)
from eradiate.units import unit_registry as ureg


def _make_iprt_ds(nlam=2, nreff=3, nthetamax=5, nphamat=4):
    """
    Minimal IPRT standard Mie particles property dataset.

    theta is NaN-padded and stored descending (180 → 0), matching the real
    format.  The dataset has no nveff dimension — that is an external
    parameter supplied to format_cloudparticles_dataset.
    """
    rng = np.random.default_rng(0)
    wavelen = np.linspace(0.5, 1.0, nlam)
    reff = np.linspace(1.0, 10.0, nreff)
    theta = np.tile(
        np.linspace(180.0, 0.0, nthetamax), (nlam, nreff, nphamat, 1)
    ).astype(np.float32)
    phase = rng.uniform(0.1, 1.0, (nlam, nreff, nphamat, nthetamax)).astype(np.float32)
    phase[:, :, 0, :] = np.abs(phase[:, :, 0, :]) + 1.0
    ext = rng.uniform(0.01, 0.1, (nlam, nreff))
    ssa = rng.uniform(0.5, 1.0, (nlam, nreff))
    return xr.Dataset(
        coords={
            "nlam": ("nlam", np.arange(nlam)),
            "nreff": ("nreff", np.arange(nreff)),
            "nthetamax": ("nthetamax", np.arange(nthetamax)),
            "nphamat": ("nphamat", np.arange(nphamat)),
        },
        data_vars={
            "wavelen": ("nlam", wavelen),
            "reff": ("nreff", reff),
            "theta": (["nlam", "nreff", "nphamat", "nthetamax"], theta),
            "phase": (["nlam", "nreff", "nphamat", "nthetamax"], phase),
            "ext": (["nlam", "nreff"], ext),
            "ssa": (["nlam", "nreff"], ssa),
        },
    )


def _make_cloud_ds(nlam=2, nreff=3, nveff=1, ntheta=5):
    """
    Compact cloud-particle dataset as returned by format_cloudparticles_dataset,
    built directly without going through the IPRT fixture so tests of the
    interpolation helpers are independent of format_cloudparticles_dataset.
    """
    rng = np.random.default_rng(1)
    w = np.linspace(0.5, 1.0, nlam)
    r = np.linspace(1.0, 10.0, nreff)
    v = np.linspace(0.01, 0.1, nveff)
    E = nlam * nreff * nveff
    theta_native = np.tile(np.linspace(0.0, 180.0, ntheta), E)
    phase_native = rng.uniform(0.0, 1.0, (E * ntheta, 16)).astype(np.float64)
    phase_native[:, 0] = np.abs(phase_native[:, 0]) + 1.0
    start = np.arange(E, dtype=np.int64).reshape(nlam, nreff, nveff) * ntheta
    n_pts = np.full((nlam, nreff, nveff), ntheta, dtype=np.int32)
    mext = rng.uniform(0.01, 0.1, (nlam, nreff, nveff))
    albedo = rng.uniform(0.5, 1.0, (nlam, nreff, nveff))
    return xr.Dataset(
        coords={
            "w": ("w", w, {"units": "micron"}),
            "r_eff": ("r_eff", r),
            "v_eff": ("v_eff", v),
            "rho": 1.0,
            "alpha": 7.0,
        },
        data_vars={
            "theta_native": ("total_pts", theta_native),
            "phase_native": (["total_pts", "ch16"], phase_native),
            "start": (["w", "r_eff", "v_eff"], start),
            "n_pts": (["w", "r_eff", "v_eff"], n_pts),
            "m_extinction": (["w", "r_eff", "v_eff"], mext),
            "albedo": (["w", "r_eff", "v_eff"], albedo),
        },
    )


def _make_profile(r_eff, v_eff):
    return xr.Dataset(
        {"r_eff": ("index", np.atleast_1d(r_eff)), "v_eff": ("index", np.atleast_1d(v_eff))}
    )


def _make_cpf(icp):
    return CloudPhaseFunction(
        interpolated_cloudproperties=icp,
        spatial_index=lambda ctx: np.zeros((2, 2, 2), dtype=np.int32),
        grid=type("G", (), {"to_world": None})(),
    )


class _FakeCtx:
    class si:
        w = 0.7 * ureg.micron


def _w_brackets_for(cloud_ds, n_wav=2):
    w_axis = cloud_ds.w.values
    wavs = np.linspace(w_axis[0] + 0.05, w_axis[-1] - 0.05, n_wav)
    return _w_brackets(w_axis, wavs), n_wav


def test_format_output_variables_present():
    out = format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05])
    for var in ("theta_native", "phase_native", "start", "n_pts", "m_extinction", "albedo"):
        assert var in out


def test_format_output_coordinates_present():
    out = format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05])
    for coord in ("w", "r_eff", "v_eff"):
        assert coord in out.coords


def test_format_w_values_match_iprt():
    iprt_ds = _make_iprt_ds(nlam=3)
    out = format_cloudparticles_dataset(iprt_ds, v_eff=[0.05])
    np.testing.assert_array_equal(out.w.values, iprt_ds.wavelen.values)


def test_format_reff_values_match_iprt():
    iprt_ds = _make_iprt_ds(nreff=4)
    out = format_cloudparticles_dataset(iprt_ds, v_eff=[0.05])
    np.testing.assert_array_equal(out.r_eff.values, iprt_ds.reff.values)


def test_format_theta_ascending_no_nan():
    out = format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05])
    theta = out.theta_native.values
    assert not np.any(np.isnan(theta))
    start = out.start.values.ravel()
    n_pts = out["n_pts"].values.ravel()
    for s, n in zip(start, n_pts):
        assert np.all(np.diff(theta[s : s + n]) >= 0)


def test_format_start_and_n_pts_shape():
    nlam, nreff, nveff, nthetamax = 2, 3, 2, 5
    out = format_cloudparticles_dataset(
        _make_iprt_ds(nlam=nlam, nreff=nreff, nthetamax=nthetamax), v_eff=[0.05, 0.10]
    )
    assert out.start.shape == (nlam, nreff, nveff)
    assert out["n_pts"].shape == (nlam, nreff, nveff)


def test_format_total_pts_consistent_with_n_pts():
    out = format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05])
    assert len(out.theta_native) == int(out["n_pts"].values.sum())


def test_format_spherical_mueller_symmetry():
    out = format_cloudparticles_dataset(
        _make_iprt_ds(nlam=1, nreff=1, nthetamax=4, nphamat=4),
        v_eff=[0.05],
        particle_shape="spherical",
    )
    ph = out.phase_native.values
    np.testing.assert_allclose(ph[:, 5], ph[:, 0])
    np.testing.assert_allclose(ph[:, 15], ph[:, 10])


def test_format_spheroidal_phase_shape():
    out = format_cloudparticles_dataset(
        _make_iprt_ds(nlam=1, nreff=1, nthetamax=4, nphamat=6),
        v_eff=[0.05],
        particle_shape="spheroidal",
    )
    assert out.phase_native.shape[1] == 16


def test_format_unknown_particle_shape_raises():
    with pytest.raises(NotImplementedError, match="not implemented"):
        format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05], particle_shape="cubic")


def test_format_multi_veff():
    out = format_cloudparticles_dataset(_make_iprt_ds(nlam=2, nreff=2, nthetamax=5), v_eff=[0.02, 0.05, 0.10])
    assert out.v_eff.shape == (3,)


def test_format_rho_alpha_scalar_coords():
    out = format_cloudparticles_dataset(_make_iprt_ds(), v_eff=[0.05])
    assert "rho" in out.coords
    assert "alpha" in out.coords


def test_w_brackets_midpoint_weights():
    w_flat, cw_flat = _w_brackets(np.array([0.0, 1.0, 2.0]), np.array([0.5]))
    assert w_flat[0] == 0 and w_flat[1] == 1
    np.testing.assert_allclose(cw_flat, [0.5, 0.5])


def test_w_brackets_exact_node_weight():
    _, cw_flat = _w_brackets(np.array([0.0, 1.0, 2.0]), np.array([1.0]))
    np.testing.assert_allclose(cw_flat[0] + cw_flat[1], 1.0)


def test_w_brackets_clamp_below():
    w_flat, _ = _w_brackets(np.array([1.0, 2.0, 3.0]), np.array([0.0]))
    assert w_flat[0] == 0


def test_w_brackets_clamp_above():
    w_flat, _ = _w_brackets(np.array([1.0, 2.0, 3.0]), np.array([5.0]))
    assert w_flat[1] == 2


def test_w_brackets_output_shape_multi_wavelength():
    w_flat, cw_flat = _w_brackets(np.linspace(0.4, 1.0, 10), np.array([0.5, 0.7, 0.9]))
    assert w_flat.shape == (6,)
    assert cw_flat.shape == (6,)


def test_w_brackets_weights_sum_to_one():
    _, cw_flat = _w_brackets(np.linspace(0.4, 1.0, 10), np.array([0.5, 0.6, 0.8]))
    np.testing.assert_allclose(cw_flat.reshape(-1, 2).sum(axis=1), 1.0)


def test_w_brackets_duplicate_node_no_nan():
    _, cw_flat = _w_brackets(np.array([0.5, 0.5, 1.0]), np.array([0.5]))
    assert not np.any(np.isnan(cw_flat))


def test_corners_on_union_output_shape():
    rng = np.random.default_rng(42)
    thetas = [np.sort(rng.uniform(0, 180, 5)) for _ in range(3)]
    phases = [rng.uniform(0, 1, (5, 16)) for _ in range(3)]
    union = np.unique(np.concatenate(thetas))
    out = _corners_on_union(thetas, phases, union)
    assert out.shape == (3, 16, len(union))


def test_corners_on_union_exact_node_values():
    theta = np.array([0.0, 90.0, 180.0])
    phase = np.ones((3, 16))
    out = _corners_on_union([theta], [phase], theta)
    np.testing.assert_allclose(out[0], np.ones((16, 3)), atol=1e-6)


def test_corners_on_union_midpoint_interpolation():
    theta = np.array([0.0, 180.0])
    phase = np.zeros((2, 16))
    phase[1, :] = 1.0
    out = _corners_on_union([theta], [phase], np.array([0.0, 90.0, 180.0]))
    np.testing.assert_allclose(out[0, :, 1], 0.5, atol=1e-6)


def test_corners_on_union_output_dtype():
    rng = np.random.default_rng(42)
    thetas = [np.sort(rng.uniform(0, 180, 5)) for _ in range(2)]
    phases = [rng.uniform(0, 1, (5, 16)) for _ in range(2)]
    union = np.unique(np.concatenate(thetas))
    assert _corners_on_union(thetas, phases, union).dtype == np.float32


def _make_gather_inputs(nlam=2, nreff=2, nveff=2, ntheta=4):
    E = nlam * nreff * nveff
    start = np.arange(E, dtype=np.int64).reshape(nlam, nreff, nveff) * ntheta
    n_pts = np.full((nlam, nreff, nveff), ntheta, dtype=np.int32)
    theta_native = np.linspace(0, 180, E * ntheta)
    phase_native = np.ones((E * ntheta, 16))
    return start, n_pts, theta_native, phase_native


def test_gather_corners_count_trilinear():
    start, n_pts, theta_native, phase_native = _make_gather_inputs()
    ct, cp = _gather_corners(np.array([0, 1]), [0, 1], [0, 1], start, n_pts, theta_native, phase_native)
    assert len(ct) == 8


def test_gather_corners_count_nearest():
    start, n_pts, theta_native, phase_native = _make_gather_inputs(nveff=1)
    ct, cp = _gather_corners(np.array([0, 1]), [0], [0], start, n_pts, theta_native, phase_native)
    assert len(ct) == 2


def test_gather_corners_slice_length():
    ntheta = 6
    start, n_pts, theta_native, phase_native = _make_gather_inputs(ntheta=ntheta, nveff=1)
    ct, _ = _gather_corners(np.array([0, 1]), [0, 1], [0], start, n_pts, theta_native, phase_native)
    for th in ct:
        assert len(th) == ntheta


def test_interp_trilinear_output_shapes():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    gs, gl, theta, phase, mext, alb = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=8)
    assert gs.shape == (3, n_wav)
    assert gl.shape == (3, n_wav)
    assert mext.shape == (3, n_wav)
    assert alb.shape == (3, n_wav)


def test_interp_nearest_output_shapes():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    gs, gl, theta, phase, mext, alb = _interp_nearest_rv_linear_w(ds, r, v, w_flat, cw_flat, n_wav)
    assert gs.shape == (3, n_wav)
    assert gl.shape == (3, n_wav)


def test_interp_trilinear_grid_indices_in_range():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    gs, gl, theta, phase, mext, alb = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=8)
    assert np.all(gs >= 0)
    assert np.all(gl > 0)
    assert np.all(gs + gl <= len(theta))


def test_interp_trilinear_p11_non_negative():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    gs, gl, theta, phase, mext, alb = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=8)
    for i in range(len(r)):
        for k in range(n_wav):
            s, n = int(gs[i, k]), int(gl[i, k])
            assert np.all(phase[s : s + n, 0] >= 0.0)


def test_interp_trilinear_mext_alb_finite():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    _, _, _, _, mext, alb = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=8)
    assert np.all(np.isfinite(mext))
    assert np.all(np.isfinite(alb))


def test_interp_trilinear_batch_size_invariance():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([2.5, 5.0, 8.0])
    v = np.array([0.03, 0.07, 0.09])
    (w_flat, cw_flat), n_wav = _w_brackets_for(ds)
    gs1, gl1, _, _, mext1, alb1 = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=1)
    gs2, gl2, _, _, mext2, alb2 = _interp_trilinear(ds, r, v, w_flat, cw_flat, n_wav, batch_size=100)
    np.testing.assert_array_equal(gl1, gl2)
    np.testing.assert_allclose(mext1, mext2, atol=1e-10)
    np.testing.assert_allclose(alb1, alb2, atol=1e-10)


def test_interp_trilinear_nearest_agree_at_grid_nodes():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([ds.r_eff.values[1]])
    v = np.array([ds.v_eff.values[0]])
    w_flat, cw_flat = _w_brackets(ds.w.values, np.array([ds.w.values[1]]))
    gs_t, gl_t, _, phase_t, _, _ = _interp_trilinear(ds, r, v, w_flat, cw_flat, 1, batch_size=8)
    gs_n, gl_n, _, phase_n, _, _ = _interp_nearest_rv_linear_w(ds, r, v, w_flat, cw_flat, 1)
    s_t, n_t = int(gs_t[0, 0]), int(gl_t[0, 0])
    s_n, n_n = int(gs_n[0, 0]), int(gl_n[0, 0])
    np.testing.assert_allclose(phase_t[s_t : s_t + n_t], phase_n[s_n : s_n + n_n], atol=1e-5)


def test_interp_nearest_deduplicated_store():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    r = np.array([ds.r_eff.values[0] + 0.01, ds.r_eff.values[0] + 0.02])
    v = np.array([ds.v_eff.values[0], ds.v_eff.values[0]])
    w_flat, cw_flat = _w_brackets(ds.w.values, np.array([ds.w.values[1]]))
    gs, gl, _, _, _, _ = _interp_nearest_rv_linear_w(ds, r, v, w_flat, cw_flat, n_wav=1)
    np.testing.assert_array_equal(gs[0], gs[1])
    np.testing.assert_array_equal(gl[0], gl[1])


def test_interpolate_profile_output_variables():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1, 2]], ds.v_eff.values[[0, 0, 1]])
    out = interpolate_cloudparticles_profile(ds, profile, np.array([0.6, 0.8]) * ureg.micron)
    for var in ("theta", "phase", "grid_start", "grid_len", "m_extinction", "albedo"):
        assert var in out


def test_interpolate_profile_index_length():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1, 2]], ds.v_eff.values[[0, 0, 1]])
    out = interpolate_cloudparticles_profile(ds, profile, np.array([0.6, 0.8]) * ureg.micron)
    assert out.sizes["index"] == 3


def test_interpolate_profile_w_length():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1]], ds.v_eff.values[[0, 1]])
    out = interpolate_cloudparticles_profile(ds, profile, np.array([0.6, 0.7, 0.9]) * ureg.micron)
    assert out.sizes["w"] == 3


def test_interpolate_profile_ragged_index_validity():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1, 2]], ds.v_eff.values[[0, 0, 1]])
    out = interpolate_cloudparticles_profile(ds, profile, np.array([0.6, 0.8]) * ureg.micron)
    gs, gl = out.grid_start.values, out.grid_len.values
    assert np.all(gs >= 0)
    assert np.all(gl > 0)
    assert np.all(gs + gl <= len(out.theta))


def test_interpolate_profile_p11_non_negative():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1, 2]], ds.v_eff.values[[0, 0, 1]])
    out = interpolate_cloudparticles_profile(ds, profile, np.array([0.6, 0.8]) * ureg.micron)
    gs, gl, phase = out.grid_start.values, out.grid_len.values, out.phase.values
    for i in range(out.sizes["index"]):
        for k in range(out.sizes["w"]):
            s, n = int(gs[i, k]), int(gl[i, k])
            assert np.all(phase[s : s + n, 0] >= 0.0)


def test_interpolate_profile_nearest_mode():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile(ds.r_eff.values[[0, 1]], ds.v_eff.values[[0, 1]])
    out = interpolate_cloudparticles_profile(
        ds, profile, np.array([0.6, 0.8]) * ureg.micron, rv_mode="nearest_rv_linear_w"
    )
    assert "grid_start" in out


def test_interpolate_profile_modes_agree_at_grid_nodes():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    prof = _make_profile([ds.r_eff.values[1]], [ds.v_eff.values[0]])
    wavs = ds.w.values[1:2] * ureg.micron
    out_t = interpolate_cloudparticles_profile(ds, prof, wavs, rv_mode="trilinear")
    out_n = interpolate_cloudparticles_profile(ds, prof, wavs, rv_mode="nearest_rv_linear_w")
    s_t, n_t = int(out_t.grid_start.values[0, 0]), int(out_t.grid_len.values[0, 0])
    s_n, n_n = int(out_n.grid_start.values[0, 0]), int(out_n.grid_len.values[0, 0])
    np.testing.assert_allclose(
        out_t.phase.values[s_t : s_t + n_t],
        out_n.phase.values[s_n : s_n + n_n],
        atol=1e-5,
    )


def test_interpolate_profile_invalid_rv_mode_raises():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    profile = _make_profile([5.0], [0.05])
    with pytest.raises(ValueError, match="Unknown rv_mode"):
        interpolate_cloudparticles_profile(ds, profile, np.array([0.7]) * ureg.micron, rv_mode="bogus")


def test_interpolate_profile_scalar_coords_present():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    out = interpolate_cloudparticles_profile(
        ds, _make_profile([5.0], [0.05]), np.array([0.7]) * ureg.micron
    )
    assert "rho" in out.coords
    assert "alpha" in out.coords


def test_interpolate_profile_single_entry():
    ds = _make_cloud_ds(nlam=3, nreff=4, nveff=2, ntheta=8)
    out = interpolate_cloudparticles_profile(
        ds, _make_profile([5.0], [0.05]), np.array([0.6, 0.8]) * ureg.micron
    )
    assert out.sizes["index"] == 1


def test_resolve_dataset_passthrough_single_w():
    ds = _make_cloud_ds(nlam=1, nreff=2, nveff=1, ntheta=4)
    cpf = _make_cpf(ds)
    result = cpf._resolve_dataset(_FakeCtx())
    assert result is ds or result.sizes.get("w", 1) == 1


def test_resolve_dataset_callable_is_called():
    ds = _make_cloud_ds(nlam=1, nreff=2, nveff=1, ntheta=4)
    called = []

    def icp(ctx):
        called.append(ctx)
        return ds

    _make_cpf(icp)._resolve_dataset(_FakeCtx())
    assert len(called) == 1


def test_resolve_dataset_multi_w_reduces_to_single_slice():
    ds = _make_cloud_ds(nlam=4, nreff=2, nveff=1, ntheta=4)
    result = _make_cpf(ds)._resolve_dataset(_FakeCtx())
    assert result.sizes.get("w", 1) <= 1


def test_resolve_dataset_multi_w_selects_correct_wavelength():
    ds = _make_cloud_ds(nlam=4, nreff=2, nveff=1, ntheta=4)
    ctx = _FakeCtx()
    result = _make_cpf(ds)._resolve_dataset(ctx)
    w_units = ds.w.attrs["units"]
    target = ctx.si.w.m_as(w_units)
    expected = ds.w.values[np.argmin(np.abs(ds.w.values - target))]
    selected = float(result.w.values) if result.sizes.get("w", 1) == 1 else float(result.w)
    assert abs(selected - expected) < 1e-9
