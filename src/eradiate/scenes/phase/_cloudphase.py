from __future__ import annotations

from functools import lru_cache
from typing import Callable, Literal, Union

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import xarray as xr

from ._core import PhaseFunction
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...util.misc import cache_by_id
from ...units import unit_registry as ureg


def format_cloudparticles_dataset(
    iprt_ds: xr.Dataset,
    v_eff: np.ndarray,
    particle_shape: Literal["spherical", "spheroidal"] = "spherical",
) -> xr.Dataset:
    """
    Reindex an IPRT standard Mie particles property dataset and return a compact :class:`xarray.Dataset`.

    The 4-component phase matrix is expanded to the full :math:`4 \\times 4`
    Mueller form.  Scattering angles and phase data are stored as contiguous
    1-D arrays (``theta_native``, ``phase_native``) together with per-entry
    ``start`` / ``n_pts`` index arrays, so that the data for entry
    ``(iw, ir, iv)`` occupies a zero-copy, NaN-free slice.

    Parameters
    ----------
    iprt_ds : xr.Dataset
        IPRT standard Mie particles property dataset with dimensions
        ``nlam``, ``nreff``, ``nthetamax``, and ``nphamat``, and variables
        ``phase``, ``theta``, ``ext``, ``ssa``, ``wavelen``, and ``reff``.
    v_eff : array-like
        Effective variances for the size distribution :math:`[\\mu m^2]`.
    particle_shape : {"spherical", "spheroidal"}, optional
        Symmetry class of the scatterer.

        * ``"spherical"`` — 4 independent Mueller elements
          :math:`[m_{11}, m_{12}, m_{33}, m_{34}]`, with
          :math:`m_{22} = m_{11}` and :math:`m_{44} = m_{33}`.
        * ``"spheroidal"`` — 6 independent Mueller elements
          :math:`[m_{11}, m_{12}, m_{22}, m_{33}, m_{34}, m_{44}]`.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates ``w``, ``r_eff``, ``v_eff``, ``rho``, and
        ``alpha``; and data variables ``theta_native``, ``phase_native``,
        ``start``, ``n_pts``, ``m_extinction``, and ``albedo``.

    Raises
    ------
    NotImplementedError
        If *particle_shape* is not ``"spherical"`` or ``"spheroidal"``.
    """
    nveff = len(v_eff)
    nlam = len(iprt_ds.nlam)
    nreff = len(iprt_ds.nreff)
    ntheta = len(iprt_ds.nthetamax)
    E = nlam * nreff * nveff

    phase_raw = iprt_ds.phase.values  # (nlam, nreff, 4, ntheta)
    phase_np = np.zeros((nlam, nreff, 4, 4, ntheta))

    if particle_shape == "spherical":
        for k in range(nlam):
            phase_np[k, :, 0, 0, :] = phase_raw[k, :, 0]
            phase_np[k, :, 1, 1, :] = phase_raw[k, :, 0]
            phase_np[k, :, 0, 1, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 0, :] = phase_raw[k, :, 1]
            phase_np[k, :, 2, 2, :] = phase_raw[k, :, 2]
            phase_np[k, :, 3, 3, :] = phase_raw[k, :, 2]
            phase_np[k, :, 2, 3, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 2, :] = phase_raw[k, :, 3]
    elif particle_shape == "spheroidal":
        for k in range(nlam):
            phase_np[k, :, 0, 0, :] = phase_raw[k, :, 0]
            phase_np[k, :, 0, 1, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 0, :] = phase_raw[k, :, 1]
            phase_np[k, :, 1, 1, :] = phase_raw[k, :, 4]
            phase_np[k, :, 2, 2, :] = phase_raw[k, :, 2]
            phase_np[k, :, 2, 3, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 2, :] = phase_raw[k, :, 3]
            phase_np[k, :, 3, 3, :] = phase_raw[k, :, 5]
    else:
        raise NotImplementedError(
            f"Particle shape '{particle_shape}' is not implemented. "
            "Use 'spherical' or 'spheroidal'."
        )

    theta_raw = iprt_ds.theta.isel(nphamat=0).values.reshape(nlam, nreff, nveff, ntheta)
    phase_4d = phase_np.reshape(nlam, nreff, nveff, 4, 4, ntheta)

    theta_flat = theta_raw.reshape(E, ntheta)
    phase_flat = phase_4d.reshape(E, 16, ntheta)

    valid_mask = ~np.isnan(theta_flat)
    valid_counts = valid_mask.sum(axis=1)
    total_pts = int(valid_counts.sum())

    start_flat = np.zeros(E, dtype=np.int64)
    start_flat[1:] = np.cumsum(valid_counts[:-1])

    theta_native = np.empty(total_pts, dtype=np.float64)
    phase_native = np.empty((total_pts, 16), dtype=np.float64)

    for e in range(E):
        nc = int(valid_counts[e])
        s = int(start_flat[e])
        valid_idx = np.where(valid_mask[e])[0]
        sidx = np.argsort(theta_flat[e, valid_idx])
        sorted_idx = valid_idx[sidx]
        theta_native[s : s + nc] = theta_flat[e, sorted_idx]
        phase_native[s : s + nc, :] = phase_flat[e, :, sorted_idx]

    start = start_flat.reshape(nlam, nreff, nveff).astype(np.int64)
    n_pts = valid_counts.reshape(nlam, nreff, nveff).astype(np.int32)

    return xr.Dataset(
        coords=dict(
            w=(
                ["w"],
                iprt_ds.wavelen.values,
                {"long_name": "wavelength", "units": "micron"},
            ),
            r_eff=(
                ["r_eff"],
                iprt_ds.reff.values,
                {"long_name": "effective_radius", "units": "micron"},
            ),
            v_eff=(
                ["v_eff"],
                v_eff,
                {"long_name": "effective_variance", "units": "micron ** 2"},
            ),
            i=(["i"], list(range(4))),
            j=(["j"], list(range(4))),
            rho=(["rho"], [1.0], {"long_name": "density", "units": "g/cm^3"}),
            alpha=(["v_eff"], [2.0]),
        ),
        data_vars=dict(
            theta_native=(["total_pts"], theta_native),
            phase_native=(["total_pts", "ch16"], phase_native),
            start=(["w", "r_eff", "v_eff"], start),
            n_pts=(["w", "r_eff", "v_eff"], n_pts),
            m_extinction=(
                ["w", "r_eff", "v_eff"],
                iprt_ds.ext.values.reshape(nlam, nreff, nveff),
            ),
            albedo=(
                ["w", "r_eff", "v_eff"],
                iprt_ds.ssa.values.reshape(nlam, nreff, nveff),
            ),
        ),
    ).squeeze(dim="rho")


def _w_brackets(
    w_axis: np.ndarray, wavelengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute linear-interpolation bracket indices and weights along the
    wavelength axis.

    Parameters
    ----------
    w_axis : ndarray
        Sorted 1-D array of dataset wavelengths.
    wavelengths : ndarray
        Query wavelengths (same units as *w_axis*).

    Returns
    -------
    w_flat : ndarray of int, shape (2 * n_wav,)
        Interleaved lower/upper bracket indices for each query wavelength.
    cw_flat : ndarray of float, shape (2 * n_wav,)
        Corresponding interpolation weights (lower, upper) for each wavelength.
    """
    il_w = np.clip(
        np.searchsorted(w_axis, wavelengths, side="right") - 1,
        0,
        len(w_axis) - 2,
    )
    iu_w = il_w + 1
    tw = np.where(
        w_axis[iu_w] == w_axis[il_w],
        0.0,
        (wavelengths - w_axis[il_w]) / (w_axis[iu_w] - w_axis[il_w]),
    )
    w_flat = np.stack([il_w, iu_w], axis=1).ravel()
    cw_flat = np.stack([1 - tw, tw], axis=1).ravel()
    return w_flat, cw_flat


def _corners_on_union(
    corner_thetas: list[np.ndarray],
    corner_phases: list[np.ndarray],
    union_theta: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate each corner's phase onto a shared *union_theta* grid.

    A *corner* is one vertex of the interpolation hypercube in the
    ``(w, r_eff, v_eff)`` parameter space (see :func:`_gather_corners`).
    Because each corner may carry its own irregular theta grid, the phase
    values cannot be mixed directly.  This function maps every corner onto
    the common *union_theta* grid — the sorted union of all corner grids —
    by piecewise-linear interpolation along the theta axis, enabling
    subsequent weighted summation across corners.

    Parameters
    ----------
    corner_thetas : list of ndarray
        Per-corner theta arrays (degrees, ascending).
    corner_phases : list of ndarray, shape (n_pts_corner, 16)
        Per-corner flattened :math:`4 \\times 4` Mueller matrix values.
    union_theta : ndarray
        Union of all corner theta grids, sorted ascending.

    Returns
    -------
    ndarray, shape (n_corners, 16, n_union)
        Phase matrix values for every corner evaluated on *union_theta*.
    """
    out = np.empty((len(corner_thetas), 16, len(union_theta)), dtype=np.float32)
    for ci, (th, ph) in enumerate(zip(corner_thetas, corner_phases)):
        il = np.clip(
            np.searchsorted(th, union_theta, side="right") - 1, 0, len(th) - 2
        )
        iu = il + 1
        denom = np.where(th[iu] == th[il], 1.0, th[iu] - th[il])
        t = (union_theta - th[il]) / denom
        out[ci] = (ph[il] * (1 - t[:, None]) + ph[iu] * t[:, None]).T
    return out


def _gather_corners(
    w_flat: np.ndarray,
    ir_list: list[int],
    iv_list: list[int],
    start_np: np.ndarray,
    n_pts_np: np.ndarray,
    theta_native: np.ndarray,
    phase_native: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Collect raw theta and phase slices for a set of ``(w, r_eff, v_eff)``
    corner indices.

    In the trilinear interpolation scheme the phase function at a query point
    ``(w*, r*, v*)`` is approximated by a weighted sum over the :math:`2^3 = 8`
    corners of the axis-aligned cell that contains the query point in the
    ``(w, r_eff, v_eff)`` parameter space.  Each corner is uniquely identified
    by a triplet of integer indices ``(iw, ir, iv)`` into the dataset's
    coordinate axes.  This function retrieves the raw (theta, phase) ragged
    slices for every such corner so they can be mapped onto a shared theta
    grid by :func:`_corners_on_union`.

    Parameters
    ----------
    w_flat : ndarray of int
        Wavelength bracket indices (lower and upper for each query wavelength,
        interleaved as produced by :func:`_w_brackets`).
    ir_list : list of int
        Lower and upper ``r_eff`` bracket indices for the current cell.
    iv_list : list of int
        Lower and upper ``v_eff`` bracket indices for the current cell.
    start_np : ndarray, shape (nlam, nreff, nveff)
        Offset of each entry in the flat ragged store.
    n_pts_np : ndarray, shape (nlam, nreff, nveff)
        Number of valid theta points for each entry.
    theta_native : ndarray
        Flat 1-D theta store (degrees, ascending).
    phase_native : ndarray, shape (total_pts, 16)
        Flat 1-D phase store (flattened :math:`4 \\times 4` Mueller matrix).

    Returns
    -------
    corner_thetas : list of ndarray
        Theta arrays for every corner, in ``(iw, ir, iv)`` iteration order.
    corner_phases : list of ndarray
        Phase arrays for every corner, shape ``(n_pts, 16)``.
    """
    corner_thetas, corner_phases = [], []
    for iw in w_flat:
        for ir in ir_list:
            for iv in iv_list:
                s = int(start_np[iw, ir, iv])
                nc = int(n_pts_np[iw, ir, iv])
                corner_thetas.append(theta_native[s : s + nc])
                corner_phases.append(phase_native[s : s + nc])
    return corner_thetas, corner_phases


def _interp_trilinear(
    ds: xr.Dataset,
    r_eff_all: np.ndarray,
    v_eff_all: np.ndarray,
    w_flat: np.ndarray,
    cw_flat: np.ndarray,
    n_wav: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trilinear interpolation in ``(w, r_eff, v_eff)`` with per-entry
    theta-grid unioning.

    For each query entry the phase function is reconstructed at *n_wav*
    wavelengths by trilinear interpolation over the :math:`2^3 = 8` corners
    of the axis-aligned cell that encloses the query point in the
    ``(w, r_eff, v_eff)`` parameter space.  Because every corner carries its
    own irregular theta grid, all eight corner grids are first merged into a
    single *union theta* grid per wavelength via :func:`_corners_on_union`;
    the corners are then linearly interpolated onto that common grid before
    the trilinear weight tensor is applied.

    Entries that share the same ``(r_eff, v_eff)`` bracket cell are collected
    into a *group* and processed together: the union theta grid and the
    corner phase arrays are computed once per group per wavelength, and the
    per-entry weight application is batched in mini-batches of *batch_size*
    rows for memory efficiency.

    The resulting interpolated phase and theta arrays are written into a pair
    of flat ragged arrays in contiguous memory.  The 2-D index arrays
    ``grid_start`` and ``grid_len`` (shape ``(N, n_wav)``) map each
    ``(entry, wavelength)`` pair to its slice in those flat stores:
    ``theta[grid_start[i, k] : grid_start[i, k] + grid_len[i, k]]`` and the
    corresponding ``phase`` rows give the complete phase function for entry
    *i* at wavelength *k*.

    Parameters
    ----------
    ds : xr.Dataset
        Cloud-particle dataset as returned by
        :func:`format_cloudparticles_dataset`.
    r_eff_all : ndarray, shape (N,)
        Query effective radii.
    v_eff_all : ndarray, shape (N,)
        Query effective variances.
    w_flat : ndarray of int, shape (2 * n_wav,)
        Wavelength bracket indices from :func:`_w_brackets`.
    cw_flat : ndarray of float, shape (2 * n_wav,)
        Wavelength bracket weights from :func:`_w_brackets`.
    n_wav : int
        Number of query wavelengths.
    batch_size : int
        Number of entries to process per mini-batch within each group.

    Returns
    -------
    grid_start : ndarray of int64, shape (N, n_wav)
    grid_len : ndarray of int32, shape (N, n_wav)
    theta : ndarray of float64, shape (total_pts,)
    phase : ndarray of float32, shape (total_pts, 16)
    mext_out : ndarray of float64, shape (N, n_wav)
    alb_out : ndarray of float64, shape (N, n_wav)
    """
    r_axis = ds.r_eff.values
    v_axis = ds.v_eff.values
    nveff = len(v_axis)
    N = len(r_eff_all)

    il_r = np.clip(np.searchsorted(r_axis, r_eff_all), 0, len(r_axis) - 2)
    il_v = (
        np.minimum(np.searchsorted(v_axis, v_eff_all), nveff - 2)
        if nveff > 1
        else np.zeros(N, dtype=int)
    )
    iu_r = il_r + 1
    iu_v = np.clip(il_v + 1, 0, nveff - 1)

    denom_r = r_axis[iu_r] - r_axis[il_r]
    denom_v = v_axis[iu_v] - v_axis[il_v]
    tr = np.divide(
        r_eff_all - r_axis[il_r], denom_r, out=np.zeros(N), where=denom_r != 0.0
    )
    tv = np.divide(
        v_eff_all - v_axis[il_v], denom_v, out=np.zeros(N), where=denom_v != 0.0
    )
    wr = np.stack([1 - tr, tr], axis=1)
    wv = np.stack([1 - tv, tv], axis=1)

    bracket_key = il_r * nveff + il_v
    sort_idx = np.argsort(bracket_key, kind="stable")
    boundaries = np.flatnonzero(np.diff(bracket_key[sort_idx])) + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [N]])

    grid_len = np.empty((N, n_wav), dtype=np.int32)
    mext_out = np.empty((N, n_wav))
    alb_out = np.empty((N, n_wav))

    group_wav_grids = []
    group_wav_phases = []
    group_rows_list = []

    for gs, ge in zip(group_starts, group_ends):
        rows = sort_idx[gs:ge]
        rep = rows[0]
        ir_pair = [int(il_r[rep]), int(iu_r[rep])]
        iv_pair = [int(il_v[rep]), int(iu_v[rep])]

        corners_mext = ds.m_extinction.values[np.ix_(w_flat, ir_pair, iv_pair)]
        corners_alb = ds.albedo.values[np.ix_(w_flat, ir_pair, iv_pair)]

        wav_union_thetas = []
        wav_corners_reg = []

        for k in range(n_wav):
            wf_k = w_flat[2 * k : 2 * k + 2]
            ct_k, cp_k = _gather_corners(
                wf_k,
                ir_pair,
                iv_pair,
                ds.start.values,
                ds["n_pts"].values,
                ds.theta_native.values,
                ds.phase_native.values,
            )
            union_theta_k = np.unique(np.concatenate(ct_k))
            wav_union_thetas.append(union_theta_k)
            wav_corners_reg.append(_corners_on_union(ct_k, cp_k, union_theta_k))

        group_phase_wav = [
            np.empty((len(rows), 4, 4, len(wav_union_thetas[k])))
            for k in range(n_wav)
        ]

        for b in range(0, len(rows), batch_size):
            idx = rows[b : b + batch_size]
            B = len(idx)

            cr_b = wr[idx]
            cv_b = wv[idx]

            for k in range(n_wav):
                cw_k = cw_flat[2 * k : 2 * k + 2]
                n_union_k = len(wav_union_thetas[k])
                cp_mat_k = wav_corners_reg[k].reshape(8, 16 * n_union_k)

                W_k = (
                    cw_k[np.newaxis, :, np.newaxis, np.newaxis]
                    * cr_b[:, np.newaxis, :, np.newaxis]
                    * cv_b[:, np.newaxis, np.newaxis, :]
                )

                mext_out[idx, k] = np.einsum(
                    "bfrs,frs->b",
                    W_k,
                    corners_mext[2 * k : 2 * k + 2],
                    optimize=True,
                )
                alb_out[idx, k] = np.einsum(
                    "bfrs,frs->b",
                    W_k,
                    corners_alb[2 * k : 2 * k + 2],
                    optimize=True,
                )

                W_flat_k = W_k.reshape(B, 8)
                group_phase_wav[k][b : b + B] = (
                    (W_flat_k @ cp_mat_k).reshape(B, 4, 4, n_union_k)
                )

        for k in range(n_wav):
            group_phase_wav[k][:, 0, 0, :] = np.maximum(
                group_phase_wav[k][:, 0, 0, :], 0.0
            )
            grid_len[rows, k] = len(wav_union_thetas[k])

        group_wav_grids.append(wav_union_thetas)
        group_wav_phases.append((rows, group_phase_wav))
        group_rows_list.append(rows)

    flat_lens = grid_len.ravel()
    flat_starts = np.zeros(N * n_wav, dtype=np.int64)
    flat_starts[1:] = np.cumsum(flat_lens[:-1])
    grid_start = flat_starts.reshape(N, n_wav)

    total_pts = int(flat_lens.sum())
    theta = np.empty(total_pts, dtype=np.float64)
    phase = np.empty((total_pts, 16), dtype=np.float32)

    for (rows, gph_wav), wav_grids in zip(group_wav_phases, group_wav_grids):
        for k in range(n_wav):
            gtheta = wav_grids[k]
            n_union = len(gtheta)
            gph_k = gph_wav[k]
            for local_i, row in enumerate(rows):
                s = int(grid_start[row, k])
                theta[s : s + n_union] = gtheta
                phase[s : s + n_union] = gph_k[local_i].reshape(16, n_union).T

    return grid_start, grid_len, theta, phase, mext_out, alb_out


def _interp_nearest_rv_linear_w(
    ds: xr.Dataset,
    r_eff_all: np.ndarray,
    v_eff_all: np.ndarray,
    w_flat: np.ndarray,
    cw_flat: np.ndarray,
    n_wav: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Nearest-neighbour lookup in ``(r_eff, v_eff)`` with linear interpolation
    in wavelength.

    Each query entry is mapped to the closest grid point in the
    ``(r_eff, v_eff)`` plane (no interpolation between corners); the phase
    function is then linearly interpolated between the two bounding wavelength
    nodes.  The result is written into flat ragged arrays with the same layout
    as :func:`_interp_trilinear`: entries that fall in the same
    ``(r_eff, v_eff)`` cell share a single copy of the union theta grid and
    phase data in the contiguous store, referenced by identical
    ``grid_start`` / ``grid_len`` values.

    The ``_rv`` suffix signals nearest-neighbour in ``(r_eff, v_eff)``;
    ``_linear_w`` signals that the wavelength dimension is still linearly
    interpolated.

    Parameters
    ----------
    ds : xr.Dataset
        Cloud-particle dataset as returned by
        :func:`format_cloudparticles_dataset`.
    r_eff_all : ndarray, shape (N,)
        Query effective radii.
    v_eff_all : ndarray, shape (N,)
        Query effective variances.
    w_flat : ndarray of int, shape (2 * n_wav,)
        Wavelength bracket indices from :func:`_w_brackets`.
    cw_flat : ndarray of float, shape (2 * n_wav,)
        Wavelength bracket weights from :func:`_w_brackets`.
    n_wav : int
        Number of query wavelengths.

    Returns
    -------
    grid_start : ndarray of int64, shape (N, n_wav)
    grid_len : ndarray of int32, shape (N, n_wav)
    theta : ndarray of float64, shape (total_pts,)
    phase : ndarray of float32, shape (total_pts, 16)
    mext_out : ndarray of float64, shape (N, n_wav)
    alb_out : ndarray of float64, shape (N, n_wav)
    """
    r_axis = ds.r_eff.values
    v_axis = ds.v_eff.values
    nveff = len(v_axis)
    N = len(r_eff_all)

    i_r = np.argmin(np.abs(r_axis[:, None] - r_eff_all[None, :]), axis=0)
    i_v = (
        np.argmin(np.abs(v_axis[:, None] - v_eff_all[None, :]), axis=0)
        if nveff > 1
        else np.zeros(N, dtype=int)
    )

    bracket_key = i_r * nveff + i_v
    sort_idx = np.argsort(bracket_key, kind="stable")
    boundaries = np.flatnonzero(np.diff(bracket_key[sort_idx])) + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [N]])

    mext_out = np.empty((N, n_wav))
    alb_out = np.empty((N, n_wav))
    group_wav_grids = []
    group_wav_phases = []
    group_rows_list = []

    for gs, ge in zip(group_starts, group_ends):
        rows = sort_idx[gs:ge]
        ir = int(i_r[rows[0]])
        iv = int(i_v[rows[0]])

        wav_union_thetas = []
        wav_phases = []

        for k in range(n_wav):
            wf_k = w_flat[2 * k : 2 * k + 2]
            cw_k = cw_flat[2 * k : 2 * k + 2]

            ct_k, cp_k = _gather_corners(
                wf_k,
                [ir],
                [iv],
                ds.start.values,
                ds["n_pts"].values,
                ds.theta_native.values,
                ds.phase_native.values,
            )
            union_theta_k = np.unique(np.concatenate(ct_k))
            reg_k = _corners_on_union(ct_k, cp_k, union_theta_k)

            phase_k = cw_k[0] * reg_k[0] + cw_k[1] * reg_k[1]
            phase_k = phase_k.reshape(4, 4, len(union_theta_k))
            phase_k[0, 0, :] = np.maximum(phase_k[0, 0, :], 0.0)

            corners_mext_k = ds.m_extinction.values[wf_k, ir, iv]
            corners_alb_k = ds.albedo.values[wf_k, ir, iv]
            mext_out[rows, k] = cw_k[0] * corners_mext_k[0] + cw_k[1] * corners_mext_k[1]
            alb_out[rows, k] = cw_k[0] * corners_alb_k[0] + cw_k[1] * corners_alb_k[1]

            wav_union_thetas.append(union_theta_k)
            wav_phases.append(phase_k)

        group_wav_grids.append(wav_union_thetas)
        group_wav_phases.append(wav_phases)
        group_rows_list.append(rows)

    n_groups = len(group_wav_grids)
    canon_lens = np.array(
        [
            [len(group_wav_grids[g][k]) for k in range(n_wav)]
            for g in range(n_groups)
        ],
        dtype=np.int32,
    )
    canon_flat = canon_lens.ravel()
    canon_starts = np.zeros(n_groups * n_wav, dtype=np.int64)
    canon_starts[1:] = np.cumsum(canon_flat[:-1])
    canon_starts = canon_starts.reshape(n_groups, n_wav)

    total_pts = int(canon_flat.sum())
    theta = np.empty(total_pts, dtype=np.float64)
    phase = np.empty((total_pts, 16), dtype=np.float32)

    for g, (wav_grids, wav_phases) in enumerate(
        zip(group_wav_grids, group_wav_phases)
    ):
        for k in range(n_wav):
            s = int(canon_starts[g, k])
            gtheta = wav_grids[k]
            n_union = len(gtheta)
            theta[s : s + n_union] = gtheta
            phase[s : s + n_union] = wav_phases[k].reshape(16, n_union).T

    grid_start = np.empty((N, n_wav), dtype=np.int64)
    grid_len = np.empty((N, n_wav), dtype=np.int32)
    for g, rows in enumerate(group_rows_list):
        grid_start[rows] = canon_starts[g]
        grid_len[rows] = canon_lens[g]

    return grid_start, grid_len, theta, phase, mext_out, alb_out


def interpolate_cloudparticles_profile(
    ds: xr.Dataset,
    cumulus_profile: xr.Dataset,
    wavelengths,
    batch_size: int = 16,
    rv_mode: Literal["trilinear", "nearest_rv_linear_w"] = "trilinear",
) -> xr.Dataset:
    """
    Interpolate a cloud-particle dataset onto a set of ``(r_eff, v_eff)``
    entries and query wavelengths.

    Each entry in *cumulus_profile* describes the microphysical state of one
    cloud element (e.g. a layer or a voxel) via its effective radius
    ``r_eff`` and effective variance ``v_eff``.  For every such entry and
    every query wavelength, this function produces the interpolated phase
    function, extinction coefficient, and single-scattering albedo by
    interpolating the Mie lookup table *ds* in the three-dimensional
    ``(w, r_eff, v_eff)`` parameter space.

    **Trilinear interpolation (** ``rv_mode="trilinear"`` **).**
    In this mode the interpolation is trilinear: it is linear in wavelength
    *and* bi-linear in the ``(r_eff, v_eff)`` plane, which together define a
    trilinear interpolation over the eight *corners* of the axis-aligned cell
    that encloses each query point.  A *corner* is one vertex of that cell,
    i.e. a specific ``(iw, ir, iv)`` triplet of grid indices pointing at a
    single pre-computed Mie entry in the lookup table.  Each corner owns a
    potentially different, irregularly spaced theta grid.  Rather than
    resampling all corners onto a fixed common grid up front, the algorithm
    first forms the *union* of all eight corner theta grids for each
    ``(entry, wavelength)`` pair, re-evaluates every corner phase onto that
    union grid by piecewise-linear interpolation, and then applies the
    trilinear weights.  This preserves the full angular detail of every
    corner without introducing smoothing artefacts from a coarser common
    grid.

    **Nearest-neighbour interpolation (** ``rv_mode="nearest_rv_linear_w"`` **).**
    In this mode only the closest ``(r_eff, v_eff)`` grid point is used;
    the wavelength dimension is still linearly interpolated.  Only two
    corners (the lower and upper wavelength bracket nodes at the nearest
    ``(r_eff, v_eff)`` point) are involved per entry.  This mode is faster
    and sufficient when the ``(r_eff, v_eff)`` grid is already dense.

    **Output layout — set of flat ragged arrays in contiguous memory.**
    Because different entries and wavelengths may yield union theta grids of
    different lengths, the phase and theta data cannot be stored in a regular
    array.  Instead they are packed into two flat 1-D arrays (``theta`` and
    ``phase``) laid out as a set of ragged arrays in contiguous memory.
    A pair of 2-D index arrays, ``grid_start`` and ``grid_len`` (both of
    shape ``(N, n_wav)``), map each ``(entry index, wavelength index)`` pair
    to its slice in those flat stores:

    .. code-block:: python

        s = grid_start[i, k]
        n = grid_len[i, k]
        theta_ik = theta[s : s + n]   # shape (n,)
        phase_ik = phase[s : s + n]   # shape (n, 16)

    Entries that fall in the same interpolation cell (nearest-neighbour
    mode) or the same bracket group (trilinear mode) point at the *same*
    flat slot, so the data is written only once per unique union grid in the
    deduplicated contiguous store.

    Parameters
    ----------
    ds : xr.Dataset
        Cloud-particle dataset as returned by
        :func:`format_cloudparticles_dataset`.
    cumulus_profile : xr.Dataset
        Dataset with ``r_eff`` and ``v_eff`` arrays of length *N* describing
        the microphysical state of each cloud element to interpolate.
    wavelengths : :class:`pint.Quantity`
        Query wavelengths.  Must be convertible to the unit stored in
        ``ds.w.attrs["units"]``.
    batch_size : int, optional
        Number of entries to process at once in the trilinear path.
        Larger values trade memory for speed.  Ignored in nearest-neighbour
        mode.
    rv_mode : {"trilinear", "nearest_rv_linear_w"}, optional
        Interpolation strategy in the ``(r_eff, v_eff)`` plane.

    Returns
    -------
    xr.Dataset
        Dataset with coordinates ``index``, ``w``, ``rho``, and ``alpha``
        and data variables ``theta``, ``phase``, ``grid_start``,
        ``grid_len``, ``m_extinction``, and ``albedo``.

        ``theta`` and ``phase`` are flat ragged arrays in contiguous memory,
        indexed via ``grid_start`` and ``grid_len`` as described above.

    Raises
    ------
    ValueError
        If *rv_mode* is not ``"trilinear"`` or ``"nearest_rv_linear_w"``.
    """
    wavelengths = np.atleast_1d(wavelengths)
    n_wav = len(wavelengths)

    w_units = ds.w.attrs["units"]
    w_axis_raw = (ds.w.values * ureg.Unit(w_units)).m_as(w_units)
    wav_raw = wavelengths.m_as(w_units)
    w_flat, cw_flat = _w_brackets(w_axis_raw, wav_raw)

    r_eff_all = cumulus_profile.r_eff.values
    v_eff_all = cumulus_profile.v_eff.values
    N = len(r_eff_all)

    if rv_mode == "trilinear":
        grid_start, grid_len, theta, phase, mext, alb = _interp_trilinear(
            ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav, batch_size
        )
    elif rv_mode == "nearest_rv_linear_w":
        grid_start, grid_len, theta, phase, mext, alb = _interp_nearest_rv_linear_w(
            ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav
        )
    else:
        raise ValueError(
            f"Unknown rv_mode {rv_mode!r}. Valid options are 'trilinear' and 'nearest_rv_linear_w'."
        )

    return xr.Dataset(
        coords=dict(
            index=(["index"], np.arange(N)),
            w=(["w"], wav_raw),
            rho=float(ds.rho.values),
            alpha=float(ds.alpha.values.mean()),
        ),
        data_vars=dict(
            theta=(["total_pts"], theta),
            phase=(["total_pts", "ch16"], phase),
            grid_start=(["index", "w"], grid_start),
            grid_len=(["index", "w"], grid_len),
            m_extinction=(["index", "w"], mext),
            albedo=(["index", "w"], alb),
        ),
    )


def _validate_particle_shape(instance, attribute, value):
    if value not in ("spherical", "spheroidal"):
        raise NotImplementedError(
            f"Particle shape '{value}' is not supported. "
            "Use 'spherical' or 'spheroidal'."
        )


@define(eq=False, slots=False)
class CloudPhaseFunction(PhaseFunction):
    r"""
    Cloud-particle phase function [``cloud_phase``].

    Heterogeneous, polarized phase function for cloud particles.  The phase
    matrix is stored as a set of flat ragged arrays (``theta``, ``phase``)
    in contiguous memory, indexed by the 2-D arrays ``grid_start`` and
    ``grid_len`` (shape ``(N, n_wav)``).  A separate volume grid
    (``index_volume``) provides an integer index per scene voxel that maps
    each kernel thread to the correct entry in those arrays.

    Notes
    -----
    * :attr:`interpolated_cloudproperties` can be either a callable
      ``(ctx) -> xr.Dataset`` or a pre-computed :class:`xarray.Dataset`
      in the layout produced by :func:`interpolate_cloudparticles_profile`.
      When a dataset is supplied directly it is used as-is without any
      further call.
    * :attr:`spatial_index` must return a 3-D :class:`numpy.ndarray` of
      integer indices referencing rows of the cloud properties dataset.
    * Scattering angles are converted from the libRadtran convention
      (:math:`\theta \in [0°, 180°]`, forward-scatter at :math:`0°`) to
      the Mitsuba convention (:math:`\mu = \cos(180° - \theta)`) during
      kernel parameter assembly.
    """

    interpolated_cloudproperties: Union[xr.Dataset, Callable] = documented(
        attrs.field(kw_only=True),
        doc="Cloud-particle properties dataset in the layout produced by "
        ":func:`interpolate_cloudparticles_profile`, or a callable "
        "``(ctx) -> xr.Dataset`` that returns such a dataset for the "
        "current rendering context.  When an :class:`xarray.Dataset` is "
        "passed directly it is used as-is without being called. "
        "This parameter has no default.",
        type="xr.Dataset or callable",
    )

    spatial_index: object = documented(
        attrs.field(kw_only=True),
        doc="Callable ``(ctx) -> ndarray`` returning a 3-D integer array "
        "whose values index into the entry dimension of the dataset "
        "provided by :attr:`interpolated_cloudproperties`.  Used to "
        "populate the ``index_volume`` grid that maps each scene voxel to "
        "its cloud-property entry. "
        "This parameter has no default.",
        type="callable",
    )

    grid: object = documented(
        attrs.field(kw_only=True),
        doc=":class:`.GridCoords` instance defining the spatial extent and "
        "orientation of the ``index_volume`` grid via its ``to_world`` "
        "transform. "
        "This parameter has no default.",
        type="GridCoords",
    )

    particle_shape: Literal["spherical", "spheroidal"] = documented(
        attrs.field(
            default="spherical",
            kw_only=True,
            validator=attrs.validators.in_(["spherical", "spheroidal"]),
        ),
        doc="Shape of the scattering particles.  Determines which symmetry "
        "relations are applied when expanding the Mueller matrix in "
        ":func:`.format_cloudparticles_dataset`.",
        type="str",
        init_type='{"spherical", "spheroidal"}',
        default='"spherical"',
    )

    def _resolve_dataset(self, ctx: object) -> xr.Dataset:
        """
        Return the cloud-properties dataset for the current rendering context,
        selecting the appropriate wavelength slice.

        :attr:`interpolated_cloudproperties` is first resolved to an
        :class:`xarray.Dataset`: if it is already a dataset it is used
        directly; if it is a callable it is called with *ctx*.

        The resolved dataset may carry a ``w`` dimension with multiple
        wavelength entries (one per query wavelength passed to
        :func:`interpolate_cloudparticles_profile`).  This method selects
        the single wavelength slice that matches the spectral index carried
        by *ctx* using nearest-neighbour lookup along the ``w`` axis.  If
        the ``w`` dimension has length 1 the dataset is returned as-is,
        avoiding an unnecessary selection step.

        Parameters
        ----------
        ctx : object
            Rendering context.  Must expose a ``si.w`` attribute carrying
            the current spectral wavelength as a :class:`pint.Quantity`.

        Returns
        -------
        xr.Dataset
            Dataset with the ``w`` dimension reduced to a single wavelength
            slice corresponding to ``ctx.si.w``, or the full dataset when
            ``w`` has length 1.
        """
        icp = self.interpolated_cloudproperties
        ds = icp if isinstance(icp, xr.Dataset) else icp(ctx)
        if ds.sizes["w"] == 1:
            return ds
        w_units = ds.w.attrs["units"]
        return ds.sel(w=ctx.si.w.m_as(w_units), method="nearest")

    @cache_by_id
    def _cloudphase_build_parameters(self, ctx: object) -> dict:
        """
        Assemble the kernel-ready parameter dictionary for the current
        rendering context.

        The cloud-properties dataset is resolved once via
        :meth:`_resolve_dataset`.  From it, six independent Mueller matrix
        elements are extracted from the flat phase store (columns 0, 1, 5,
        10, 11, 15 of the 16-element row-major flattened :math:`4 \\times 4`
        matrix, corresponding to
        :math:`[m_{11}, m_{12}, m_{22}, m_{33}, m_{34}, m_{44}]`), and
        scattering angles are converted from the libRadtran convention to the
        Mitsuba :math:`\\mu = \\cos(180° - \\theta)` convention.  The
        wavelength slice matching ``ctx.si.w`` is selected by
        :meth:`_resolve_dataset` before this method is called.

        Parameters
        ----------
        ctx : object
            Rendering context.

        Returns
        -------
        dict
            Dictionary with keys ``n_entries``, ``nodes``,
            ``phase_mueller``, ``grid_start``, and ``grid_len``.
        """
        ds = self._resolve_dataset(ctx)

        mueller = np.ascontiguousarray(
            ds.phase.values[:, [0, 1, 5, 10, 11, 15]].astype(np.float32)
        ).flatten()

        nodes = np.cos(np.deg2rad(180.0 - ds.theta.values)).astype(np.float32)

        w_idx = 0
        grid_start = ds.grid_start.values[:, w_idx].astype(np.int32)
        grid_len = ds.grid_len.values[:, w_idx].astype(np.int32)

        return {
            "n_entries": int(len(grid_start)),
            "nodes": dr.scalar.ArrayXf64(nodes),
            "phase_mueller": dr.scalar.ArrayXf64(mueller),
            "grid_start": dr.scalar.ArrayXu(grid_start),
            "grid_len": dr.scalar.ArrayXu(grid_len),
        }

    @property
    def template(self) -> dict:
        return {
            "type": "cloudphase",
            "index_volume": {
                "type": "gridvolume",
                "grid": DictParameter(
                    lambda ctx: mi.VolumeGrid(
                        self.spatial_index(ctx).astype(np.float32).T
                    )
                ),
                "filter_type": "nearest",
                "to_world": self.grid.to_world,
                "wrap_mode": "repeat",
            },
            "n_entries": DictParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["n_entries"]
            ),
            "nodes": DictParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["nodes"]
            ),
            "phase_mueller": DictParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["phase_mueller"]
            ),
            "grid_start": DictParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["grid_start"]
            ),
            "grid_len": DictParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["grid_len"]
            ),
        }

    @property
    def params(self) -> dict[str, SceneParameter]:
        return {
            "index_volume.grid": SceneParameter(
                lambda ctx: mi.VolumeGrid(
                    self.spatial_index(ctx).astype(np.float32)
                ),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "n_entries": SceneParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["n_entries"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "nodes": SceneParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["nodes"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "phase_mueller": SceneParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["phase_mueller"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_start": SceneParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["grid_start"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_len": SceneParameter(
                lambda ctx: self._cloudphase_build_parameters(ctx)["grid_len"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
        }

    def __hash__(self) -> int:
        return id(self)
