from __future__ import annotations

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import xarray as xr

from eradiate.attrs import define
from eradiate.kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from eradiate.scenes.phase import PhaseFunction
from eradiate.units import unit_registry as ureg


def format_cloudparticles_dataset(mie_ws, v_eff, particle_shape="spherical"):
    """
    Reindex a libRadtran Mie CDF file and return a xr.Dataset in a contiguous format.

    The 4-component phase matrix is expanded to the full 4x4 Mueller form.
    Theta and phase data are stored as contiguous 1-D arrays with start/count
    indices so that entry (iw, ir, iv) occupies a zero-copy slice with no
    NaN padding.
    """

    nveff = len(v_eff)
    nlam = len(mie_ws.nlam)
    nreff = len(mie_ws.nreff)
    ntheta = len(mie_ws.nthetamax)
    E = nlam * nreff * nveff

    phase_raw = mie_ws.phase.values  # (nlam, nreff, 4, ntheta)
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
        raise NotImplementedError(f"{particle_shape} particle shape is not implemented")

    theta_raw = mie_ws.theta.isel(nphamat=0).values.reshape(nlam, nreff, nveff, ntheta)
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
        mask = valid_mask[e]
        # sort ascending (theta stored descending in raw data)
        valid_idx = np.where(mask)[0]  # indices of valid theta
        sidx = np.argsort(theta_flat[e, valid_idx])  # sort ascending
        sorted_idx = valid_idx[sidx]
        theta_native[s : s + nc] = theta_flat[e, sorted_idx]
        phase_native[s : s + nc, :] = phase_flat[e, :, sorted_idx]

    start = start_flat.reshape(nlam, nreff, nveff).astype(np.int64)
    n_pts = valid_counts.reshape(nlam, nreff, nveff).astype(np.int32)

    return xr.Dataset(
        coords=dict(
            w=(
                ["w"],
                mie_ws.wavelen.values,
                dict(long_name="wavelength", units="micron"),
            ),
            r_eff=(
                ["r_eff"],
                mie_ws.reff.values,
                dict(long_name="effective_radius", units="micron"),
            ),
            v_eff=(
                ["v_eff"],
                v_eff,
                dict(long_name="effective_variance", units="micron ** 2"),
            ),
            i=(["i"], list(range(4))),
            j=(["j"], list(range(4))),
            rho=(["rho"], [1.0], dict(long_name="density", units="g/cm^3")),
            alpha=(["v_eff"], [2.0]),
        ),
        data_vars=dict(
            theta_native=(["total_pts"], theta_native),
            phase_native=(["total_pts", "ch16"], phase_native),
            start=(["w", "r_eff", "v_eff"], start),
            n_pts=(["w", "r_eff", "v_eff"], n_pts),
            m_extinction=(
                ["w", "r_eff", "v_eff"],
                mie_ws.ext.values.reshape(nlam, nreff, nveff),
            ),
            albedo=(
                ["w", "r_eff", "v_eff"],
                mie_ws.ssa.values.reshape(nlam, nreff, nveff),
            ),
        ),
    ).squeeze(dim="rho")


def _w_brackets(w_axis, wavelengths):
    il_w = np.clip(
        np.searchsorted(w_axis, wavelengths, side="right") - 1, 0, len(w_axis) - 2
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


def _corners_on_union(corner_thetas, corner_phases, union_theta):
    out = np.empty((len(corner_thetas), 16, len(union_theta)), dtype=np.float32)
    for ci, (th, ph) in enumerate(zip(corner_thetas, corner_phases)):
        il = np.clip(np.searchsorted(th, union_theta, side="right") - 1, 0, len(th) - 2)
        iu = il + 1
        denom = np.where(th[iu] == th[il], 1.0, th[iu] - th[il])
        t = (union_theta - th[il]) / denom
        out[ci] = (ph[il] * (1 - t[:, None]) + ph[iu] * t[:, None]).T
    return out


def _gather_corners(
    w_flat, ir_list, iv_list, start_np, n_pts_np, theta_native, phase_native
):
    corner_thetas, corner_phases = [], []
    for iw in w_flat:
        for ir in ir_list:
            for iv in iv_list:
                s = int(start_np[iw, ir, iv])
                nc = int(n_pts_np[iw, ir, iv])
                corner_thetas.append(theta_native[s : s + nc])
                corner_phases.append(phase_native[s : s + nc])
    return corner_thetas, corner_phases


def _interp_bilinear(ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav, batch_size):
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

    # Group rows by (il_r, il_v) bracket — corners in r/v are shared within a group.
    bracket_key = il_r * nveff + il_v
    sort_idx = np.argsort(bracket_key, kind="stable")
    boundaries = np.flatnonzero(np.diff(bracket_key[sort_idx])) + 1
    group_starts = np.concatenate([[0], boundaries])
    group_ends = np.concatenate([boundaries, [N]])

    # grid_len[row, wav] = size of union theta grid for that (row, wavelength) pair
    grid_len = np.empty((N, n_wav), dtype=np.int32)
    mext_out = np.empty((N, n_wav))
    alb_out = np.empty((N, n_wav))

    # Accumulate per-group results; each group stores one union grid per wavelength.
    group_wav_grids = []  # list of lists: [n_wav][union_theta]
    group_wav_phases = []  # list of (rows, n_wav, ...) phase arrays per wav
    group_rows_list = []

    for gs, ge in zip(group_starts, group_ends):
        rows = sort_idx[gs:ge]
        rep = rows[0]
        ir_pair = [int(il_r[rep]), int(iu_r[rep])]
        iv_pair = [int(il_v[rep]), int(iu_v[rep])]

        corners_mext = ds.m_extinction.values[np.ix_(w_flat, ir_pair, iv_pair)]
        corners_alb = ds.albedo.values[np.ix_(w_flat, ir_pair, iv_pair)]

        # Build per-wavelength union grids: for wavelength index k the two bracket
        # indices into w_flat are [2k, 2k+1].
        wav_union_thetas = []  # length n_wav
        wav_corners_reg = []  # length n_wav, each (8, 16, n_union_k)
        # 8 = 2_w × 2_r × 2_v corners

        for k in range(n_wav):
            wf_k = w_flat[2 * k : 2 * k + 2]  # [il_w_k, iu_w_k]
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
            reg_k = _corners_on_union(ct_k, cp_k, union_theta_k)  # (8, 16, n_union_k)
            wav_corners_reg.append(reg_k)

        # Pre-allocate per-row, per-wavelength phase storage (list of wav arrays)
        # group_phase_wav[k]: (len(rows), 4, 4, n_union_k)
        group_phase_wav = [
            np.empty((len(rows), 4, 4, len(wav_union_thetas[k]))) for k in range(n_wav)
        ]

        for b in range(0, len(rows), batch_size):
            idx = rows[b : b + batch_size]
            B = len(idx)

            cr_b = wr[idx]  # (B, 2)
            cv_b = wv[idx]  # (B, 2)

            for k in range(n_wav):
                cw_k = cw_flat[2 * k : 2 * k + 2]  # (2,) — w bracket weights
                n_union_k = len(wav_union_thetas[k])
                # 8 corners: reshape to (8, 16*n_union_k) for batched matmul
                cp_mat_k = wav_corners_reg[k].reshape(8, 16 * n_union_k)

                # Full weight tensor: (B, 2_w, 2_r, 2_v)
                W_k = (
                    cw_k[np.newaxis, :, np.newaxis, np.newaxis]
                    * cr_b[:, np.newaxis, :, np.newaxis]
                    * cv_b[:, np.newaxis, np.newaxis, :]
                )  # (B, 2, 2, 2)

                # Scalar outputs for this wavelength — slice corners_mext to (2, 2, 2)
                me_k = np.einsum(
                    "bfrs,frs->b", W_k, corners_mext[2 * k : 2 * k + 2], optimize=True
                )
                al_k = np.einsum(
                    "bfrs,frs->b", W_k, corners_alb[2 * k : 2 * k + 2], optimize=True
                )
                mext_out[idx, k] = me_k
                alb_out[idx, k] = al_k

                # Phase: (B, 8) @ (8, 16*n_union_k) → (B, 16*n_union_k)
                W_flat_k = W_k.reshape(B, 8)
                w_rv_k = W_flat_k @ cp_mat_k
                group_phase_wav[k][b : b + B] = w_rv_k.reshape(B, 4, 4, n_union_k)

        # Enforce non-negative P11 per wavelength
        for k in range(n_wav):
            group_phase_wav[k][:, 0, 0, :] = np.maximum(
                group_phase_wav[k][:, 0, 0, :], 0.0
            )
            grid_len[rows, k] = len(wav_union_thetas[k])

        group_wav_grids.append(wav_union_thetas)
        group_wav_phases.append((rows, group_phase_wav))
        group_rows_list.append(rows)

    # Build contiguous output arrays.
    # grid_start[row, k] is the offset into theta/phase for row `row`, wavelength k.
    grid_start = np.zeros((N, n_wav), dtype=np.int64)
    flat_lens = grid_len.ravel()  # (N*n_wav,) row-major
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
            gph_k = gph_wav[k]  # (len(rows), 4, 4, n_union)
            for local_i, row in enumerate(rows):
                s = int(grid_start[row, k])
                theta[s : s + n_union] = gtheta
                phase[s : s + n_union] = (
                    gph_k[local_i].reshape(16, n_union).T
                )  # (n_union, 16)

    return grid_start, grid_len, theta, phase, mext_out, alb_out


def _interp_nearest(ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav):
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
    group_wav_grids = []  # per-group: list of n_wav union theta arrays
    group_wav_phases = []  # per-group: list of n_wav phase arrays (n_wav, 4, 4, n_union_k)
    group_rows_list = []

    for gs, ge in zip(group_starts, group_ends):
        rows = sort_idx[gs:ge]
        ir = int(i_r[rows[0]])
        iv = int(i_v[rows[0]])

        wav_union_thetas = []
        wav_phases = []

        for k in range(n_wav):
            wf_k = w_flat[2 * k : 2 * k + 2]  # [il_w_k, iu_w_k]
            cw_k = cw_flat[2 * k : 2 * k + 2]  # [w_lo, w_hi]

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
            reg_k = _corners_on_union(ct_k, cp_k, union_theta_k)  # (2, 16, n_union_k)

            # Wavelength-interpolated phase: weighted sum over the 2 w-brackets
            phase_k = cw_k[0] * reg_k[0] + cw_k[1] * reg_k[1]  # (16, n_union_k)
            phase_k = phase_k.reshape(4, 4, len(union_theta_k))
            phase_k[0, 0, :] = np.maximum(phase_k[0, 0, :], 0.0)

            # Scalar interpolation for this wavelength
            corners_mext_k = ds.m_extinction.values[wf_k, ir, iv]  # (2,)
            corners_alb_k = ds.albedo.values[wf_k, ir, iv]  # (2,)
            mext_out[rows, k] = (
                cw_k[0] * corners_mext_k[0] + cw_k[1] * corners_mext_k[1]
            )
            alb_out[rows, k] = cw_k[0] * corners_alb_k[0] + cw_k[1] * corners_alb_k[1]

            wav_union_thetas.append(union_theta_k)
            wav_phases.append(phase_k)

        group_wav_grids.append(wav_union_thetas)
        group_wav_phases.append(wav_phases)
        group_rows_list.append(rows)

    # Build deduplicated contiguous store: one slot per (group, k).
    # All rows in the same group share the same flat offset — grid_start[row, k]
    # is identical for every row in the group, so the phase data is written once.
    n_groups = len(group_wav_grids)

    # canonical_starts[g, k]: flat offset for group g, wavelength k
    canon_lens = np.array(
        [[len(group_wav_grids[g][k]) for k in range(n_wav)] for g in range(n_groups)],
        dtype=np.int32,
    )  # (n_groups, n_wav)
    canon_flat = canon_lens.ravel()
    canon_starts = np.zeros(n_groups * n_wav, dtype=np.int64)
    canon_starts[1:] = np.cumsum(canon_flat[:-1])
    canon_starts = canon_starts.reshape(n_groups, n_wav)

    total_pts = int(canon_flat.sum())
    theta = np.empty(total_pts, dtype=np.float64)
    phase = np.empty((total_pts, 16), dtype=np.float32)

    for g, (wav_grids, wav_phases) in enumerate(zip(group_wav_grids, group_wav_phases)):
        for k in range(n_wav):
            s = int(canon_starts[g, k])
            gtheta = wav_grids[k]
            n_union = len(gtheta)
            theta[s : s + n_union] = gtheta
            phase[s : s + n_union] = (
                wav_phases[k].reshape(16, n_union).T
            )  # (n_union, 16)

    # Broadcast canonical offsets/lengths to every row — rows in the same group
    # get the same start/len values, pointing at the single shared flat slot.
    # This level of redundancy in the index only is considered acceptable.
    grid_start = np.empty((N, n_wav), dtype=np.int64)
    grid_len = np.empty((N, n_wav), dtype=np.int32)
    for g, rows in enumerate(group_rows_list):
        grid_start[rows] = canon_starts[g]  # (n_wav,) broadcast over rows
        grid_len[rows] = canon_lens[g]

    return grid_start, grid_len, theta, phase, mext_out, alb_out


def interpolate_cloudparticles_profile(
    ds, cumulus_profile, wavelengths, batch_size=16, rv_mode="bilinear"
):
    wavelengths = np.atleast_1d(wavelengths)
    n_wav = len(wavelengths)

    w_axis_raw = (ds.w.values * ureg.Unit(ds.w.attrs["units"])).m_as(
        ds.w.attrs["units"]
    )
    wav_raw = wavelengths.m_as(ds.w.attrs["units"])
    w_flat, cw_flat = _w_brackets(w_axis_raw, wav_raw)

    r_eff_all = cumulus_profile.r_eff.values
    v_eff_all = cumulus_profile.v_eff.values
    N = len(r_eff_all)

    if rv_mode == "bilinear":
        grid_start, grid_len, theta, phase, mext, alb = _interp_bilinear(
            ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav, batch_size
        )
    elif rv_mode == "nearest":
        grid_start, grid_len, theta, phase, mext, alb = _interp_nearest(
            ds, r_eff_all, v_eff_all, w_flat, cw_flat, n_wav
        )
    else:
        raise ValueError(f"Unknown rv_mode {rv_mode!r}. Use 'bilinear' or 'nearest'.")

    return xr.Dataset(
        coords=dict(
            index=(["index"], np.arange(N)),
            w=(["w"], wav_raw),
            rho=float(ds.rho.values),
            alpha=float(ds.alpha.values.mean()),
        ),
        data_vars=dict(
            # theta/phase are a flat ragged store; grid_start[row, k] and
            # grid_len[row, k] index into them per (row, wavelength) pair.
            theta=(["total_pts"], theta),
            phase=(["total_pts", "ch16"], phase),
            grid_start=(["index", "w"], grid_start),
            grid_len=(["index", "w"], grid_len),
            m_extinction=(["index", "w"], mext),
            albedo=(["index", "w"], alb),
        ),
    )


@define()
class CloudPhaseFunction(PhaseFunction):
    # TBD improve this API
    arrays = attrs.field(kw_only=True)
    index = attrs.field(kw_only=True)
    grid = attrs.field(kw_only=True)

    def _tensors(self, ctx):
        ds = self.arrays(ctx)
        # TBD support non spherical particles
        p = ds.phase.values[:, [0, 1, 5, 10, 11, 15]].astype(
            np.float32
        )  # (total_pts, 6)
        mueller = np.ascontiguousarray(p).flatten()  # actual explicit copy
        nodes = np.cos(np.deg2rad(180 - ds.theta.values)).astype(
            np.float32
        )  # flip convention to Mitsuba's

        w_idx = 0
        grid_start = ds.grid_start.values[:, w_idx].astype(np.int32)
        grid_len = ds.grid_len.values[:, w_idx].astype(np.int32)

        return {
            "n_entries": int(len(grid_start)),
            # TBD: non scalar variants
            "nodes": dr.scalar.ArrayXf64(nodes),
            "phase_mueller": dr.scalar.ArrayXf64(mueller),
            "grid_start": dr.scalar.ArrayXi(grid_start),
            "grid_len": dr.scalar.ArrayXi(grid_len),
        }

    @property
    def template(self):
        return {
            "type": "cloudphase",
            "index_volume": {
                "type": "gridvolume",
                "grid": DictParameter(
                    lambda ctx: mi.VolumeGrid(self.index(ctx).astype(np.float32).T)
                ),
                "filter_type": "nearest",
                "to_world": self.grid.to_world,
                "wrap_mode": "repeat",
            },
            "n_entries": DictParameter(lambda ctx: self._tensors(ctx)["n_entries"]),
            "nodes": DictParameter(lambda ctx: self._tensors(ctx)["nodes"]),
            "phase_mueller": DictParameter(
                lambda ctx: self._tensors(ctx)["phase_mueller"]
            ),
            "grid_start": DictParameter(lambda ctx: self._tensors(ctx)["grid_start"]),
            "grid_len": DictParameter(lambda ctx: self._tensors(ctx)["grid_len"]),
        }

    @property
    def params(self):
        return {
            "index_volume.grid": SceneParameter(
                lambda ctx: mi.VolumeGrid(self.index(ctx).astype(np.float32)),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "n_entries": SceneParameter(
                lambda ctx: self._tensors(ctx)["n_entries"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "nodes": SceneParameter(
                lambda ctx: self._tensors(ctx)["nodes"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "phase_mueller": SceneParameter(
                lambda ctx: self._tensors(ctx)["phase_mueller"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_start": SceneParameter(
                lambda ctx: self._tensors(ctx)["grid_start"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_len": SceneParameter(
                lambda ctx: self._tensors(ctx)["grid_len"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
        }

    def __hash__(self):
        return id(self)  # TBD check if still necessary?
