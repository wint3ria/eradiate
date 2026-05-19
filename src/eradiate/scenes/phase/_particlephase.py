from __future__ import annotations

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np

from ._core import PhaseFunction
from ...attrs import define, documented
from ...kernel import DictParameter, KernelSceneParameterFlags, SceneParameter
from ...util.misc import cache_by_id


@define(eq=False, slots=False)
class ParticlePhase(PhaseFunction):
    """
    Phase function for spatially heterogeneous cloud particle fields.

    Wraps the ``particlephase`` kernel plugin, which performs bilinear
    interpolation of the phase matrix in the ``(r_eff, v_eff)`` parameter
    space at each scattering interaction.
    """

    r_eff_volume: np.ndarray = documented(
        attrs.field(kw_only=True),
        doc="Dense ``(n_z, n_y, n_x)`` array of effective radius values "
        "on the render grid. NaN marks empty cells.",
        type="ndarray",
    )

    v_eff_volume: np.ndarray = documented(
        attrs.field(kw_only=True),
        doc="Dense ``(n_z, n_y, n_x)`` array of effective variance values "
        "on the render grid. NaN marks empty cells.",
        type="ndarray",
    )

    r_eff_grid: np.ndarray = documented(
        attrs.field(kw_only=True),
        doc="Sorted 1-D array of effective radius values in the properties dataset.",
        type="ndarray",
    )

    v_eff_grid: np.ndarray = documented(
        attrs.field(kw_only=True),
        doc="Sorted 1-D array of effective variance values in the properties dataset.",
        type="ndarray",
    )

    phase_data: object = documented(
        attrs.field(kw_only=True),
        doc="Callable ``(ctx) -> xr.Dataset`` returning the interpolated "
        "properties dataset at the current spectral index.",
        type="callable",
    )

    grid: object = documented(
        attrs.field(kw_only=True),
        doc=":class:`.GridCoords` instance defining the spatial extent of "
        "the render grid.",
        type="GridCoords",
    )

    wrap_mode: str = documented(
        attrs.field(default="clamp", kw_only=True),
        doc="Phase reff and veff wrap_mode. "
        'Either ``"clamp"``, ``"repeat"`` or ``"mirror"``.',
        type="str",
        default='"clamp"',
    )

    filter_type: str = documented(
        attrs.field(default="trilinear", kw_only=True),
        doc='Phase reff and veff filter_type. Either ``"trilinear"`` or ``"nearest"``.',
        type="str",
        default='"trilinear"',
    )

    blending_method: str = documented(
        attrs.field(default="blended_cdf", kw_only=True),
        doc="Phase blending method passed to the kernel plugin. "
        'Either ``"blended_cdf"`` or ``"stochastic"``.',
        type="str",
        default='"blended_cdf"',
    )

    @cache_by_id
    def _build_phase_parameters(self, ctx: object) -> dict:
        ds = self.phase_data(ctx)

        w_idx = 0
        n_r = ds.sizes["r_eff"]
        n_v = ds.sizes["v_eff"]

        nodes_raw = np.cos(np.deg2rad(ds.theta_native.values)).astype(np.float64)

        mueller_raw = ds.phase_native.values[:, [0, 1, 5, 10, 11, 15]].astype(
            np.float32
        )

        grid_start = ds.start.values[w_idx].flatten().astype(np.uint32)
        grid_len = ds.n_pts.values[w_idx].flatten().astype(np.uint32)

        return {
            "n_r": n_r,
            "n_v": n_v,
            "r_eff_grid": ds.r_eff.values.astype(np.float32),
            "v_eff_grid": ds.v_eff.values.astype(np.float32),
            "nodes": dr.scalar.ArrayXf64(nodes_raw),
            "phase_mueller": dr.scalar.ArrayXf64(mueller_raw.flatten()),
            "grid_start": grid_start,
            "grid_len": grid_len,
        }

    @property
    def template(self) -> dict:
        return {
            "type": "particlephase",
            "r_eff_volume": {
                "type": "gridvolume",
                "grid": mi.VolumeGrid(
                    self.r_eff_volume.astype(np.float32).T[..., np.newaxis]
                ),
                "wrap_mode": self.wrap_mode,  # TBD improve config
                "use_grid_bbox": True,
                "filter_type": self.filter_type,
                "to_world": self.grid.to_world,
            },
            "v_eff_volume": {
                "type": "gridvolume",
                "grid": mi.VolumeGrid(
                    self.v_eff_volume.astype(np.float32).T[..., np.newaxis]
                ),
                "wrap_mode": self.wrap_mode,  # TBD improve config
                "use_grid_bbox": True,
                "filter_type": self.filter_type,
                "to_world": self.grid.to_world,
            },
            "r_eff_grid": mi.VolumeGrid(
                self.r_eff_grid.astype(np.float32).reshape(1, 1, -1, 1)
            ),
            "v_eff_grid": mi.VolumeGrid(
                self.v_eff_grid.astype(np.float32).reshape(1, 1, -1, 1)
            ),
            "n_r": len(self.r_eff_grid),
            "n_v": len(self.v_eff_grid),
            "nodes": DictParameter(
                lambda ctx: self._build_phase_parameters(ctx)["nodes"]
            ),
            "phase_mueller": DictParameter(
                lambda ctx: self._build_phase_parameters(ctx)["phase_mueller"]
            ),
            "grid_start": DictParameter(
                lambda ctx: dr.scalar.ArrayXu(
                    self._build_phase_parameters(ctx)["grid_start"]
                )
            ),
            "grid_len": DictParameter(
                lambda ctx: dr.scalar.ArrayXu(
                    self._build_phase_parameters(ctx)["grid_len"]
                )
            ),
            "blending_method": self.blending_method,
        }

    @property
    def params(self) -> dict[str, SceneParameter]:
        return {
            "nodes": SceneParameter(
                lambda ctx: self._build_phase_parameters(ctx)["nodes"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "phase_mueller": SceneParameter(
                lambda ctx: self._build_phase_parameters(ctx)["phase_mueller"],
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_start": SceneParameter(
                lambda ctx: dr.scalar.ArrayXu(
                    self._build_phase_parameters(ctx)["grid_start"]
                ),
                KernelSceneParameterFlags.SPECTRAL,
            ),
            "grid_len": SceneParameter(
                lambda ctx: dr.scalar.ArrayXu(
                    self._build_phase_parameters(ctx)["grid_len"]
                ),
                KernelSceneParameterFlags.SPECTRAL,
            ),
        }

    def __hash__(self) -> int:
        return id(self)
