from __future__ import annotations

from abc import ABC
from functools import partial, singledispatchmethod
from typing import Callable

import attrs
import mitsuba as mi
import numpy as np
import pint

from .attrs import define, documented
from .contexts import KernelContext
from .kernel import (
    DictParameter,
    SceneParameter,
    SearchSceneParameter,
)
from .scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
    XYGrid,
)


def make_volume_grid_plane_parallel(
    ctx: KernelContext,
    eval_grid=None,
    shape_x=None,
    shape_y=None,
    shape_z=None,
    permissive=None,
    dtype=None,
    unit=None,
    si_mode=False,
    to_mi=True,
    extra_dim=False,
) -> np.ndarray | mi.VolumeGrid:
    grid = eval_grid(ctx.si if si_mode else ctx)
    if unit:
        grid = grid.m_as(unit)
    assert np.size(grid) > 0
    shape = (shape_z, shape_y, shape_x)
    grid: np.ndarray = np.asarray(grid, dtype=dtype)
    if grid.size == 1:
        grid = np.full(shape, grid, dtype=dtype)
    elif np.squeeze(grid).shape == (shape_z,):
        grid = np.repeat(grid.reshape(shape_z, 1), shape_y, axis=1)
        grid = np.repeat(grid.reshape(shape_z, shape_y, 1), shape_x, axis=2)
    elif not permissive and grid.shape == shape:
        raise ValueError(
            f"Invalid grid shape, expected {tuple(reversed(shape))}, got {grid.shape}"
        )
    elif grid.shape != tuple(reversed(shape)):
        raise ValueError(
            f"Invalid grid shape, expected {tuple(reversed(shape))}, got {grid.shape}"
        )
    else:
        grid = grid.T
    if extra_dim:
        grid = grid.reshape(*grid.shape, 1)
    if to_mi:
        return mi.VolumeGrid(grid)
    return grid


def make_volume_grid_spherical_shell(
    ctx: KernelContext,
    eval_grid=None,
    shape_x=None,
    shape_y=None,
    shape_z=None,
    permissive=None,
    dtype=None,
    unit=None,
    si_mode=False,
    to_mi=True,
    extra_dim=False,
) -> np.ndarray | mi.VolumeGrid:
    grid = eval_grid(ctx.si if si_mode else ctx)
    if unit:
        grid = grid.m_as(unit)
    assert np.size(grid) > 0
    shape = (shape_z, shape_y, shape_x)
    grid: np.ndarray = np.asarray(grid, dtype=dtype)
    if grid.size == 1:
        grid = np.full(shape, grid, dtype=dtype)
    elif np.squeeze(grid).shape == (shape_z,):
        grid = np.repeat(grid.reshape(1, shape_z), shape_y, axis=0)
        grid = np.repeat(grid.reshape(1, shape_y, shape_z), shape_x, axis=0)
    elif not permissive and grid.shape == tuple(reversed(shape)):
        raise ValueError(f"Invalid grid shape, expected {shape}, got {grid.shape}")
    elif grid.shape != shape:
        raise ValueError(f"Invalid grid shape, expected {shape}, got {grid.shape}")
    if extra_dim:
        grid = grid.reshape(*grid.shape, 1)
    if to_mi:
        return mi.VolumeGrid(grid)
    return grid


class _partial(partial):
    def __repr__(self):
        first = next(iter(self.keywords.values()), None)
        if hasattr(first, "__self__"):
            cls_name = first.__self__.__class__.__name__
            return f"partial({self.func.__name__}, {cls_name}.{first.__name__}, extra_parameters={list(self.keywords)})"
        return f"partial({self.func.__name__}, extra_parameters={list(self.keywords)})"

    def get_bound_instance(self):
        first = next(iter(self.keywords.values()), None)
        if hasattr(first, "__self__"):
            return first.__self__
        else:
            return first


@define(eq=False)
class VolumeGridFactory(ABC):
    dtype: np.dtype = documented(
        attrs.field(
            default=np.float32,
            converter=np.dtype,
            validator=attrs.validators.instance_of(np.dtype),
        ),
        doc="Data type of the buffer",
        type="np.dtype",
        init_type="np.dtype or str",
        default="np.float32",
    )

    permissive: bool = documented(
        attrs.field(
            default=False,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Allow more compatible input shapes to be converted to the volume shape",
        type="bool",
        init_type="bool",
        default="False",
    )

    si_mode: bool = documented(
        attrs.field(
            default=False,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Evaluate the grid function using the context spectral index instead",
        type="bool",
        init_type="bool",
        default="False",
    )

    unit: pint.Unit | None = documented(
        attrs.field(
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(pint.Unit)
            ),
        ),
        doc="Perform a unit conversion before interpreting the grid values",
        type="pint.Unit or None",
        init_type="pint.Unit or None",
        default="None",
    )

    def xy_shape(geometry: Any):
        if isinstance(geometry, XYGrid):
            return geometry.xy_resolution
        return (1, 1)

    def _partial_factory(self, geometry: SceneGeometry):
        if isinstance(geometry, PlaneParallelGeometry):
            return make_volume_grid_plane_parallel
        elif isinstance(geometry, SphericalShellGeometry):
            return make_volume_grid_spherical_shell
        else:
            raise NotImplementedError(
                f"Geometric type {type(geometry)} is not supported by VolumeGridFactory"
            )

    @singledispatchmethod
    def _postprocess_template(
        self,
        geometry: Any,
        partial_factory: Callable,
        filter_type_kw: str,
        wrap_mode_kw: str,
    ):
        raise NotImplementedError(
            f"Geometric type {type(geometry)} is not supported by VolumeGridFactory"
        )

    @_postprocess_template.register
    def _(
        self,
        geometry: PlaneParallelGeometry,
        partial_factory: Callable,
        filter_type_kw: str,
        wrap_mode_kw: str,
    ):
        return {
            "type": "gridvolume",
            "grid": DictParameter(partial_factory),
            "filter_type": filter_type_kw,
            "wrap_mode": wrap_mode_kw,
            "to_world": geometry.atmosphere_volume_to_world,
        }

    @_postprocess_template.register
    def _(
        self,
        geometry: SphericalShellGeometry,
        partial_factory: Callable,
        filter_type_kw: str,
        wrap_mode_kw: str,
    ):
        volume_rmin = geometry.atmosphere_volume_rmin

        return {
            "type": "sphericalcoordsvolume",
            "volume": {
                "type": "gridvolume",
                "grid": DictParameter(partial_factory),
                "filter_type": filter_type_kw,
                "wrap_mode": wrap_mode_kw,
            },
            "to_world": geometry.atmosphere_volume_to_world,
            "rmin": volume_rmin,
        }

    @singledispatchmethod
    def _postprocess_params(
        self, geometry: Any, partial_factory: Callable, flags, **search_kwargs
    ):
        raise NotImplementedError(
            f"Geometric type {type(geometry)} is not supported by VolumeGridFactory"
        )

    @_postprocess_params.register
    def _(
        self, geometry: PlaneParallelGeometry, partial_factory, flags, **search_kwargs
    ):
        return SceneParameter(
            partial_factory, flags, search=SearchSceneParameter(**search_kwargs)
        )

    def generate_template(
        self,
        geometry: SceneGeometry,
        eval_grid: Callable,
        filter_type=None,
        wrap_mode=None,
        si_mode=None,
        unit=None,
    ) -> dict:
        filter_type = filter_type or str(geometry.filter_type)
        wrap_mode = wrap_mode or str(geometry.wrap_mode)
        n_layers = geometry.zgrid.n_layers
        shape_x, shape_y = VolumeGridFactory.xy_shape(geometry)
        si_mode = si_mode or self.si_mode
        unit = unit or self.unit

        factory = self._partial_factory(geometry)

        partial_factory = _partial(
            factory,
            eval_grid=eval_grid,
            shape_x=shape_x,
            shape_y=shape_y,
            shape_z=n_layers,
            permissive=self.permissive,
            dtype=self.dtype,
            unit=unit,
            si_mode=si_mode,
        )

        return self._postprocess_template(
            geometry, partial_factory, filter_type, wrap_mode
        )

    def generate_params(
        self,
        geometry: SceneGeometry,
        eval_grid: Callable,
        flag,
        search_kwargs,
        si_mode=None,
        unit=None,
    ) -> SceneParameter:
        n_layers = geometry.zgrid.n_layers
        shape_x, shape_y = VolumeGridFactory.xy_shape(geometry)
        si_mode = si_mode or self.si_mode
        unit = unit or self.unit

        factory = self._partial_factory(geometry)

        partial_factory = partial(
            factory,
            eval_grid=eval_grid,
            shape_x=shape_x,
            shape_y=shape_y,
            shape_z=n_layers,
            permissive=self.permissive,
            dtype=self.dtype,
            unit=unit,
            si_mode=si_mode,
            extra_dim=True,
            to_mi=False,
        )

        return SceneParameter(
            partial_factory, flag, search=SearchSceneParameter(**search_kwargs)
        )
